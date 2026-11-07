"""Transformer backbone for encoding observed temporal features."""

import math

import torch
from torch import Tensor, nn

from .attention import ProjectedAttention
from .distillation import AttentionDistillation
from .masking import compact_padding, require_right_padding, valid_mask, zero_invalid


class SinusoidalPositionEncoding(nn.Module):
    """On-demand history positions, including odd hidden dimensions."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.d_model = d_model

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> Tensor:
        valid = valid_mask(x, time_valid, "position encoding input")
        if x.shape[-1] != self.d_model:
            raise ValueError("position encoding dimension must equal d_model")
        positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0).float()
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions.unsqueeze(-1) * frequencies
        encoding = torch.zeros(*x.shape[:2], self.d_model, device=x.device, dtype=torch.float32)
        encoding[..., 0::2] = angles.sin()
        encoding[..., 1::2] = angles[..., : self.d_model // 2].cos()
        return zero_invalid(zero_invalid(x, valid) + encoding.to(dtype=x.dtype), valid)


class HistoryEmbedding(nn.Module):
    """A standalone observed-feature projection with history position encoding.

    Inputs may have left/right padding. Returned features/mask use canonical
    right padding; normalization and coordinate conventions belong to callers.
    """

    def __init__(self, input_dim: int = 6, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if min(input_dim, d_model) < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid embedding dimensions or dropout")
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, d_model)
        self.position_encoding = SinusoidalPositionEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history: Tensor, time_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        valid = valid_mask(history, time_valid, "history")
        if not history.is_floating_point() or history.shape[-1] != self.input_dim:
            raise ValueError("history must be floating point with input_dim features")
        values = history.masked_select(valid.unsqueeze(-1).expand_as(history))
        if not bool(torch.isfinite(values).all()):
            raise ValueError("valid history values must be finite")
        history, valid = compact_padding(history, valid)
        embedded = zero_invalid(self.projection(history), valid)
        embedded = self.dropout(self.position_encoding(embedded, valid))
        return zero_invalid(embedded, valid), valid


class GlobalEncoderLayer(nn.Module):
    """Pre-norm self-attention and feed-forward processing of observed history."""

    def __init__(
        self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
        dropout: float = 0.1, attention: str = "probsparse",
        sparse_factor: float = 3.0, sparse_threshold: int = 64,
    ) -> None:
        super().__init__()
        if d_ff < 1:
            raise ValueError("d_ff must be positive")
        self.attention = ProjectedAttention(d_model, n_heads, dropout, attention, sparse_factor, sparse_threshold)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> Tensor:
        valid = valid_mask(x, time_valid, "encoder input")
        normalized = zero_invalid(self.norm1(zero_invalid(x, valid)), valid)
        attended = self.attention(normalized, normalized, normalized, valid, valid)
        x = zero_invalid(x, valid) + self.dropout(attended)
        x = zero_invalid(x + self.dropout(self.feed_forward(self.norm2(x))), valid)
        return x


class GlobalEncoder(nn.Module):
    """Feature-level history encoder, optionally distilled between layers.

    Attention is bidirectional within the observed history.
    Canonical right padding is required so downsampling has stable alignment.
    """

    def __init__(
        self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
        n_layers: int = 2, dropout: float = 0.1,
        attention: str = "probsparse", sparse_factor: float = 3.0,
        sparse_threshold: int = 64, distill: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers < 1:
            raise ValueError("n_layers must be a positive integer")
        self.layers = nn.ModuleList(
            GlobalEncoderLayer(d_model, n_heads, d_ff, dropout, attention, sparse_factor, sparse_threshold)
            for _ in range(n_layers)
        )
        self.distillers = nn.ModuleList(AttentionDistillation(d_model) for _ in range(n_layers - 1)) if distill else nn.ModuleList()
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def attention_modes(self) -> tuple[str, ...]:
        return tuple(layer.attention.last_mode for layer in self.layers)

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        valid = valid_mask(x, time_valid, "encoder input")
        require_right_padding(valid)
        x = zero_invalid(x, valid)
        for index, layer in enumerate(self.layers):
            x = layer(x, valid)
            if index < len(self.distillers):
                x, valid = self.distillers[index](x, valid)
        return zero_invalid(self.final_norm(x), valid), valid


class TransformerBackbone(nn.Module):
    """Reusable history embedding and global feature backbone.

    ``forward(history[B,T,F], time_valid=None)`` returns
    ``(features[B,Tg,D], valid[B,Tg])``. With n_layers=2 and distill=True,
    Tg=ceil(T/2); per-sample valid lengths are propagated separately.
    """

    def __init__(
        self, input_dim: int = 6, d_model: int = 64, n_heads: int = 4,
        d_ff: int = 128, n_layers: int = 2, dropout: float = 0.1,
        attention: str = "probsparse", sparse_factor: float = 3.0,
        sparse_threshold: int = 64, distill: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = HistoryEmbedding(input_dim, d_model, dropout)
        self.encoder = GlobalEncoder(d_model, n_heads, d_ff, n_layers, dropout, attention, sparse_factor, sparse_threshold, distill)

    @property
    def attention_modes(self) -> tuple[str, ...]:
        return self.encoder.attention_modes

    def forward(self, history: Tensor, time_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features, valid = self.embedding(history, time_valid)
        return self.encoder(features, valid)
