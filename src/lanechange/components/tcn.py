"""Causal local temporal features with learnable fusion across receptive fields."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .masking import valid_mask, zero_invalid


class CausalTemporalBlock(nn.Module):
    """Two left-padded convolutions; normalization never mixes timesteps."""

    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        if min(d_model, kernel_size, dilation) < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid convolution dimensions or dropout")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _convolve(self, x: Tensor, conv: nn.Conv1d) -> Tensor:
        return conv(F.pad(x.transpose(1, 2), (self.left_padding, 0))).transpose(1, 2)

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> Tensor:
        valid = valid_mask(x, time_valid, "TCN input")
        x = zero_invalid(x, valid)
        hidden = self.dropout(F.gelu(self.norm1(self._convolve(x, self.conv1))))
        hidden = zero_invalid(hidden, valid)
        hidden = self.dropout(F.gelu(self.norm2(self._convolve(hidden, self.conv2))))
        hidden = zero_invalid(hidden, valid)
        return zero_invalid(self.output_norm(x + hidden), valid)


class MultiScaleCausalTCN(nn.Module):
    """Independent causal branches with normalized trainable scale weights.

    A branch with dilation d has receptive field 1 + 2*(kernel_size-1)*d.
    Inputs/outputs are [B,T,D]; the caller supplies feature projection if needed.
    """

    def __init__(
        self, d_model: int = 64, dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dilations = tuple(dilations)
        if not dilations or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in dilations):
            raise ValueError("dilations must contain positive integers")
        self.d_model = d_model
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.branches = nn.ModuleList(CausalTemporalBlock(d_model, kernel_size, d, dropout) for d in dilations)
        self.scale_logits = nn.Parameter(torch.zeros(len(dilations)))

    @property
    def scale_weights(self) -> Tensor:
        return self.scale_logits.softmax(dim=0)

    @property
    def receptive_fields(self) -> tuple[int, ...]:
        return tuple(1 + 2 * (self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> Tensor:
        valid = valid_mask(x, time_valid, "TCN input")
        if x.shape[-1] != self.d_model:
            raise ValueError("TCN input feature dimension must equal d_model")
        branches = torch.stack([branch(x, valid) for branch in self.branches], dim=1)
        result = (branches * self.scale_weights.view(1, -1, 1, 1)).sum(dim=1)
        return zero_invalid(result, valid)
