"""Cross-attention and additive fusion of local/global temporal features."""

import torch
from torch import Tensor, nn

from .attention import ProjectedAttention
from .masking import require_right_padding, valid_mask, zero_invalid


class LocalGlobalFusion(nn.Module):
    """Local queries read global context through a residual connection.

    Inputs are independent feature tensors [B,Tlocal,D] / [B,Tglobal,D].
    ``cross`` returns (LayerNorm(local + Dropout(cross_context)), None).
    ``gated_cross`` is an optional engineering extension with a learned gate;
    ``sum`` is an additive feature baseline. This is one-way cross-attention.
    Both input masks use canonical right padding. Cross-attention explicitly
    uses dense scores of size Tlocal*Tglobal.
    """

    def __init__(
        self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1,
        mode: str = "cross",
    ) -> None:
        super().__init__()
        if mode not in {"cross", "gated_cross", "sum"}:
            raise ValueError("mode must be cross, gated_cross, or sum")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.mode = mode
        self.d_model = d_model
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cross_attention = ProjectedAttention(d_model, n_heads, dropout, "dense") if mode in {"cross", "gated_cross"} else None
        self.gate = nn.Linear(2 * d_model, d_model) if mode == "gated_cross" else None

    def forward(
        self, local: Tensor, global_features: Tensor,
        local_valid: Tensor | None = None, global_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        local_valid = valid_mask(local, local_valid, "local features")
        global_valid = valid_mask(global_features, global_valid, "global features")
        require_right_padding(local_valid)
        require_right_padding(global_valid)
        if local.shape[0] != global_features.shape[0] or local.shape[-1] != self.d_model or global_features.shape[-1] != self.d_model:
            raise ValueError("feature batches/dimensions do not match the fusion module")
        local = zero_invalid(local, local_valid)
        global_features = zero_invalid(global_features, global_valid)
        if self.mode in {"cross", "gated_cross"}:
            assert self.cross_attention is not None
            context = self.cross_attention(local, global_features, global_features, local_valid, global_valid)
            if self.mode == "gated_cross":
                assert self.gate is not None
                gate = zero_invalid(torch.sigmoid(self.gate(torch.cat((local, context), dim=-1))), local_valid)
                result = local + gate * self.dropout(context)
            else:
                result, gate = local + self.dropout(context), None
        else:
            local_lengths, global_lengths = local_valid.sum(1), global_valid.sum(1)
            positions = torch.arange(local.shape[1], device=local.device)[None, :]
            indices = positions * global_lengths[:, None] // local_lengths[:, None]
            indices = torch.minimum(indices, global_lengths[:, None] - 1)
            context = global_features.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.d_model))
            result, gate = local + self.dropout(context), None
        return zero_invalid(self.output_norm(result), local_valid), gate
