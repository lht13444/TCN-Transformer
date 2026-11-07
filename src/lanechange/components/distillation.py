"""History-feature compression with explicit padding propagation."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .masking import require_right_padding, valid_mask, zero_invalid


class AttentionDistillation(nn.Module):
    """Conv1d / ELU / masked max-pool, reducing each sample to ceil(Tvalid/2).

    The input mask must use canonical right padding. ``compact_padding`` is
    available to prepare standalone inputs. Output is (features, valid_mask),
    both with temporal dimension ceil(T/2). No running batch statistics exist.
    """

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        valid = valid_mask(x, time_valid, "distillation input")
        require_right_padding(valid)
        x = zero_invalid(x, valid)
        hidden = F.elu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        pooled = F.max_pool1d(hidden.transpose(1, 2), kernel_size=3, stride=2, padding=1).transpose(1, 2)
        # Restrict to the same output length an unpadded sample would have;
        # merely OR-ing pooling-window masks would create extra tail tokens.
        new_lengths = (valid.sum(dim=1) + 1) // 2
        new_valid = torch.arange(pooled.shape[1], device=x.device)[None, :] < new_lengths[:, None]
        return zero_invalid(pooled, new_valid), new_valid
