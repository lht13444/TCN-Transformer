"""Direct trajectory-offset prediction from temporal feature sequences."""

import math

import torch
from torch import Tensor, nn

from .masking import require_observations, valid_mask, zero_invalid


class TrajectoryPredictionHead(nn.Module):
    """Pool history features and predict all future position offsets directly.

    ``features`` has shape [B,T,D]. A boolean ``time_valid`` mask uses True for
    observed timesteps and may contain left/right padding or masked positions.
    The output shape is [B,horizon,output_dim]. Each output is an offset from
    the last observed coordinate origin, rather than an increment from the
    preceding future step. No cumulative sum or coordinate conversion is used.
    Target units and the absolute coordinate origin are supplied by the caller.
    """

    def __init__(
        self, d_model: int = 64, horizon: int = 40, output_dim: int = 2,
        hidden_dim: int = 128, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = {
            "d_model": d_model, "horizon": horizon,
            "output_dim": output_dim, "hidden_dim": hidden_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.d_model = d_model
        self.horizon = horizon
        self.output_dim = output_dim
        self.pool_norm = nn.LayerNorm(2 * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, features: Tensor, time_valid: Tensor | None = None) -> Tensor:
        valid = valid_mask(features, time_valid, "head features")
        if not features.is_floating_point() or features.shape[-1] != self.d_model:
            raise ValueError("features must be floating point with d_model channels")
        require_observations(valid)
        observed_values = features.masked_select(valid.unsqueeze(-1).expand_as(features))
        if not bool(torch.isfinite(observed_values).all()):
            raise ValueError("valid feature values must be finite")
        clean = zero_invalid(features, valid)
        count = valid.sum(dim=1, keepdim=True).to(dtype=features.dtype)
        mean = clean.sum(dim=1) / count
        # Index the last observed token, including when padding is on the left.
        positions = torch.arange(features.shape[1], device=features.device)[None, :].expand_as(valid)
        last_indices = positions.masked_fill(~valid, -1).max(dim=1).values
        batch_indices = torch.arange(features.shape[0], device=features.device)
        last = clean[batch_indices, last_indices]
        pooled = self.pool_norm(torch.cat((mean, last), dim=-1))
        return self.mlp(pooled).reshape(features.shape[0], self.horizon, self.output_dim)
