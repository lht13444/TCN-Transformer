"""Shared padding utilities; True always means an observed timestep."""

import torch
from torch import Tensor


def valid_mask(x: Tensor, valid: Tensor | None = None, name: str = "input") -> Tensor:
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"{name} must be nonempty [batch, time, features]")
    if valid is None:
        return torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    if valid.dtype != torch.bool or tuple(valid.shape) != tuple(x.shape[:2]):
        raise ValueError(f"{name} validity mask must be bool [batch, time]")
    return valid.to(device=x.device)


def zero_invalid(x: Tensor, valid: Tensor) -> Tensor:
    """Remove even NaN/Inf padding; multiplying by zero would not remove NaN."""
    return x.masked_fill(~valid.unsqueeze(-1), 0.0)


def require_observations(valid: Tensor) -> None:
    if bool((~valid.any(dim=1)).any()):
        raise ValueError("each sample requires at least one valid timestep")


def require_right_padding(valid: Tensor) -> None:
    require_observations(valid)
    expected = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid.sum(1)[:, None]
    if not bool((valid == expected).all()):
        raise ValueError("this component requires canonical right padding")


def compact_padding(history: Tensor, valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Move contiguous observed history before padding without reordering it.

    The mask describes padding rather than missing physical time. Interior gaps
    need history-only imputation before this function is called.
    """
    valid = valid_mask(history, valid, "history")
    require_observations(valid)
    lengths = valid.sum(dim=1)
    positions = torch.arange(history.shape[1], device=history.device)
    first = positions.expand_as(valid).masked_fill(~valid, history.shape[1]).min(1).values
    last = positions.expand_as(valid).masked_fill(~valid, -1).max(1).values
    if bool(((last - first + 1) != lengths).any()):
        raise ValueError("time_valid describes padding only; impute interior time gaps first")
    order = torch.argsort((~valid).long(), dim=1, stable=True)
    clean = zero_invalid(history, valid).gather(1, order.unsqueeze(-1).expand_as(history))
    packed_valid = positions[None, :] < lengths[:, None]
    return zero_invalid(clean, packed_valid), packed_valid
