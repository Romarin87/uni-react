"""Regression metrics."""
from typing import Optional

import torch
from torch import Tensor


def mae(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Mean absolute error.

    Args:
        pred: Predictions of any shape.
        target: Targets with the same shape as *pred*.
        mask: Boolean mask (``True`` = valid). If ``None``, all elements count.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar tensor.
    """
    diff = torch.abs(pred - target)
    if mask is not None:
        diff = diff[mask]
    if reduction == "sum":
        return diff.sum()
    return diff.mean() if diff.numel() > 0 else diff.new_zeros([])


def rmse(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Root mean squared error.

    Args:
        pred: Predictions.
        target: Targets.
        mask: Boolean mask (``True`` = valid).

    Returns:
        Scalar tensor.
    """
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff[mask]
    mse = diff.mean() if diff.numel() > 0 else diff.new_zeros([])
    return torch.sqrt(mse)
