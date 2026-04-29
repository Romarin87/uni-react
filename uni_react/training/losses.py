"""Shared loss helpers for regression-style task heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


REGRESSION_LOSSES = {"mse", "l1", "mae", "huber", "smooth_l1", "charbonnier", "log_cosh"}


def validate_regression_loss(name: str) -> None:
    if name not in REGRESSION_LOSSES:
        raise ValueError(f"regression_loss must be one of {sorted(REGRESSION_LOSSES)}, got {name!r}")


def regression_loss(
    pred: Tensor,
    target: Tensor,
    *,
    loss: str = "mse",
    reduction: str = "mean",
    huber_delta: float = 1.0,
    charbonnier_eps: float = 1e-3,
) -> Tensor:
    """Compute a configurable elementwise regression loss.

    ``huber`` uses PyTorch's Huber definition. ``smooth_l1`` is kept as an
    explicit alias because some PyTorch docs and older code use that name for
    the same robust-regression family with slightly different scaling.
    """

    validate_regression_loss(loss)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"reduction must be one of none/mean/sum, got {reduction!r}")
    if huber_delta <= 0:
        raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
    if charbonnier_eps <= 0:
        raise ValueError(f"charbonnier_eps must be > 0, got {charbonnier_eps}")

    if loss == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    if loss in {"l1", "mae"}:
        return F.l1_loss(pred, target, reduction=reduction)
    if loss == "huber":
        return F.huber_loss(pred, target, reduction=reduction, delta=float(huber_delta))
    if loss == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction=reduction, beta=float(huber_delta))

    diff = pred - target
    if loss == "charbonnier":
        elem = torch.sqrt(diff.square() + float(charbonnier_eps) ** 2) - float(charbonnier_eps)
    else:
        # Stable log(cosh(x)) = x + softplus(-2x) - log(2).
        elem = diff + F.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, dtype=diff.dtype, device=diff.device))

    if reduction == "none":
        return elem
    if reduction == "sum":
        return elem.sum()
    return elem.mean()


class RegressionLoss(nn.Module):
    """Small module wrapper around :func:`regression_loss` for task losses."""

    def __init__(
        self,
        loss: str = "mse",
        *,
        huber_delta: float = 1.0,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        validate_regression_loss(loss)
        if huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        if charbonnier_eps <= 0:
            raise ValueError(f"charbonnier_eps must be > 0, got {charbonnier_eps}")
        self.loss = loss
        self.huber_delta = float(huber_delta)
        self.charbonnier_eps = float(charbonnier_eps)

    def forward(self, pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        return regression_loss(
            pred,
            target,
            loss=self.loss,
            reduction=reduction,
            huber_delta=self.huber_delta,
            charbonnier_eps=self.charbonnier_eps,
        )
