"""Electron-density regression loss."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from ....training.losses import RegressionLoss


class DensityRegressionLoss:
    def __init__(
        self,
        regression_loss_name: str = "mse",
        huber_delta: float = 1.0,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        self.regression_loss = RegressionLoss(
            regression_loss_name,
            huber_delta=huber_delta,
            charbonnier_eps=charbonnier_eps,
        )

    def __call__(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        loss = self.regression_loss(pred, target)
        mse = torch.nn.functional.mse_loss(pred, target)
        mae = torch.nn.functional.l1_loss(pred, target)
        return {
            "loss": loss,
            "mse": mse,
            "mae": mae,
            "rmse": torch.sqrt(torch.clamp(mse, min=0.0)),
        }
