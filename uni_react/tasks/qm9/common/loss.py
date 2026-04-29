"""QM9 property regression loss."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ....training.losses import RegressionLoss


class QM9RegressionLoss:
    """Normalised configurable loss for QM9 property regression."""

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

    def metric_keys(self) -> Tuple[str, ...]:
        return ("loss", "mae")

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        target_mean: Optional[Tensor] = None,
        target_std: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        pred_norm = outputs["pred"]
        if pred_norm.ndim == 1:
            pred_norm = pred_norm.unsqueeze(-1)
        target = batch["y"]
        pred_is_normalized = bool(outputs.get("pred_is_normalized", True))

        if pred_is_normalized and target_mean is not None and target_std is not None:
            mean = target_mean.unsqueeze(0)
            std = target_std.unsqueeze(0)
            target_norm = (target - mean) / std
            loss = self.regression_loss(pred_norm, target_norm)
            pred = pred_norm * std + mean
            mae = torch.abs(pred - target).mean()
        else:
            loss = self.regression_loss(pred_norm, target)
            mae = torch.abs(pred_norm - target).mean()

        return {"loss": loss, "mae": mae}
