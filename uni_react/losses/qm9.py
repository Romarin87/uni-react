"""QM9 property regression loss."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("qm9_regression")
class QM9RegressionLoss:
    """Normalised MSE loss for QM9 property regression.

    Predictions are assumed to be in normalised space
    (zero mean, unit variance per target).  The loss is computed on the
    normalised predictions and normalised targets.

    Config example::

        loss:
          type: qm9_regression
    """

    def metric_keys(self) -> Tuple[str, ...]:
        return ("loss", "mae")

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        target_mean: Optional[Tensor] = None,
        target_std: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute normalised MSE + MAE.

        Args:
            outputs: Must contain ``"pred"`` of shape ``(B, T)``.
            batch: Must contain ``"y"`` of shape ``(B, T)``.
            target_mean: Per-target mean ``(T,)`` for unnormalising MAE.
            target_std: Per-target std ``(T,)`` for unnormalising MAE.

        Returns:
            Dict with keys ``"loss"`` (normalised MSE) and ``"mae"``.
        """
        pred_norm = outputs["pred"]
        if pred_norm.ndim == 1:
            pred_norm = pred_norm.unsqueeze(-1)
        y = batch["y"]

        if target_mean is not None and target_std is not None:
            mean = target_mean.unsqueeze(0)
            std = target_std.unsqueeze(0)
            y_norm = (y - mean) / std
            loss = F.mse_loss(pred_norm, y_norm)
            pred = pred_norm * std + mean
            mae = torch.abs(pred - y).mean()
        else:
            loss = F.mse_loss(pred_norm, y)
            mae = torch.abs(pred_norm - y).mean()

        return {"loss": loss, "mae": mae}
