"""Electronic-structure pretraining loss."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class ElectronicStructureLoss:
    """Weighted sum of VIP/VEA MSE and Fukui-index MSE."""

    def __init__(
        self,
        vip_vea_weight: float = 1.0,
        fukui_weight: float = 1.0,
        vip_vea_keys: Optional[List[str]] = None,
        fukui_keys: Optional[List[str]] = None,
    ) -> None:
        self.vip_vea_weight = float(vip_vea_weight)
        self.fukui_weight = float(fukui_weight)
        self.vip_vea_keys: List[str] = vip_vea_keys or ["vip", "vea"]
        self.fukui_keys: List[str] = fukui_keys or ["f_plus", "f_minus", "f_zero"]

    def metric_keys(self) -> Tuple[str, ...]:
        base = ("loss", "vip_vea_loss", "fukui_loss")
        vip_keys = tuple(f"vip_vea_loss_{key}" for key in self.vip_vea_keys)
        fukui_keys = tuple(f"fukui_loss_{key}" for key in self.fukui_keys)
        return base + vip_keys + fukui_keys

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        zero = self._zero(outputs)

        vip_vea_loss = zero
        component_vip_vea: Dict[str, Tensor] = {}
        vip_pred = outputs.get("vip_vea_pred")
        vip_target = batch.get("reactivity_global")
        if isinstance(vip_pred, Tensor) and isinstance(vip_target, Tensor):
            vip_vea_loss = F.mse_loss(vip_pred, vip_target[:, : vip_pred.shape[-1]])
            count = min(vip_pred.shape[-1], len(self.vip_vea_keys))
            for idx in range(count):
                component_vip_vea[f"vip_vea_loss_{self.vip_vea_keys[idx]}"] = F.mse_loss(
                    vip_pred[:, idx],
                    vip_target[:, idx],
                )

        fukui_loss = zero
        component_fukui: Dict[str, Tensor] = {}
        fukui_pred = outputs.get("fukui_pred")
        fukui_target = batch.get("reactivity_atom")
        atom_padding = batch.get("atom_padding")
        atom_valid_mask = batch.get("reactivity_atom_valid")
        if isinstance(fukui_pred, Tensor) and isinstance(fukui_target, Tensor):
            valid = torch.ones(
                fukui_pred.shape[:2],
                dtype=torch.bool,
                device=fukui_pred.device,
            )
            if isinstance(atom_padding, Tensor):
                valid = valid & (~atom_padding)
            if isinstance(atom_valid_mask, Tensor):
                valid = valid & atom_valid_mask
            valid_f = valid.float()
            denom = valid_f.sum() + 1e-8
            diff = F.mse_loss(
                fukui_pred,
                fukui_target[:, :, : fukui_pred.shape[-1]],
                reduction="none",
            )
            fukui_loss = (diff.mean(-1) * valid_f).sum() / denom
            count = min(fukui_pred.shape[-1], len(self.fukui_keys))
            for idx in range(count):
                component_fukui[f"fukui_loss_{self.fukui_keys[idx]}"] = (
                    (diff[..., idx] * valid_f).sum() / denom
                )

        total = self.vip_vea_weight * vip_vea_loss + self.fukui_weight * fukui_loss
        metrics: Dict[str, Tensor] = {
            "loss": total,
            "vip_vea_loss": vip_vea_loss,
            "fukui_loss": fukui_loss,
        }
        metrics.update(component_vip_vea)
        metrics.update(component_fukui)
        return metrics

    @staticmethod
    def _zero(outputs: Dict[str, Tensor]) -> Tensor:
        for value in outputs.values():
            if isinstance(value, Tensor):
                return value.sum() * 0.0
        return torch.tensor(0.0)
