"""Electronic-structure pretraining loss."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("electronic_structure")
class ElectronicStructureLoss:
    """Weighted sum of VIP/VEA MSE and Fukui-index MSE.

    Config example::

        loss:
          type: electronic_structure
          vip_vea_weight: 1.0
          fukui_weight: 1.0
          vip_vea_keys: [vip, vea]
          fukui_keys: [f_plus, f_minus, f_zero]
    """

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
        vk = tuple(f"vip_vea_loss_{k}" for k in self.vip_vea_keys)
        fk = tuple(f"fukui_loss_{k}" for k in self.fukui_keys)
        return base + vk + fk

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        zero = self._zero(outputs)

        # VIP/VEA component
        vip_vea_loss = zero
        component_vip_vea: Dict[str, Tensor] = {}
        vip_pred = outputs.get("vip_vea_pred")
        vip_target = batch.get("reactivity_global")
        if isinstance(vip_pred, Tensor) and isinstance(vip_target, Tensor):
            vip_vea_loss = F.mse_loss(vip_pred, vip_target[:, : vip_pred.shape[-1]])
            n = min(vip_pred.shape[-1], len(self.vip_vea_keys))
            for i in range(n):
                component_vip_vea[f"vip_vea_loss_{self.vip_vea_keys[i]}"] = F.mse_loss(
                    vip_pred[:, i], vip_target[:, i]
                )

        # Fukui component
        fukui_loss = zero
        component_fukui: Dict[str, Tensor] = {}
        fukui_pred = outputs.get("fukui_pred")
        fukui_target = batch.get("reactivity_atom")
        atom_padding = batch.get("atom_padding")
        atom_valid_mask = batch.get("reactivity_atom_valid")
        if isinstance(fukui_pred, Tensor) and isinstance(fukui_target, Tensor):
            valid = torch.ones(
                fukui_pred.shape[:2], dtype=torch.bool, device=fukui_pred.device
            )
            if isinstance(atom_padding, Tensor):
                valid = valid & (~atom_padding)
            if isinstance(atom_valid_mask, Tensor):
                valid = valid & atom_valid_mask
            valid_f = valid.float()
            denom = valid_f.sum() + 1e-8
            diff = F.mse_loss(fukui_pred, fukui_target[:, :, : fukui_pred.shape[-1]], reduction="none")
            fukui_loss = (diff.mean(-1) * valid_f).sum() / denom
            n = min(fukui_pred.shape[-1], len(self.fukui_keys))
            for i in range(n):
                component_fukui[f"fukui_loss_{self.fukui_keys[i]}"] = (
                    (diff[..., i] * valid_f).sum() / denom
                )

        total = self.vip_vea_weight * vip_vea_loss + self.fukui_weight * fukui_loss
        out: Dict[str, Tensor] = {
            "loss": total,
            "vip_vea_loss": vip_vea_loss,
            "fukui_loss": fukui_loss,
        }
        out.update(component_vip_vea)
        out.update(component_fukui)
        return out

    @staticmethod
    def _zero(outputs: Dict[str, Tensor]) -> Tensor:
        for v in outputs.values():
            if isinstance(v, Tensor):
                return v.sum() * 0.0
        return torch.tensor(0.0)
