"""Electronic-structure pretraining pipeline (VIP/VEA + Fukui)."""
from typing import Dict

import torch
from torch import Tensor

from .fukui import FukuiHead
from .vip_vea import VipVeaHead


class ElectronicStructureTask(torch.nn.Module):
    """Bundle of electronic-structure pretraining heads.

    Composes VipVeaHead + FukuiHead and exposes a unified ``forward`` and
    ``compute_loss_dict`` interface.
    """

    name = "electronic_structure"

    def __init__(self, emb_dim: int, vip_vea_dim: int = 2, fukui_dim: int = 3) -> None:
        super().__init__()
        self.vip_vea = VipVeaHead(emb_dim=emb_dim, out_dim=vip_vea_dim)
        self.fukui   = FukuiHead(emb_dim=emb_dim,  out_dim=fukui_dim)

    def forward(self, descriptors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = {}
        out.update(self.vip_vea(descriptors))
        out.update(self.fukui(descriptors))
        return out

    def compute_loss_dict(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        vip_vea_weight: float = 1.0,
        fukui_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        vip_vea_loss = self.vip_vea.compute_loss(outputs, batch)
        fukui_loss   = self.fukui.compute_loss(outputs, batch)
        total = vip_vea_weight * vip_vea_loss + fukui_weight * fukui_loss
        return {"loss": total, "vip_vea_loss": vip_vea_loss, "fukui_loss": fukui_loss}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        return self.compute_loss_dict(outputs, batch)["loss"]
