"""VIP/VEA (vertical ionisation potential / electron affinity) prediction head."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("vip_vea")
class VipVeaHead(torch.nn.Module):
    """Predicts graph-level VIP and VEA values.

    Config example::

        heads:
          - type: vip_vea
            emb_dim: 256
            out_dim: 2
    """

    name = "vip_vea"

    def __init__(self, emb_dim: int, out_dim: int = 2) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        self.out_dim = int(out_dim)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, self.out_dim),
        )

    def forward(self, descriptors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"vip_vea_pred": self.head(descriptors["graph_feats"])}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        if "reactivity_global" not in batch:
            raise KeyError("Missing batch key: reactivity_global")
        return F.mse_loss(outputs["vip_vea_pred"], batch["reactivity_global"])
