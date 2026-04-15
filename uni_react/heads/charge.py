"""Atomic-charge prediction head."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("charge")
class ChargeHead(torch.nn.Module):
    """Predicts per-atom partial charges.

    Config example::

        heads:
          - type: charge
            emb_dim: 256
    """

    name = "charge"

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, 1),
        )

    def forward(self, descriptors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_feats   = descriptors["node_feats"]
        atom_padding = descriptors["atom_padding"]
        charge_pred  = self.head(node_feats).squeeze(-1)
        charge_pred  = charge_pred.masked_fill(atom_padding, 0)
        return {"charge_pred": charge_pred}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        atom_padding = batch["atom_padding"]
        valid_atom   = ~atom_padding
        charge_mask  = batch.get("charge_valid", valid_atom) & valid_atom
        if charge_mask.any():
            return F.mse_loss(
                outputs["charge_pred"][charge_mask],
                batch["charges"][charge_mask],
            )
        return outputs["charge_pred"].sum() * 0.0
