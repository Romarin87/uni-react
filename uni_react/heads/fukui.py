"""Fukui index (f+/f-/f0) prediction head."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("fukui")
class FukuiHead(torch.nn.Module):
    """Predicts per-atom Fukui indices.

    Config example::

        heads:
          - type: fukui
            emb_dim: 256
            out_dim: 3
    """

    name = "fukui"

    def __init__(self, emb_dim: int, out_dim: int = 3) -> None:
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
        node_feats   = descriptors["node_feats"]
        atom_padding = descriptors["atom_padding"]
        pred = self.head(node_feats)
        pred = pred.masked_fill(atom_padding[..., None], 0)
        return {"fukui_pred": pred}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        if "reactivity_atom" not in batch:
            raise KeyError("Missing batch key: reactivity_atom")
        atom_padding = batch["atom_padding"]
        valid_atom   = ~atom_padding
        atom_valid   = batch.get("reactivity_atom_valid", valid_atom) & valid_atom
        if atom_valid.any():
            return F.mse_loss(
                outputs["fukui_pred"][atom_valid],
                batch["reactivity_atom"][atom_valid],
            )
        return outputs["fukui_pred"].sum() * 0.0
