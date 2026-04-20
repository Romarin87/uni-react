"""Atom-masking prediction head."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


class AtomMaskHead(torch.nn.Module):
    """Predict masked atom types from corrupted node features."""

    name = "atom_mask"

    def __init__(self, emb_dim: int, atom_vocab_size: int = 128) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, atom_vocab_size),
        )

    def forward(self, descriptors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_feats = descriptors["node_feats"]
        atom_padding = descriptors["atom_padding"]
        atom_logits = self.head(node_feats)
        atom_logits = atom_logits.masked_fill(atom_padding[..., None], 0)
        return {"atom_logits": atom_logits}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        atom_padding = batch["atom_padding"]
        valid_atom = ~atom_padding
        mask_positions = batch["mask_positions"] & valid_atom
        if mask_positions.any():
            return F.cross_entropy(
                outputs["atom_logits"][mask_positions],
                batch["atomic_numbers"][mask_positions],
            )
        return outputs["atom_logits"].sum() * 0.0
