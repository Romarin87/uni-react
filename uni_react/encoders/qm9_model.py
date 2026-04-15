"""QM9FineTuneNet: configurable backbone + regression head for QM9 properties."""
from typing import Dict, Optional

import torch

from .single_mol import SingleMolEncoder


class QM9FineTuneNet(torch.nn.Module):
    """Fine-tuning model for QM9 property regression.

    Wraps :class:`SingleMolPretrainNet` as a frozen-or-trainable backbone
    and attaches a per-property MLP regression head.
    """

    def __init__(
        self,
        emb_dim: int,
        inv_layer: int,
        se3_layer: int,
        heads: int,
        atom_vocab_size: int = 128,
        cutoff: float = 5.0,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_kernel: int = 128,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        num_targets: int = 1,
        descriptor: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor or SingleMolEncoder(
            emb_dim=emb_dim,
            inv_layer=inv_layer,
            se3_layer=se3_layer,
            heads=heads,
            atom_vocab_size=atom_vocab_size,
            cutoff=cutoff,
            path_dropout=path_dropout,
            activation_dropout=activation_dropout,
            attn_dropout=attn_dropout,
            num_kernel=num_kernel,
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, head_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(head_dropout),
            torch.nn.Linear(head_hidden_dim, num_targets),
        )
        self.num_targets = int(num_targets)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        atom_padding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if atom_padding is None:
            atom_padding = torch.zeros_like(atomic_numbers, dtype=torch.bool)
        backbone_out = self.descriptor(
            input_atomic_numbers=atomic_numbers,
            coords_noisy=coords,
            atom_padding=atom_padding,
        )
        node_feats = backbone_out["node_feats"]
        valid  = (~atom_padding).unsqueeze(-1)
        denom  = valid.sum(dim=1).clamp_min(1)
        pooled = (node_feats * valid).sum(dim=1) / denom
        pred   = self.reg_head(pooled)
        if self.num_targets == 1:
            pred = pred.squeeze(-1)
        return {"pred": pred, "node_feats": node_feats, "pooled_feats": pooled}
