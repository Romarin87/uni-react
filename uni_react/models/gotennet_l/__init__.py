"""GotenNet-L backbone definition."""

from __future__ import annotations

from .backbone import GotenNetLEncoder


def build_backbone(cfg):
    return GotenNetLEncoder(
        emb_dim=cfg.emb_dim,
        num_layers=cfg.se3_layer,
        heads=cfg.heads,
        atom_vocab_size=cfg.atom_vocab_size,
        cutoff=cfg.cutoff,
        num_rbf=cfg.num_kernel,
        path_dropout=cfg.path_dropout,
        activation_dropout=cfg.activation_dropout,
        attn_dropout=cfg.attn_dropout,
    )


__all__ = ["GotenNetLEncoder", "build_backbone"]
