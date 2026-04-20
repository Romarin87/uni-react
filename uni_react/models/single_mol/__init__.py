"""SingleMol backbone definition."""

from __future__ import annotations

from .backbone import SingleMolEncoder


def build_backbone(cfg):
    return SingleMolEncoder(
        emb_dim=cfg.emb_dim,
        inv_layer=cfg.inv_layer,
        se3_layer=cfg.se3_layer,
        heads=cfg.heads,
        atom_vocab_size=cfg.atom_vocab_size,
        cutoff=cfg.cutoff,
        num_kernel=cfg.num_kernel,
        path_dropout=cfg.path_dropout,
        activation_dropout=cfg.activation_dropout,
        attn_dropout=cfg.attn_dropout,
    )


__all__ = ["SingleMolEncoder", "build_backbone"]
