"""Shared builders for geometric/CDFT pretraining entry-points."""
from __future__ import annotations

import torch

from uni_react.configs import PretrainConfig
from uni_react.encoders import (
    ReacFormerHybridEncoder,
    ReacFormerSE3Encoder,
    ReacFormerSO2Encoder,
    SingleMolEncoder,
    SingleMolPretrainNet,
)
from uni_react.registry import LOSS_REGISTRY


def build_pretrain_encoder(cfg: PretrainConfig) -> torch.nn.Module:
    if cfg.encoder_type == "single_mol":
        return SingleMolEncoder(
            emb_dim=cfg.emb_dim,
            inv_layer=cfg.inv_layer,
            se3_layer=cfg.se3_layer,
            heads=cfg.heads,
            atom_vocab_size=cfg.atom_vocab_size,
            cutoff=cfg.cutoff,
            path_dropout=cfg.path_dropout,
            activation_dropout=cfg.activation_dropout,
            attn_dropout=cfg.attn_dropout,
            num_kernel=cfg.num_kernel,
        )
    if cfg.encoder_type == "reacformer_se3":
        return ReacFormerSE3Encoder(
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
    if cfg.encoder_type == "reacformer_so2":
        return ReacFormerSO2Encoder(
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
    if cfg.encoder_type == "reacformer_hybrid":
        return ReacFormerHybridEncoder(
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
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")


def build_pretrain_model(cfg: PretrainConfig, train_mode: str) -> SingleMolPretrainNet:
    encoder = build_pretrain_encoder(cfg)
    return SingleMolPretrainNet(
        emb_dim=cfg.emb_dim,
        encoder=encoder,
        enable_electronic_structure_task=(train_mode == "cdft"),
    )


def build_pretrain_loss(cfg: PretrainConfig, train_mode: str):
    if train_mode == "geometric_structure":
        return LOSS_REGISTRY.build({
            "type": "geometric_structure",
            "atom_weight": cfg.atom_weight,
            "coord_weight": cfg.coord_weight,
            "charge_weight": cfg.charge_weight,
        })
    if train_mode == "cdft":
        return LOSS_REGISTRY.build({
            "type": "electronic_structure",
            "vip_vea_weight": cfg.vip_vea_weight,
            "fukui_weight": cfg.fukui_weight,
            "vip_vea_keys": cfg.vip_vea_keys,
            "fukui_keys": cfg.fukui_keys,
        })
    raise ValueError(f"Unsupported pretrain mode: {train_mode}")
