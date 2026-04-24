"""Shared GotenNet backbone definitions for S/B/L and hat variants."""

from __future__ import annotations

from .backbone import GotenNetLEncoder

_VARIANT_DEFAULTS = {
    "gotennet_s": {"num_layers": 4, "sep_dir": True, "sep_tensor": True},
    "gotennet_b": {"num_layers": 6, "sep_dir": True, "sep_tensor": True},
    "gotennet_l": {"num_layers": 12, "sep_dir": True, "sep_tensor": True},
    "gotennet_s_hat": {"num_layers": 4, "sep_dir": False, "sep_tensor": False},
    "gotennet_b_hat": {"num_layers": 6, "sep_dir": False, "sep_tensor": False},
    "gotennet_l_hat": {"num_layers": 12, "sep_dir": False, "sep_tensor": False},
}


def build_backbone(cfg):
    model_name = getattr(cfg, "model_name", "gotennet_l")
    defaults = _VARIANT_DEFAULTS.get(
        model_name,
        {"num_layers": getattr(cfg, "se3_layer", 12), "sep_dir": True, "sep_tensor": True},
    )
    return GotenNetLEncoder(
        emb_dim=cfg.emb_dim,
        num_layers=defaults["num_layers"],
        heads=cfg.heads,
        atom_vocab_size=cfg.atom_vocab_size,
        cutoff=cfg.cutoff,
        num_rbf=cfg.num_kernel,
        path_dropout=cfg.path_dropout,
        activation_dropout=cfg.activation_dropout,
        attn_dropout=cfg.attn_dropout,
        sep_dir=defaults["sep_dir"],
        sep_tensor=defaults["sep_tensor"],
    )


__all__ = ["GotenNetLEncoder", "build_backbone"]
