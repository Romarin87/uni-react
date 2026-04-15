"""Optimizer construction with per-module learning-rate groups."""
from typing import Optional

import torch


def build_optimizer(
    model: torch.nn.Module,
    distributed: bool,
    lr_default: float,
    weight_decay: float,
    descriptor_lr: Optional[float] = None,
    task_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer with separate LR groups for the descriptor backbone
    and the task / prediction heads.

    Args:
        model: The (possibly DDP-wrapped) model.
        distributed: Whether the model is wrapped in DDP.
        lr_default: Fallback LR when group-specific LRs are not given.
        weight_decay: L2 regularisation coefficient.
        descriptor_lr: LR for ``model.descriptor`` parameters.
        task_lr: LR for all non-descriptor parameters.

    Returns:
        A configured :class:`torch.optim.AdamW` instance.
    """
    target = model.module if distributed else model
    desc_lr = float(lr_default if descriptor_lr is None else descriptor_lr)
    head_lr = float(lr_default if task_lr is None else task_lr)

    descriptor_params = [
        p for p in target.descriptor.parameters() if p.requires_grad
    ]
    task_params = [
        p
        for name, p in target.named_parameters()
        if not name.startswith("descriptor.") and p.requires_grad
    ]

    param_groups = []
    if descriptor_params:
        param_groups.append(
            {"params": descriptor_params, "lr": desc_lr, "name": "descriptor"}
        )
    if task_params:
        param_groups.append(
            {"params": task_params, "lr": head_lr, "name": "tasks"}
        )
    if not param_groups:
        raise ValueError("No trainable parameters found in model.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_split_lr_optimizer(
    model: torch.nn.Module,
    backbone_module: torch.nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    backbone_prefix: str,
    head_group_name: str = "head",
) -> torch.optim.Optimizer:
    """Build AdamW with separate backbone and head parameter groups.

    This variant is used by training entry-points whose backbone module is not
    necessarily exposed as ``model.descriptor``.
    """
    backbone_params = [p for p in backbone_module.parameters() if p.requires_grad]
    head_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(backbone_prefix) and p.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(backbone_lr), "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(head_lr), "name": head_group_name})
    if not param_groups:
        raise ValueError("No trainable parameters found in model.")

    return torch.optim.AdamW(param_groups, weight_decay=float(weight_decay))
