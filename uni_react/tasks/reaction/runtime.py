"""Reaction task runtime helpers."""

from __future__ import annotations

from ...configs import ReactionConfig
from ...models import build_model_spec
from ...training.checkpoint import load_init_checkpoint
from ...training.optimizer import build_split_lr_optimizer
from .common import build_reaction_model
from .common.trainer import ReactionPretrainTrainer
from .spec import ReactionTaskSpec


def build_reaction_trainer(
    cfg: ReactionConfig,
    task_spec: ReactionTaskSpec,
    *,
    device,
    distributed: bool,
    rank: int,
    world_size: int,
    logger,
):
    model_spec = build_model_spec(cfg.model_name)
    model = build_reaction_model(cfg, model_spec, task_spec).to(device)

    if cfg.init_ckpt:
        load_init_checkpoint(
            model=model.online_descriptor,
            ckpt_path=cfg.init_ckpt,
            device=device,
            strict=cfg.init_strict,
            rank=rank,
            logger=logger,
        )
        model.teacher_descriptor.load_state_dict(model.online_descriptor.state_dict(), strict=True)

    optimizer = build_split_lr_optimizer(
        model=model,
        backbone_module=model.online_descriptor,
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        backbone_prefix="online_descriptor.",
        head_group_name="heads",
    )
    return ReactionPretrainTrainer(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        logger=logger,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
    )
