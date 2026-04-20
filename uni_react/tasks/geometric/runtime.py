"""Geometric pretraining runtime helpers."""

from __future__ import annotations

from ...configs import PretrainConfig
from ...models import build_model_spec
from ...training.checkpoint import load_init_checkpoint
from ...training.optimizer import build_optimizer
from ...training.scheduler import build_scheduler
from .common import GeometricStructureLoss, build_geometric_model
from .common.trainer import PretrainTrainer
from .spec import GeometricTaskSpec


def build_geometric_trainer(
    cfg: PretrainConfig,
    task_spec: GeometricTaskSpec,
    *,
    device,
    distributed: bool,
    rank: int,
    world_size: int,
    logger,
):
    model_spec = build_model_spec(cfg.model_name)
    model = build_geometric_model(cfg, model_spec, task_spec).to(device)
    if cfg.init_ckpt:
        load_init_checkpoint(
            model=model,
            ckpt_path=cfg.init_ckpt,
            device=device,
            strict=cfg.init_strict,
            rank=rank,
            logger=logger,
        )
    loss_fn = GeometricStructureLoss(
        atom_weight=cfg.atom_weight,
        coord_weight=cfg.coord_weight,
        charge_weight=cfg.charge_weight,
    )
    optimizer = build_optimizer(
        model=model,
        distributed=False,
        lr_default=cfg.lr,
        weight_decay=cfg.weight_decay,
        descriptor_lr=cfg.descriptor_lr,
        task_lr=cfg.task_lr,
    )
    scheduler = None
    if cfg.lr_scheduler != "none":
        scheduler = build_scheduler(
            cfg.lr_scheduler,
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=1,
        )
    return PretrainTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        cfg=cfg,
        scheduler=scheduler,
        logger=logger,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
        find_unused_parameters=False,
    )
