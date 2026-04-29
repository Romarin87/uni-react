"""Density task runtime helpers."""

from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler

from ...configs import DensityConfig
from ...models import build_model_spec
from ...training.checkpoint import load_init_checkpoint
from ...training.optimizer import build_split_lr_optimizer
from ...training.scheduler import build_scheduler
from ..geometric.common.dataset_helpers import expand_h5_files
from .common import H5DensityPretrainDataset, build_density_model, collate_fn_density
from .common.trainer import DensityPretrainTrainer
from .spec import DensityTaskSpec


def build_density_trainer(
    cfg: DensityConfig,
    task_spec: DensityTaskSpec,
    *,
    device,
    distributed: bool,
    rank: int,
    world_size: int,
    logger,
):
    train_files = expand_h5_files(cfg.train_h5)
    val_files = expand_h5_files(cfg.val_h5) if cfg.val_h5 else []
    if rank == 0:
        logger.log({"train_files": len(train_files), "val_files": len(val_files)}, phase="data")

    center_coords = cfg.center_coords and not cfg.no_center_coords
    train_ds = H5DensityPretrainDataset(
        h5_files=train_files,
        num_query_points=cfg.num_query_points,
        center_coords=center_coords,
        deterministic=False,
        seed=cfg.seed,
        return_ids=False,
    )
    val_ds = (
        H5DensityPretrainDataset(
            h5_files=val_files,
            num_query_points=cfg.num_query_points,
            center_coords=center_coords,
            deterministic=True,
            seed=cfg.seed,
            return_ids=False,
        )
        if val_files
        else None
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if (distributed and val_ds is not None) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
        collate_fn=collate_fn_density,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(cfg.num_workers > 0),
            drop_last=False,
            collate_fn=collate_fn_density,
        )

    model_spec = build_model_spec(cfg.model_name)
    model = build_density_model(cfg, model_spec, task_spec).to(device)
    if cfg.init_ckpt:
        load_init_checkpoint(
            model=model,
            ckpt_path=cfg.init_ckpt,
            device=device,
            strict=bool(cfg.init_strict),
            rank=rank,
            logger=logger,
        )

    optimizer = build_split_lr_optimizer(
        model=model,
        backbone_module=model.descriptor,
        backbone_lr=cfg.descriptor_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        backbone_prefix="descriptor.",
    )
    scheduler = None
    if cfg.lr_scheduler == "cosine":
        scheduler = build_scheduler(
            "cosine",
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=1,
        )

    return DensityPretrainTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        scheduler=scheduler,
        logger=logger,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
        find_unused_parameters=False,
    )
