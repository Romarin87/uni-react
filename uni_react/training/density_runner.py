"""Shared runtime wiring for density pretraining entry-points."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from uni_react.configs import DensityPretrainConfig
from uni_react.encoders import DensityPretrainNet
from uni_react.registry import SCHEDULER_REGISTRY
from uni_react.training.checkpoint import load_init_checkpoint
from uni_react.training.distributed import cleanup_distributed, init_distributed, is_main_process
from uni_react.training.entrypoint_utils import (
    build_console_logger,
    build_dataclass_arg_parser,
    dump_runtime_config,
    load_dataclass_config,
)
from uni_react.training.optimizer import build_split_lr_optimizer
from uni_react.training.seed import set_seed
from uni_react.trainers import DensityPretrainTrainer
from uni_react.utils import H5DensityPretrainDataset, collate_fn_density, expand_h5_files

import uni_react.loggers  # noqa: F401
import uni_react.schedulers  # noqa: F401


def run_density_entry() -> None:
    parser = build_dataclass_arg_parser(
        DensityPretrainConfig,
        "Stage-2 electron density pretraining",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, DensityPretrainConfig)

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    del local_rank
    set_seed(cfg.seed + rank)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_console_logger(out_dir, cfg.log_file, rank)

    if is_main_process(rank):
        dump_runtime_config(cfg, out_dir)
        logger.log_config(dataclasses.asdict(cfg))

    train_files = expand_h5_files(cfg.train_h5)
    val_files = expand_h5_files(cfg.val_h5) if cfg.val_h5 else []
    if is_main_process(rank):
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

    model = DensityPretrainNet(
        encoder_type=cfg.encoder_type,
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
        point_hidden_dim=cfg.point_hidden_dim,
        cond_hidden_dim=cfg.cond_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        radial_sigma=cfg.radial_sigma,
    ).to(device)

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
        scheduler = SCHEDULER_REGISTRY.build({
            "type": "cosine",
            "optimizer": optimizer,
            "warmup_steps": cfg.warmup_steps,
            "total_steps": 1,
        })

    trainer = DensityPretrainTrainer(
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
        find_unused_parameters=cfg.encoder_type in {"reacformer_se3", "reacformer_so2"},
    )

    start_epoch = 1
    if cfg.restart:
        start_epoch = trainer.load_checkpoint(
            cfg.restart,
            strict=True,
            ignore_config_mismatch=cfg.restart_ignore_config,
        )
        if is_main_process(rank):
            logger.log({"ckpt": cfg.restart, "start_epoch": start_epoch}, phase="restart")

    try:
        trainer.fit(start_epoch=start_epoch)
    finally:
        cleanup_distributed(distributed)
