"""Shared runtime wiring for single-molecule pretraining entry-points."""
from __future__ import annotations

import dataclasses

from uni_react.configs import PretrainConfig
from uni_react.registry import SCHEDULER_REGISTRY
from uni_react.training.checkpoint import load_init_checkpoint
from uni_react.training.distributed import cleanup_distributed, init_distributed, is_main_process
from uni_react.training.entrypoint_utils import (
    build_dataclass_arg_parser,
    build_console_logger,
    dump_runtime_config,
    load_dataclass_config,
)
from uni_react.training.optimizer import build_optimizer
from uni_react.training.pretrain_builders import build_pretrain_loss, build_pretrain_model
from uni_react.training.seed import set_seed
from uni_react.trainers.pretrain import PretrainTrainer


def run_pretrain_entry(train_mode: str, description: str) -> None:
    args = build_dataclass_arg_parser(PretrainConfig, description).parse_args()
    cfg = load_dataclass_config(args, PretrainConfig)
    cfg = dataclasses.replace(cfg, train_mode=train_mode)
    if not cfg.out_dir:
        stage_name = "geometric" if train_mode == "geometric_structure" else "cdft"
        cfg = dataclasses.replace(cfg, out_dir=f"runs/{cfg.encoder_type}_{stage_name}")

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    del local_rank
    set_seed(cfg.seed + rank)

    logger = build_console_logger(cfg.out_dir, cfg.log_file, rank)
    if is_main_process(rank):
        dump_runtime_config(cfg, cfg.out_dir)
        logger.log_config(dataclasses.asdict(cfg))

    model = build_pretrain_model(cfg, train_mode=train_mode).to(device)
    if cfg.init_ckpt:
        load_init_checkpoint(
            model=model,
            ckpt_path=cfg.init_ckpt,
            device=device,
            strict=cfg.init_strict,
            rank=rank,
            logger=logger,
        )

    loss_fn = build_pretrain_loss(cfg, train_mode=train_mode)
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
        scheduler = SCHEDULER_REGISTRY.build({
            "type": cfg.lr_scheduler,
            "optimizer": optimizer,
            "warmup_steps": cfg.warmup_steps,
            "total_steps": 1,
        })

    trainer = PretrainTrainer(
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
        find_unused_parameters=True,
    )
    start_epoch = (
        trainer.load_checkpoint(
            cfg.restart,
            strict=True,
            ignore_config_mismatch=cfg.restart_ignore_config,
        )
        if cfg.restart
        else 1
    )

    try:
        trainer.fit(start_epoch=start_epoch)
    finally:
        cleanup_distributed(distributed)
