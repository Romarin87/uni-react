"""Density pretraining CLI entry helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ...configs import (
    DensityPretrainConfig,
    build_console_logger,
    build_dataclass_arg_parser,
    dump_runtime_config,
    load_dataclass_config,
)
from ...training.distributed import cleanup_distributed, init_distributed, is_main_process
from ...training.seed import set_seed
from .runtime import build_density_trainer
from .spec import resolve_density_task_spec


def run_density_entry() -> None:
    parser = build_dataclass_arg_parser(
        DensityPretrainConfig,
        "Stage-2 electron density pretraining",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, DensityPretrainConfig)
    task_spec = resolve_density_task_spec(cfg)

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    del local_rank
    set_seed(cfg.seed + rank)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_console_logger(out_dir, cfg.log_file, rank)

    if is_main_process(rank):
        dump_runtime_config(cfg, out_dir)
        logger.log_config(dataclasses.asdict(cfg))

    trainer = build_density_trainer(
        cfg,
        task_spec,
        device=device,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        logger=logger,
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
