"""Geometric pretraining CLI entry helpers."""

from __future__ import annotations

import dataclasses

from ...configs import (
    PretrainConfig,
    build_console_logger,
    build_dataclass_arg_parser,
    dump_runtime_config,
    load_dataclass_config,
)
from ...training.distributed import cleanup_distributed, init_distributed, is_main_process
from ...training.seed import set_seed
from .runtime import build_geometric_trainer
from .spec import resolve_geometric_task_spec


def run_geometric_entry() -> None:
    parser = build_dataclass_arg_parser(
        PretrainConfig,
        "uni-react geometric pretraining",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, PretrainConfig)
    cfg = dataclasses.replace(cfg, train_mode="geometric_structure")
    task_spec = resolve_geometric_task_spec(cfg)
    if not cfg.out_dir:
        cfg = dataclasses.replace(cfg, out_dir=f"runs/{cfg.model_name}_{task_spec.name}")

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    set_seed(cfg.seed + rank)

    logger = build_console_logger(cfg.out_dir, cfg.log_file, rank)
    runtime = {
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
    }
    if is_main_process(rank):
        dump_runtime_config(cfg, cfg.out_dir, runtime=runtime)
        logger.log_metrics("init", runtime)
        logger.log_config(dataclasses.asdict(cfg))

    trainer = build_geometric_trainer(
        cfg,
        task_spec,
        device=device,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        logger=logger,
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
