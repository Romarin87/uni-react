"""Joint task training CLI entry helpers."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from ...configs import (
    JointConfig,
    build_console_logger,
    build_dataclass_arg_parser,
    dump_runtime_config,
    load_dataclass_config,
)
from ...training.distributed import cleanup_distributed, init_distributed, is_main_process
from ...training.seed import set_seed
from .runtime import build_joint_trainer


def run_joint_entry() -> None:
    parser = build_dataclass_arg_parser(JointConfig, "uni-react joint task training")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Legacy launcher override for run.out_dir.",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, JointConfig)
    if args.out_dir:
        cfg.run = {**cfg.run, "out_dir": args.out_dir}

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.run_value("device", "cuda"))
    set_seed(int(cfg.run_value("seed", 42)) + rank)

    out_dir = Path(cfg.run_value("out_dir", "runs/joint"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg.advanced_value("diagnostics", "log_file", default="train.log")
    logger = build_console_logger(out_dir, log_file, rank)
    runtime = {
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
    }
    if is_main_process(rank):
        dump_runtime_config(cfg, out_dir, runtime=runtime)
        logger.log_metrics("init", runtime)
        logger.log_config(dataclasses.asdict(cfg))

    trainer = build_joint_trainer(
        cfg,
        device=device,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        logger=logger,
    )
    try:
        trainer.fit()
    finally:
        cleanup_distributed(distributed)
