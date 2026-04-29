"""QM9 task CLI entry helpers."""

from __future__ import annotations

from ...configs import (
    QM9Config,
    build_dataclass_arg_parser,
    dump_runtime_config,
    load_dataclass_config,
)
from ...training.logger import build_event_logger
from ...training.distributed import cleanup_distributed, init_distributed, is_main_process
from ...training.seed import set_seed
from .runtime import build_qm9_trainer, finalize_qm9_training, prepare_qm9_config


def run_qm9_entry() -> None:
    parser = build_dataclass_arg_parser(
        QM9Config,
        "uni-react QM9 task",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, QM9Config)
    cfg, task_spec, targets = prepare_qm9_config(cfg)

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    set_seed(cfg.seed + rank)

    logger = build_event_logger(cfg.out_dir, cfg.log_file, rank)
    runtime = {
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
    }
    if is_main_process(rank):
        dump_runtime_config(cfg, cfg.out_dir, runtime=runtime)
        logger.log_metrics("init", runtime)
    if cfg.pretrained_ckpt and is_main_process(rank):
        logger.log_metrics("init", {"ckpt": cfg.pretrained_ckpt, "strict": bool(cfg.pretrained_strict)})

    trainer = build_qm9_trainer(
        cfg,
        task_spec,
        targets,
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

    try:
        trainer.fit(start_epoch=start_epoch)
        finalize_qm9_training(
            cfg,
            trainer,
            distributed=distributed,
            rank=rank,
            logger=logger,
        )
    finally:
        cleanup_distributed(distributed)
