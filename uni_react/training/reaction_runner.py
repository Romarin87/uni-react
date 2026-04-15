"""Shared runtime wiring for reaction pretraining entry-points."""
from __future__ import annotations

import torch

from uni_react.configs import ReactionPretrainConfig
from uni_react.encoders.reaction_model import ReactionPretrainNet
from uni_react.training.checkpoint import load_init_checkpoint
from uni_react.training.distributed import cleanup_distributed, init_distributed, is_main_process
from uni_react.training.entrypoint_utils import (
    build_console_logger,
    build_dataclass_arg_parser,
    dump_runtime_config,
    load_dataclass_config,
)
from uni_react.training.optimizer import build_split_lr_optimizer
from uni_react.training.pretrain_builders import build_pretrain_encoder
from uni_react.training.seed import set_seed
from uni_react.trainers.pretrain_reaction import ReactionPretrainTrainer

import uni_react.loggers  # noqa: F401


def run_reaction_entry() -> None:
    parser = build_dataclass_arg_parser(
        ReactionPretrainConfig,
        "uni-react stage-3 reaction triplet pretraining",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, ReactionPretrainConfig)

    if not cfg.train_h5:
        parser.error("--train_h5 is required (or set train_h5 in config file).")

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    del local_rank
    set_seed(cfg.seed + rank)

    if is_main_process(rank):
        dump_runtime_config(cfg, cfg.out_dir)

    descriptor = build_pretrain_encoder(cfg).to(device)

    logger = build_console_logger(cfg.out_dir, cfg.log_file, rank)

    if cfg.init_ckpt:
        load_init_checkpoint(
            model=descriptor,
            ckpt_path=cfg.init_ckpt,
            device=device,
            strict=cfg.init_strict,
            rank=rank,
            logger=logger,
        )

    model = ReactionPretrainNet(
        descriptor=descriptor,
        emb_dim=cfg.emb_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        teacher_momentum=cfg.teacher_momentum,
    ).to(device)

    optimizer = build_split_lr_optimizer(
        model=model,
        backbone_module=model.online_descriptor,
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        backbone_prefix="online_descriptor.",
        head_group_name="heads",
    )
    trainer = ReactionPretrainTrainer(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        logger=logger,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
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
    finally:
        cleanup_distributed(distributed)
