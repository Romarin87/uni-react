"""Shared runtime wiring for QM9 fine-tuning entry-points."""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from uni_react.configs import FinetuneQM9Config
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
from uni_react.trainers.finetune_qm9 import FinetuneQM9Trainer
from uni_react.trainers.gotennet_qm9 import GotenNetQM9Trainer
from uni_react.utils.qm9_dataset import QM9_TARGETS, get_qm9_atomref
from uni_react.registry import SCHEDULER_REGISTRY

import uni_react.loggers  # noqa: F401


def parse_targets(cfg: FinetuneQM9Config) -> List[str]:
    targets = cfg.targets or [cfg.target]
    if len(targets) == 1 and targets[0].lower() == "all":
        return list(QM9_TARGETS)
    return targets


def _sanitize_qm9_family_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return token or "custom"


def infer_qm9_run_family(cfg: FinetuneQM9Config) -> str:
    if not cfg.pretrained_ckpt:
        return "scratch"
    ckpt = cfg.pretrained_ckpt.lower()
    if "reaction" in ckpt:
        return "pretrain_reaction"
    if "density" in ckpt:
        return "pretrain_density"
    if "cdft" in ckpt or "electronic" in ckpt:
        return "pretrain_cdft"
    if "geometric" in ckpt:
        return "pretrain_geometric"
    ckpt_path = Path(cfg.pretrained_ckpt)
    source = ckpt_path.parent.name if ckpt_path.stem in {"best", "latest"} or ckpt_path.stem.startswith("epoch_") else ckpt_path.stem
    return f"pretrain_{_sanitize_qm9_family_token(source)}"


def _derive_qm9_out_dir(cfg: FinetuneQM9Config, targets: List[str]) -> str:
    if cfg.restart:
        return str(Path(cfg.restart).resolve().parent)
    run_family = infer_qm9_run_family(cfg)
    target_tag = targets[0].lower() if len(targets) == 1 else "multi"
    return f"runs/qm9_{run_family}_{cfg.encoder_type}_{cfg.split}_{target_tag}"


def _write_qm9_structured_outputs(
    out_dir: str | Path,
    trainer: FinetuneQM9Trainer,
    best_metrics: Dict[str, Dict[str, float]],
) -> None:
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = list(getattr(trainer, "epoch_history", []))
    history_path = run_dir / "train_log.jsonl"
    with history_path.open("w", encoding="utf-8") as f:
        for item in history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    best_entry = None
    for item in history:
        if item.get("is_best"):
            best_entry = item
    if best_entry is None and history:
        best_entry = min(
            history,
            key=lambda item: float(item.get("val", {}).get("loss", float("inf"))),
        )

    payload = {
        "best_epoch": int(best_entry["epoch"]) if best_entry is not None else -1,
        "train": {k: float(v) for k, v in best_metrics.get("train", {}).items()},
        "val": {k: float(v) for k, v in best_metrics.get("val", {}).items()},
        "test": {k: float(v) for k, v in best_metrics.get("test", {}).items()},
    }
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_best_model_for_qm9_test(trainer: FinetuneQM9Trainer) -> None:
    best_path = Path(trainer.out_dir) / "best.pt"
    if not best_path.exists():
        return
    ckpt = torch.load(best_path, map_location=trainer.device)
    trainer.raw_model.load_state_dict(ckpt["model"], strict=True)


def _evaluate_best_qm9_splits(trainer: FinetuneQM9Trainer) -> Dict[str, Dict[str, float]]:
    return {
        "train": trainer.eval_train(),
        "val": trainer.eval_val(),
        "test": trainer.eval_test(),
    }


def run_qm9_entry() -> None:
    parser = build_dataclass_arg_parser(
        FinetuneQM9Config,
        "uni-react QM9 fine-tuning (v2 – registry architecture)",
    )
    args = parser.parse_args()
    cfg = load_dataclass_config(args, FinetuneQM9Config)
    targets = parse_targets(cfg)
    if not cfg.out_dir:
        cfg = dataclasses.replace(cfg, out_dir=_derive_qm9_out_dir(cfg, targets))

    distributed, rank, world_size, local_rank, device = init_distributed(cfg.device)
    del local_rank
    set_seed(cfg.seed + rank)

    if is_main_process(rank):
        dump_runtime_config(cfg, cfg.out_dir)

    descriptor = build_pretrain_encoder(cfg)
    if cfg.encoder_type == "gotennet_l":
        if len(targets) != 1:
            raise ValueError("gotennet_l with official QM9 heads only supports single-target runs.")
        from uni_react.encoders.gotennet_qm9_model import GotenNetQM9Net, build_gotennet_qm9_metadata

        target_name = targets[0]
        atomref = None if target_name.lower() in {"mu", "r2"} else get_qm9_atomref(
            root=cfg.data_root,
            target=target_name,
            max_z=max(cfg.atom_vocab_size, 100),
            force_reload=cfg.force_reload,
            target_index_variant=cfg.qm9_target_variant,
        )
        metadata = build_gotennet_qm9_metadata(target=target_name, atomref=atomref)
        model = GotenNetQM9Net(
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
            descriptor=descriptor,
            target=target_name,
            metadata=metadata,
        ).to(device)
    else:
        from uni_react.encoders import QM9FineTuneNet

        model = QM9FineTuneNet(
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
            head_hidden_dim=cfg.head_hidden_dim,
            head_dropout=cfg.head_dropout,
            num_targets=len(targets),
            descriptor=descriptor,
        ).to(device)

    if cfg.pretrained_ckpt:
        load_init_checkpoint(
            model=model,
            ckpt_path=cfg.pretrained_ckpt,
            device=device,
            strict=cfg.pretrained_strict,
            rank=rank,
            logger=None,
        )

    if cfg.encoder_type == "gotennet_l":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = build_split_lr_optimizer(
            model=model,
            backbone_module=model.descriptor,
            backbone_lr=cfg.backbone_lr,
            head_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
            backbone_prefix="descriptor.",
        )

    scheduler = None
    if cfg.lr_scheduler in {"cosine", "linear"}:
        scheduler = SCHEDULER_REGISTRY.build({
            "type": cfg.lr_scheduler,
            "optimizer": optimizer,
            "warmup_steps": cfg.warmup_steps,
            "total_steps": 1,
        })

    logger = build_console_logger(cfg.out_dir, cfg.log_file, rank)
    if cfg.pretrained_ckpt and is_main_process(rank):
        logger.log({"ckpt": cfg.pretrained_ckpt, "strict": bool(cfg.pretrained_strict)}, phase="init")

    trainer_cls = GotenNetQM9Trainer if cfg.encoder_type == "gotennet_l" else FinetuneQM9Trainer
    trainer = trainer_cls(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        targets=targets,
        scheduler=scheduler,
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
        if distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()
        _load_best_model_for_qm9_test(trainer)
        best_metrics = _evaluate_best_qm9_splits(trainer)
        if is_main_process(rank):
            logger.log(best_metrics["train"], phase="best_ckpt_train")
            logger.log(best_metrics["val"], phase="best_ckpt_val")
            logger.log(best_metrics["test"], phase="best_ckpt_test")
            _write_qm9_structured_outputs(cfg.out_dir, trainer, best_metrics)
    finally:
        cleanup_distributed(distributed)
