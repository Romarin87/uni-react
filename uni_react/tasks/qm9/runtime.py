"""QM9 runtime assembly helpers."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from ...configs import FinetuneQM9Config
from ...models import build_qm9_model_spec
from ...training.checkpoint import load_init_checkpoint
from ...training.distributed import is_main_process
from ...training.optimizer import build_split_lr_optimizer
from ...training.scheduler import build_scheduler
from .dataset import QM9_TARGETS, get_qm9_atomref
from .common import build_qm9_model as build_common_qm9_model
from .common.trainer import FinetuneQM9Trainer
from .gotennet_l import build_qm9_model as build_gotennet_l_qm9_model
from .gotennet_l.trainer import GotenNetQM9Trainer
from .results import write_qm9_outputs
from .spec import QM9TaskSpec, resolve_qm9_task_spec


def parse_targets(cfg: FinetuneQM9Config) -> List[str]:
    targets = cfg.targets or [cfg.target]
    if len(targets) == 1 and targets[0].lower() == "all":
        return list(QM9_TARGETS)
    return targets


def prepare_qm9_config(cfg: FinetuneQM9Config) -> tuple[FinetuneQM9Config, QM9TaskSpec, List[str]]:
    targets = parse_targets(cfg)
    task_spec = resolve_qm9_task_spec(cfg)
    if not cfg.task_variant:
        cfg = dataclasses.replace(
            cfg,
            task_variant=task_spec.variant,
            split=task_spec.split,
            qm9_target_variant=task_spec.target_index_variant,
            no_center_coords=not task_spec.center_coords,
        )
        task_spec = resolve_qm9_task_spec(cfg)
    if not cfg.out_dir:
        cfg = dataclasses.replace(cfg, out_dir=_derive_qm9_out_dir(cfg, targets))
    return cfg, task_spec, targets


def build_qm9_trainer(
    cfg: FinetuneQM9Config,
    task_spec: QM9TaskSpec,
    targets: List[str],
    *,
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
    logger,
):
    model_spec = build_qm9_model_spec(cfg.model_name, task_spec.variant)
    if cfg.model_name == "gotennet_l":
        target_name = targets[0]
        atomref = None if target_name.lower() in {"mu", "r2"} else get_qm9_atomref(
            root=cfg.data_root,
            target=target_name,
            max_z=max(cfg.atom_vocab_size, 100),
            force_reload=cfg.force_reload,
            target_index_variant=cfg.qm9_target_variant,
        )
        model = build_gotennet_l_qm9_model(cfg, model_spec, targets, task_spec, atomref=atomref).to(device)
    else:
        model = build_common_qm9_model(cfg, model_spec, targets, task_spec).to(device)

    if cfg.pretrained_ckpt:
        load_init_checkpoint(
            model=model,
            ckpt_path=cfg.pretrained_ckpt,
            device=device,
            strict=cfg.pretrained_strict,
            rank=rank,
            logger=None,
        )

    optimizer = _build_qm9_optimizer(cfg, model)
    scheduler = _build_qm9_scheduler(cfg, optimizer)
    trainer_cls = GotenNetQM9Trainer if cfg.model_name == "gotennet_l" else FinetuneQM9Trainer
    return trainer_cls(
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


def finalize_qm9_training(
    cfg: FinetuneQM9Config,
    trainer,
    *,
    distributed: bool,
    rank: int,
    logger,
) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
    _load_best_model_for_qm9_test(trainer)
    best_metrics = _evaluate_best_qm9_splits(trainer)
    if is_main_process(rank):
        logger.log_metrics("best_ckpt_train", best_metrics["train"])
        logger.log_metrics("best_ckpt_val", best_metrics["val"])
        logger.log_metrics("best_ckpt_test", best_metrics["test"])
        write_qm9_outputs(cfg.out_dir, trainer, best_metrics)


def _build_qm9_optimizer(cfg: FinetuneQM9Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.model_name == "gotennet_l":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
    return build_split_lr_optimizer(
        model=model,
        backbone_module=model.descriptor,
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        backbone_prefix="descriptor.",
    )


def _build_qm9_scheduler(
    cfg: FinetuneQM9Config,
    optimizer: torch.optim.Optimizer,
):
    if cfg.lr_scheduler in {"cosine", "linear"}:
        return build_scheduler(
            cfg.lr_scheduler,
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=1,
        )
    return None


def _sanitize_qm9_family_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return token or "custom"


def _infer_qm9_run_family(cfg: FinetuneQM9Config) -> str:
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
    source = (
        ckpt_path.parent.name
        if ckpt_path.stem in {"best", "latest"} or ckpt_path.stem.startswith("epoch_")
        else ckpt_path.stem
    )
    return f"pretrain_{_sanitize_qm9_family_token(source)}"


def _derive_qm9_out_dir(cfg: FinetuneQM9Config, targets: List[str]) -> str:
    if cfg.restart:
        return str(Path(cfg.restart).resolve().parent)
    run_family = _infer_qm9_run_family(cfg)
    target_tag = targets[0].lower() if len(targets) == 1 else "multi"
    variant = cfg.task_variant or ("gotennet" if cfg.model_name == "gotennet_l" else "default")
    return f"runs/qm9_{run_family}_{cfg.model_name}_{variant}_{cfg.split}_{target_tag}"


def _load_best_model_for_qm9_test(trainer) -> None:
    best_path = Path(trainer.out_dir) / "best.pt"
    if not best_path.exists():
        return
    ckpt = torch.load(best_path, map_location=trainer.device)
    trainer.raw_model.load_state_dict(ckpt["model"], strict=True)


def _evaluate_best_qm9_splits(trainer) -> Dict[str, Dict[str, float]]:
    return {
        "train": trainer.eval_train(),
        "val": trainer.eval_val(),
        "test": trainer.eval_test(),
    }
