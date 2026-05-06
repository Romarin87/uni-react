"""Runtime helpers for joint task training."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

import torch

from ...configs import JointConfig
from ...models import build_model_spec
from ...training.checkpoint import load_init_checkpoint
from .adapters import TaskAdapter, build_adapter
from .data_plan import build_data_plan
from .model import JointTaskModel
from .trainer import JointTrainer


def _model_namespace(model_cfg: Dict) -> SimpleNamespace:
    defaults = {
        "name": "single_mol",
        "emb_dim": 256,
        "inv_layer": 2,
        "se3_layer": 4,
        "heads": 8,
        "atom_vocab_size": 128,
        "cutoff": 5.0,
        "num_kernel": 128,
        "path_dropout": 0.1,
        "activation_dropout": 0.1,
        "attn_dropout": 0.1,
    }
    merged = {**defaults, **model_cfg}
    merged["model_name"] = merged.get("name", merged.get("model_name", "single_mol"))
    return SimpleNamespace(**merged)


def _eval_task_names(cfg: JointConfig) -> List[str]:
    mode = cfg.evaluation.get("eval_tasks", "active")
    if isinstance(mode, list):
        return [name for name in mode if bool(cfg.tasks[name].get("enabled", True))]
    if mode == "all":
        return [name for name, task_cfg in cfg.tasks.items() if bool(task_cfg.get("enabled", True))]
    if mode == "active":
        return list(cfg.active_train_tasks)
    raise ValueError("evaluation.eval_tasks must be active, all, or a list of task names")


def _build_optimizer(model: JointTaskModel, cfg: JointConfig) -> torch.optim.Optimizer:
    param_groups = []
    desc_params = [p for p in model.descriptor.parameters() if p.requires_grad]
    if desc_params:
        first_task = cfg.active_train_tasks[0]
        param_groups.append(
            {
                "params": desc_params,
                "lr": float(cfg.learning_rates["descriptor"][first_task]),
                "name": "descriptor",
            }
        )
    for task_name in cfg.tasks:
        if task_name not in model.tasks:
            continue
        params = [p for p in model.task_parameters(task_name) if p.requires_grad]
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": float(cfg.learning_rates["head"].get(task_name, 1e-4)),
                    "name": f"task.{task_name}",
                }
            )
    if not param_groups:
        raise ValueError("No trainable parameters found for joint model")
    return torch.optim.AdamW(param_groups, weight_decay=float(cfg.optimization.get("weight_decay", 1e-2)))


def build_joint_trainer(
    cfg: JointConfig,
    *,
    device,
    distributed: bool,
    rank: int,
    world_size: int,
    logger,
) -> JointTrainer:
    model_cfg = _model_namespace(cfg.model)
    model_spec = build_model_spec(model_cfg.model_name)
    descriptor = model_spec.build_backbone(model_cfg)
    model = JointTaskModel(
        descriptor=descriptor,
        emb_dim=int(model_cfg.emb_dim),
        atom_vocab_size=int(model_cfg.atom_vocab_size),
        task_configs=cfg.tasks,
    ).to(device)

    adapters: Dict[str, TaskAdapter] = {
        name: build_adapter(
            name,
            task_cfg,
            run_cfg=cfg.run,
            model_cfg=cfg.model,
            advanced_cfg=cfg.advanced,
        )
        for name, task_cfg in cfg.tasks.items()
        if bool(task_cfg.get("enabled", True))
    }
    eval_names = _eval_task_names(cfg)
    file_limit = int(cfg.advanced_value("limits", "h5_file_limit", default=0) or 0)
    data_plan = build_data_plan(
        adapters,
        cfg.tasks,
        active_train_tasks=cfg.active_train_tasks,
        eval_task_names=eval_names,
        file_limit=file_limit,
    )

    if cfg.checkpoint.get("init_ckpt"):
        load_init_checkpoint(
            model=model,
            ckpt_path=cfg.checkpoint["init_ckpt"],
            device=device,
            strict=bool(cfg.checkpoint.get("init_strict", False)),
            rank=rank,
            logger=logger,
        )

    optimizer = _build_optimizer(model, cfg)
    trainer = JointTrainer(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        data_plan=data_plan,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
        logger=logger,
    )
    if cfg.checkpoint.get("restart"):
        trainer.load_checkpoint(
            cfg.checkpoint["restart"],
            strict=True,
            ignore_config_mismatch=bool(cfg.checkpoint.get("restart_ignore_config", False)),
        )
    return trainer
