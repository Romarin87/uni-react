"""Dataclass schema for joint task training runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SUPPORTED_JOINT_TASKS = {
    "atom_mask",
    "coord_denoise",
    "charge",
    "electron_density",
    "vip",
    "vea",
    "fukui",
}


@dataclass
class JointConfig:
    run: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    schedule: Dict[str, Dict[str, float]] = field(default_factory=dict)
    loss_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    learning_rates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    advanced: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("JointConfig.tasks must not be empty")

        unknown = set(self.tasks) - SUPPORTED_JOINT_TASKS
        if unknown:
            raise ValueError(
                f"Unknown joint task(s): {sorted(unknown)}. "
                f"Supported tasks: {sorted(SUPPORTED_JOINT_TASKS)}"
            )

        sample_prob = self.schedule.get("sample_prob", {})
        if not sample_prob:
            raise ValueError("schedule.sample_prob must be provided")
        for task_name, prob in sample_prob.items():
            if task_name not in self.tasks:
                raise ValueError(f"schedule.sample_prob contains unknown configured task: {task_name!r}")
            if float(prob) < 0:
                raise ValueError(f"schedule.sample_prob.{task_name} must be >= 0")
            if float(prob) > 0 and not bool(self.tasks[task_name].get("enabled", True)):
                raise ValueError(
                    f"schedule.sample_prob.{task_name} > 0 but task is disabled"
                )

        active = [
            name
            for name, task_cfg in self.tasks.items()
            if bool(task_cfg.get("enabled", True)) and float(sample_prob.get(name, 0.0)) > 0.0
        ]
        if not active:
            raise ValueError("At least one enabled task must have sample_prob > 0")

        for name in active:
            task_cfg = self.tasks[name]
            if not task_cfg.get("train_h5"):
                raise ValueError(f"Active task {name!r} requires train_h5")
            if int(task_cfg.get("batch_size", 0)) <= 0:
                raise ValueError(f"Task {name!r} requires batch_size > 0")

        optimization = self.optimization
        train_unit = optimization.get("train_unit", "steps")
        if train_unit not in {"steps", "epochs"}:
            raise ValueError("optimization.train_unit must be one of steps/epochs")
        if train_unit == "steps":
            if int(optimization.get("max_steps") or 0) <= 0:
                raise ValueError("optimization.max_steps must be > 0 when train_unit=steps")
        else:
            epochs = optimization.get("epochs")
            reference = optimization.get("epoch_reference_task")
            if int(epochs or 0) <= 0:
                raise ValueError("optimization.epochs must be > 0 when train_unit=epochs")
            if not reference:
                raise ValueError("optimization.epoch_reference_task is required when train_unit=epochs")
            if reference not in active:
                raise ValueError("optimization.epoch_reference_task must be enabled with sample_prob > 0")

        eval_tasks = self.evaluation.get("eval_tasks", "active")
        if isinstance(eval_tasks, list):
            unknown_eval = set(eval_tasks) - set(self.tasks)
            if unknown_eval:
                raise ValueError(f"evaluation.eval_tasks contains unknown tasks: {sorted(unknown_eval)}")
        elif eval_tasks not in {"active", "all"}:
            raise ValueError("evaluation.eval_tasks must be active, all, or a list of task names")

        for section in ("initial", "final"):
            if section in self.loss_weights:
                for task_name, value in self.loss_weights[section].items():
                    if task_name not in self.tasks:
                        raise ValueError(f"loss_weights.{section} contains unknown task: {task_name!r}")
                    if float(value) < 0:
                        raise ValueError(f"loss_weights.{section}.{task_name} must be >= 0")

        for group in ("descriptor", "head"):
            if group not in self.learning_rates:
                raise ValueError(f"learning_rates.{group} must be provided")
            for task_name, value in self.learning_rates[group].items():
                if task_name not in self.tasks:
                    raise ValueError(f"learning_rates.{group} contains unknown task: {task_name!r}")
                if float(value) <= 0:
                    raise ValueError(f"learning_rates.{group}.{task_name} must be > 0")
            missing_active = [name for name in active if name not in self.learning_rates[group]]
            if missing_active:
                raise ValueError(
                    f"learning_rates.{group} missing active task(s): {missing_active}"
                )

    def run_value(self, key: str, default: Any = None) -> Any:
        return self.run.get(key, default)

    def optimization_value(self, key: str, default: Any = None) -> Any:
        return self.optimization.get(key, default)

    def evaluation_value(self, key: str, default: Any = None) -> Any:
        return self.evaluation.get(key, default)

    def checkpoint_value(self, key: str, default: Any = None) -> Any:
        return self.checkpoint.get(key, default)

    def advanced_value(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.advanced
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    @property
    def active_train_tasks(self) -> List[str]:
        sample_prob = self.schedule.get("sample_prob", {})
        return [
            name
            for name, task_cfg in self.tasks.items()
            if bool(task_cfg.get("enabled", True)) and float(sample_prob.get(name, 0.0)) > 0.0
        ]
