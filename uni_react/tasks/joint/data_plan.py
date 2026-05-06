"""Data access planning for joint task training."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..components.dataset_helpers import expand_h5_files
from .adapters import DatasetBuildResult, TaskAdapter


@dataclass
class TaskData:
    task_name: str
    adapter: TaskAdapter
    train: Optional[DatasetBuildResult]
    val: Optional[DatasetBuildResult]


@dataclass
class DataSourceGroup:
    split: str
    source_id: str
    schema: str
    files: Sequence[str]
    tasks: Sequence[str]
    required_keys: Sequence[str]
    samples: int


@dataclass
class DataPlan:
    task_data: Dict[str, TaskData]
    groups: Sequence[DataSourceGroup]

    def format(self) -> str:
        lines = ["[joint:data_plan]"]
        for group in self.groups:
            lines.append(
                f"split={group.split} id={group.source_id} schema={group.schema} "
                f"files={len(group.files)} samples={group.samples} "
                f"tasks={','.join(group.tasks)} "
                f"required_keys={','.join(group.required_keys)}"
            )
        return "\n".join(lines)


def _limit_files(files: Sequence[str], limit: int) -> List[str]:
    out = list(files)
    if limit > 0:
        out = out[:limit]
    return out


def _expand(paths, file_limit: int) -> List[str]:
    files = expand_h5_files(paths)
    return _limit_files(files, file_limit)


def build_data_plan(
    adapters: Dict[str, TaskAdapter],
    task_configs: Dict[str, Dict],
    *,
    active_train_tasks: Iterable[str],
    eval_task_names: Iterable[str],
    file_limit: int = 0,
) -> DataPlan:
    active_set = set(active_train_tasks)
    eval_set = set(eval_task_names)
    task_data: Dict[str, TaskData] = {}

    for task_name, adapter in adapters.items():
        cfg = task_configs[task_name]
        train_result = None
        val_result = None
        if task_name in active_set:
            train_files = _expand(cfg.get("train_h5", []), file_limit)
            train_result = adapter.build_dataset(train_files, split="train")
        if task_name in eval_set and cfg.get("val_h5"):
            val_files = _expand(cfg.get("val_h5", []), file_limit)
            val_result = adapter.build_dataset(val_files, split="val")
        task_data[task_name] = TaskData(task_name, adapter, train_result, val_result)

    groups: List[DataSourceGroup] = []
    grouped: Dict[Tuple[str, str, Tuple[str, ...]], List[Tuple[str, DatasetBuildResult]]] = defaultdict(list)
    for task_name, item in task_data.items():
        for split, result in (("train", item.train), ("val", item.val)):
            if result is None:
                continue
            grouped[(split, result.schema, tuple(result.files))].append((task_name, result))

    counters: Dict[Tuple[str, str], int] = defaultdict(int)
    for (split, schema, files), entries in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        idx = counters[(split, schema)]
        counters[(split, schema)] += 1
        required = sorted({key for _, result in entries for key in result.required_keys})
        samples = sum(len(result.dataset) for _, result in entries)
        # If multiple tasks share the same files/schema, report sample count once.
        if entries:
            samples = len(entries[0][1].dataset)
        groups.append(
            DataSourceGroup(
                split=split,
                source_id=f"{split}.{schema}.{idx}",
                schema=schema,
                files=files,
                tasks=[task for task, _ in entries],
                required_keys=required,
                samples=int(samples),
            )
        )

    return DataPlan(task_data=task_data, groups=groups)
