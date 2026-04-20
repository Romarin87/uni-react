"""QM9 structured outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from uni_react.logger import ResultWriter


def write_qm9_outputs(
    out_dir: str | Path,
    trainer,
    best_metrics: Dict[str, Dict[str, float]],
) -> None:
    writer = ResultWriter(out_dir)
    history = list(getattr(trainer, "epoch_history", []))
    writer.write_jsonl("train_log.jsonl", history)

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
    writer.write_json("test_metrics.json", payload)
