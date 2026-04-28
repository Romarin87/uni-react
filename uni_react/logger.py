"""Unified logger and structured result writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class LoggerProtocol(Protocol):
    """Logger contract used by trainers and task runners."""

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        ...

    def log_metrics(
        self,
        phase: str,
        payload: Dict[str, Any],
        *,
        step: Optional[int] = None,
    ) -> None:
        ...

    def log_config(self, payload: Dict[str, Any]) -> None:
        ...

    def log_artifact(self, path: str | Path, kind: str) -> None:
        ...

    def set_rank(self, rank: int) -> None:
        ...

    def finish(self) -> None:
        ...


class ConsoleLogger:
    """Console-backed logger with optional file mirroring."""

    def __init__(
        self,
        rank: int = 0,
        log_file: str = "",
        *,
        file_phases: Optional[set[str]] = None,
    ) -> None:
        self._rank = rank
        self._current_rank: Optional[int] = None
        self._log_file = Path(log_file) if log_file else None
        self._file_phases = set(file_phases) if file_phases is not None else {"init", "epoch", "early_stop", "nonfinite"}

    def set_rank(self, rank: int) -> None:
        self._current_rank = rank

    def _should_log(self) -> bool:
        if self._current_rank is None:
            return True
        return self._current_rank == self._rank

    def _write_file(self, line: str, phase: str) -> None:
        if self._log_file is None:
            return
        if phase not in self._file_phases:
            return
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        if not self._should_log():
            return
        parts = [f"[{phase}]"]
        if step is not None:
            parts.append(f"step={step}")
        parts += [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        line = " ".join(parts)
        print(line)
        self._write_file(line, phase)

    def log_metrics(
        self,
        phase: str,
        payload: Dict[str, Any],
        *,
        step: Optional[int] = None,
    ) -> None:
        self.log(payload, step=step, phase=phase)

    def log_config(self, payload: Dict[str, Any]) -> None:
        if not self._should_log():
            return
        line = "[config] " + json.dumps(payload, indent=2, default=str)
        print(line)
        self._write_file(line, "config")

    def log_artifact(self, path: str | Path, kind: str) -> None:
        if not self._should_log():
            return
        line = f"[artifact] kind={kind} path={Path(path)}"
        print(line)
        self._write_file(line, "artifact")

    def finish(self) -> None:
        return


class ResultWriter:
    """Structured result writer for experiment outputs."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.out_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    def write_jsonl(self, name: str, records: list[Dict[str, Any]]) -> Path:
        path = self.out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path


def build_event_logger(out_dir: str | Path, log_file: str, rank: int) -> ConsoleLogger:
    logger = ConsoleLogger(log_file=str(Path(out_dir) / log_file))
    logger.set_rank(rank)
    return logger
