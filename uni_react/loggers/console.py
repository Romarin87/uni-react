"""Console (stdout) logger – the default logging backend."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..registry import LOGGER_REGISTRY


@LOGGER_REGISTRY.register("console")
class ConsoleLogger:
    """Logs metrics to stdout via ``print``.

    Config example::

        logger:
          type: console
          rank: 0
    """

    def __init__(self, rank: int = 0, log_file: str = "") -> None:
        """Args:
            rank: Only the process with this rank will print.
                  Defaults to 0 (main process).
        """
        self._rank = rank
        self._current_rank: Optional[int] = None
        self._log_file = Path(log_file) if log_file else None

    def set_rank(self, rank: int) -> None:
        """Set the current process rank (called by the Trainer)."""
        self._current_rank = rank

    def _should_log(self) -> bool:
        if self._current_rank is None:
            return True
        return self._current_rank == self._rank

    def _write_file(self, line: str) -> None:
        if self._log_file is None:
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
        self._write_file(line)

    def log_config(self, config: Dict[str, Any]) -> None:
        if not self._should_log():
            return
        line = "[config] " + json.dumps(config, indent=2, default=str)
        print(line)
        self._write_file(line)

    def finish(self) -> None:
        pass
