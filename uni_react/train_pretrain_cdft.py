#!/usr/bin/env python3
"""CDFT reactivity pretraining entry-point."""
from __future__ import annotations

import uni_react.encoders  # noqa: F401
import uni_react.losses  # noqa: F401
import uni_react.loggers  # noqa: F401
import uni_react.schedulers  # noqa: F401
from uni_react.training.pretrain_runner import run_pretrain_entry


def main() -> None:
    run_pretrain_entry(
        train_mode="cdft",
        description="uni-react CDFT pretraining",
    )


if __name__ == "__main__":
    main()
