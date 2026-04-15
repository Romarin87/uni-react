#!/usr/bin/env python3
"""QM9 fine-tuning entry-point.

Usage
-----
python -m uni_react.train_finetune_qm9 --config configs/finetune_qm9_gap.yaml

torchrun --nproc_per_node=4 -m uni_react.train_finetune_qm9 \
    --config configs/finetune_qm9_gap.yaml
"""
from __future__ import annotations

from uni_react.training.qm9_runner import run_qm9_entry


def main() -> None:
    run_qm9_entry()


if __name__ == "__main__":
    main()
