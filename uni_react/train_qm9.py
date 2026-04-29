#!/usr/bin/env python3
"""QM9 task entry-point.

Usage
-----
python -m uni_react.train_qm9 --config configs/single_mol/qm9.yaml

torchrun --nproc_per_node=4 -m uni_react.train_qm9 \
    --config configs/gotennet_l/qm9.yaml --model_name gotennet_l

torchrun --nproc_per_node=4 -m uni_react.train_qm9 \
    --config configs/gotennet_b/qm9.yaml --model_name gotennet_b
"""
from __future__ import annotations

from uni_react.tasks.qm9 import run_qm9_entry


def main() -> None:
    run_qm9_entry()


if __name__ == "__main__":
    main()
