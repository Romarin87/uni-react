#!/usr/bin/env python3
"""Stage-3 reaction triplet pretraining entry-point.

Usage
-----
# Single GPU
python -m uni_react.train_pretrain_reaction --config configs/single_mol/reaction.yaml

# Multi-GPU (torchrun)
torchrun --nproc_per_node=8 -m uni_react.train_pretrain_reaction \
    --config configs/reacformer_se3/reaction.yaml

# CLI override
python -m uni_react.train_pretrain_reaction \
    --config configs/reacformer_so2/reaction.yaml \
    --train_h5 /path/to/train.h5 --epochs 30
"""
from __future__ import annotations

from uni_react.training.reaction_runner import run_reaction_entry


def main() -> None:
    run_reaction_entry()


if __name__ == "__main__":
    main()
