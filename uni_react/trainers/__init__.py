"""Trainer implementations.

Trainers encapsulate the complete train + validate loop.  They accept
pre-built components (encoder, loss, optimizer, scheduler, logger) via
dependency injection, making every piece independently swappable.

Available trainers
------------------
:class:`~uni_react.trainers.base.BaseTrainer`
    Abstract base with shared bookkeeping (checkpoint, distributed reduce,
    progress logging).

:class:`~uni_react.trainers.pretrain.PretrainTrainer`
    Pretraining trainer for geometric- and electronic-structure tasks.

:class:`~uni_react.trainers.finetune_qm9.FinetuneQM9Trainer`
    Fine-tuning trainer for QM9 property regression.
"""
from .base import BaseTrainer
from .finetune_qm9 import FinetuneQM9Trainer
from .pretrain import PretrainTrainer
from .pretrain_density import DensityPretrainTrainer
from .pretrain_reaction import ReactionPretrainTrainer

__all__ = [
    "BaseTrainer",
    "PretrainTrainer",
    "DensityPretrainTrainer",
    "FinetuneQM9Trainer",
    "ReactionPretrainTrainer",
]
