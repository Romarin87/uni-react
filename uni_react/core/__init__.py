"""Core interface layer for uni_react.

This package defines only *structural* contracts (PEP 544 Protocols and ABCs).
No concrete implementation lives here.  Every other package depends on these
interfaces, not on each other, keeping inter-module coupling minimal.

Import map
----------
:mod:`core.encoder`    – MolecularEncoderProtocol
:mod:`core.head`       – TaskHeadProtocol
:mod:`core.loss`       – LossFnProtocol, PretrainLossFnProtocol
:mod:`core.dataset`    – MolDatasetProtocol
:mod:`core.logger`     – LoggerProtocol
:mod:`core.scheduler`  – LRSchedulerProtocol
:mod:`core.trainer`    – TrainerProtocol
"""
from .dataset import MolDatasetProtocol
from .encoder import MolecularEncoderProtocol
from .head import TaskHeadProtocol
from .logger import LoggerProtocol
from .loss import LossFnProtocol, PretrainLossFnProtocol
from .scheduler import LRSchedulerProtocol
from .trainer import TrainerProtocol

__all__ = [
    "MolecularEncoderProtocol",
    "TaskHeadProtocol",
    "LossFnProtocol",
    "PretrainLossFnProtocol",
    "MolDatasetProtocol",
    "LoggerProtocol",
    "LRSchedulerProtocol",
    "TrainerProtocol",
]
