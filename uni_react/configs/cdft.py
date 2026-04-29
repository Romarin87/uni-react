"""Dataclass schema for CDFT task runs."""

from dataclasses import dataclass

from .geometric import GeometricConfig


@dataclass
class CDFTConfig(GeometricConfig):
    """Configuration for CDFT/electronic-structure task training."""

    train_mode: str = "cdft"
