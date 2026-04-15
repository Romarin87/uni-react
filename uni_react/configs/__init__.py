"""Configuration schema and I/O for uni_react training runs.

Usage
-----
Load a YAML config, override with CLI args, then pass the resulting
:class:`PretrainConfig` / :class:`FinetuneQM9Config` to the training loop::

    from uni_react.configs import PretrainConfig, load_config, merge_cli_args

    cfg = load_config("configs/single_mol/geometric.yaml", PretrainConfig)
    cfg = merge_cli_args(cfg, cli_overrides)   # dict of CLI key→value
"""
from .components import (
    ElectronicLossConfig,
    EncoderConfig,
    GeometricLossConfig,
    LoggerConfig,
    SchedulerConfig,
)
from .finetune_qm9 import FinetuneQM9Config
from .io import dump_config, load_config, merge_cli_args
from .pretrain_density import DensityPretrainConfig
from .pretrain import PretrainConfig
from .pretrain_reaction import ReactionPretrainConfig

__all__ = [
    # top-level run configs
    "PretrainConfig",
    "DensityPretrainConfig",
    "FinetuneQM9Config",
    "ReactionPretrainConfig",
    # component sub-configs
    "EncoderConfig",
    "GeometricLossConfig",
    "ElectronicLossConfig",
    "LoggerConfig",
    "SchedulerConfig",
    # I/O
    "load_config",
    "dump_config",
    "merge_cli_args",
]
