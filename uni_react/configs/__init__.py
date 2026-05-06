"""Configuration schema and I/O for uni_react training runs.

Usage
-----
Load a YAML config, override with CLI args, then pass the resulting
task config to the training loop::

    from uni_react.configs import JointConfig, load_config, merge_cli_args

    cfg = load_config("configs/gotennet_l/joint.yaml", JointConfig)
    cfg = merge_cli_args(cfg, cli_overrides)   # dict of CLI key→value
"""
from .joint import JointConfig
from .io import (
    build_console_logger,
    build_dataclass_arg_parser,
    dump_config,
    dump_runtime_config,
    load_config,
    load_dataclass_config,
    merge_cli_args,
)
from .qm9 import QM9Config
from .reaction import ReactionConfig

__all__ = [
    # top-level run configs
    "QM9Config",
    "ReactionConfig",
    "JointConfig",
    # I/O
    "build_dataclass_arg_parser",
    "load_dataclass_config",
    "load_config",
    "dump_config",
    "dump_runtime_config",
    "build_console_logger",
    "merge_cli_args",
]
