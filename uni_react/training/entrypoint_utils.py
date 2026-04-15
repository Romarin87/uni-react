"""Shared helpers for training entry-points."""
from __future__ import annotations

import argparse
import dataclasses
import types
from pathlib import Path
from typing import Any, Type, TypeVar, Union, get_args, get_origin

from uni_react.configs import dump_config, load_config, merge_cli_args
from uni_react.registry import LOGGER_REGISTRY

T = TypeVar("T")


def _parse_cli_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _field_is_bool(field: dataclasses.Field) -> bool:
    if isinstance(field.default, bool):
        return True
    field_type = field.type
    if field_type in ("bool", bool):
        return True
    origin = get_origin(field_type)
    args = get_args(field_type)
    return origin is None and False or bool in args


def _field_is_list(field: dataclasses.Field, default: Any) -> bool:
    if isinstance(default, list):
        return True
    field_type = field.type
    origin = get_origin(field_type)
    if origin in (list, tuple):
        return True
    args = get_args(field_type)
    return any(get_origin(arg) in (list, tuple) or arg in (list, tuple) for arg in args)


def _resolve_scalar_cli_type(field: dataclasses.Field, default: Any) -> type:
    if default is not None and not isinstance(default, (list, dict)):
        return type(default)

    field_type = field.type
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        scalar_args = [arg for arg in non_none_args if arg in (str, int, float)]
        if len(scalar_args) == 1:
            return scalar_args[0]

    if field_type in (str, int, float):
        return field_type

    return str


def build_dataclass_arg_parser(
    config_cls: Type[T],
    description: str,
    *,
    formatter_class: type[argparse.HelpFormatter] | None = argparse.ArgumentDefaultsHelpFormatter,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=formatter_class,
    )
    parser.add_argument("--config", type=str, default="", metavar="FILE")
    for field in dataclasses.fields(config_cls):  # type: ignore[arg-type]
        default = field.default if field.default is not dataclasses.MISSING else None
        if default is None and field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = field.default_factory()  # type: ignore[misc]
        if _field_is_bool(field):
            parser.add_argument(
                f"--{field.name}",
                type=_parse_cli_bool,
                nargs="?",
                const=True,
                default=None,
                help=f"Override config.{field.name} with true/false",
            )
            continue
        if _field_is_list(field, default):
            parser.add_argument(
                f"--{field.name}",
                type=str,
                nargs="+",
                default=None,
                help=f"Override config.{field.name} with one or more values",
            )
            continue
        cli_type = _resolve_scalar_cli_type(field, default)
        parser.add_argument(f"--{field.name}", type=cli_type, default=None, help=f"Override config.{field.name}")
    return parser


def load_dataclass_config(args: argparse.Namespace, config_cls: Type[T]) -> T:
    cfg = load_config(args.config, config_cls) if args.config else config_cls()  # type: ignore[call-arg]
    return merge_cli_args(cfg, vars(args))


def build_console_logger(out_dir: str | Path, log_file: str, rank: int):
    logger = LOGGER_REGISTRY.build({
        "type": "console",
        "log_file": str(Path(out_dir) / log_file),
    })
    if hasattr(logger, "set_rank"):
        logger.set_rank(rank)
    return logger


def dump_runtime_config(cfg: Any, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    yaml_path = out_dir / "config.yaml"
    try:
        dump_config(cfg, str(yaml_path))
        return yaml_path
    except ImportError:
        json_path = out_dir / "config.json"
        dump_config(cfg, str(json_path))
        return json_path
