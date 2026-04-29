"""YAML / JSON config I/O, dataclass CLI parsing, and runtime config dumping."""
from __future__ import annotations

import argparse
import dataclasses
import json
import types
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_args, get_origin

try:
    import yaml  # PyYAML
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from ..training.logger import build_event_logger

T = TypeVar("T")


def load_config(path: str, schema: Type[T]) -> T:
    """Load a YAML or JSON file and return a validated config dataclass.

    Only fields declared in *schema* are accepted; unknown keys raise
    :class:`ValueError` to catch typos early.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` or ``.json`` config file.
        schema: A dataclass type (e.g. :class:`~uni_react.configs.GeometricConfig`).

    Returns:
        A fully initialised instance of *schema*.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw: Dict[str, Any]
    if p.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install it with: pip install pyyaml"
            )
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    elif p.suffix == ".json":
        with p.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}. Use .yaml or .json.")

    return _build(raw, schema)


def _config_payload(cfg: Any, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    if runtime is not None:
        data["runtime"] = dict(runtime)
    return data


def dump_config(cfg: Any, path: str) -> None:
    """Serialise a config dataclass to YAML or JSON.

    The output format is inferred from the file extension.

    Args:
        cfg: A dataclass instance.
        path: Destination file path (``.yaml`` / ``.json``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _config_payload(cfg)

    if p.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError("PyYAML is required. pip install pyyaml")
        with p.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    elif p.suffix == ".json":
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}.")


def merge_cli_args(cfg: T, overrides: Dict[str, Any]) -> T:
    """Return a new config with fields overridden by *overrides*.

    *overrides* is typically built from ``vars(argparse.Namespace)`` after
    stripping ``None`` sentinel values for optional arguments.

    Only known fields are applied; unknown keys are silently ignored so that
    argparse-specific keys (e.g. ``config``) do not cause errors.

    Args:
        cfg: Existing config dataclass instance.
        overrides: Mapping of field name → new value.

    Returns:
        A new instance of the same dataclass type.
    """
    known = {f.name for f in dataclasses.fields(cfg)}  # type: ignore[arg-type]
    updates = {k: v for k, v in overrides.items() if k in known and v is not None}
    return dataclasses.replace(cfg, **updates)  # type: ignore[arg-type]


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
    return build_event_logger(out_dir, log_file, rank)


def dump_runtime_config(
    cfg: Any,
    out_dir: str | Path,
    runtime: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(out_dir)
    data = _config_payload(cfg, runtime)
    json_path = out_dir / "config.json"
    dump_config(data, str(json_path))

    yaml_path = out_dir / "config.yaml"
    try:
        dump_config(data, str(yaml_path))
        return yaml_path
    except ImportError:
        return json_path


def _build(raw: Dict[str, Any], schema: Type[T]) -> T:
    """Construct *schema* from *raw* dict, rejecting unknown keys."""
    known = {f.name for f in dataclasses.fields(schema)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown config keys for {schema.__name__}: {sorted(unknown)}. "
            "Check for typos."
        )
    return schema(**{k: v for k, v in raw.items() if k in known})  # type: ignore[call-arg]
