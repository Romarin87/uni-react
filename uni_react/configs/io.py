"""YAML / JSON config I/O and CLI override merging."""
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

try:
    import yaml  # PyYAML
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

T = TypeVar("T")


def load_config(path: str, schema: Type[T]) -> T:
    """Load a YAML or JSON file and return a validated config dataclass.

    Only fields declared in *schema* are accepted; unknown keys raise
    :class:`ValueError` to catch typos early.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` or ``.json`` config file.
        schema: A dataclass type (e.g. :class:`~uni_react.configs.PretrainConfig`).

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


def dump_config(cfg: Any, path: str) -> None:
    """Serialise a config dataclass to YAML or JSON.

    The output format is inferred from the file extension.

    Args:
        cfg: A dataclass instance.
        path: Destination file path (``.yaml`` / ``.json``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(cfg)

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
