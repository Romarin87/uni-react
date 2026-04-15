#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.qm9_dataset import QM9_TARGETS

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_SCRATCH_GLOB = "runs/qm9_scratch*"
DEFAULT_PRETRAINED_GLOB = "runs/qm9_pretrain*cdft*"
DEFAULT_GEOMETRIC_GLOB = "runs/qm9_pretrain*geometric*"
DEFAULT_CDFT_GLOB = "runs/qm9_pretrain*cdft*"
DEFAULT_REACTION_GLOB = "runs/qm9_pretrain*reaction*"
DEFAULT_GEOMETRIC_PREFIX = "runs/qm9_pretrain_geometric_"
LEGACY_STAGE1_PREFIX = "runs/qm9_pretrain_"
DEFAULT_CDFT_PREFIX = "runs/qm9_pretrain_cdft_"
DEFAULT_REACTION_PREFIX = "runs/qm9_pretrain_reaction_"
BACKBONE_PREFIXES = ("single_mol", "reacformer_se3", "reacformer_so2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare best-checkpoint train/val/test MAE between scratch and pretrained PyG QM9 finetune runs."
    )
    parser.add_argument(
        "--scratch_glob",
        type=str,
        default=DEFAULT_SCRATCH_GLOB,
        help=(
            "Glob for scratch run directories. Prefer family-aware globs rather than exact prefixes, "
            f'for example "{DEFAULT_SCRATCH_GLOB}".'
        ),
    )
    parser.add_argument(
        "--pretrained_glob",
        type=str,
        default=DEFAULT_PRETRAINED_GLOB,
        help=(
            "Glob for pretrained run directories. Prefer explicit family globs when multiple families/splits coexist. "
            f'Default uses CDFT pretraining runs: "{DEFAULT_PRETRAINED_GLOB}" '
            '(covers single_mol/ReacFormer family names; for geometric use e.g. '
            f'"{DEFAULT_GEOMETRIC_GLOB}").'
        ),
    )
    parser.add_argument(
        "--scratch_prefix",
        type=str,
        default="",
        help=(
            'Optional legacy prefix or modern family prefix for scratch runs, e.g. "runs/qm9_scratch_" '
            'or "runs/qm9_scratch_single_mol_egnn_". Prefer --scratch_glob unless you intentionally want one prefix.'
        ),
    )
    parser.add_argument(
        "--pretrained_prefix",
        type=str,
        default="",
        help=(
            'Optional legacy prefix or modern family prefix for pretrained runs, e.g. "runs/qm9_pretrain_geometric_" '
            'or "runs/qm9_pretrain_cdft_single_mol_egnn_". Prefer explicit globs unless you want one exact prefix.'
        ),
    )
    parser.add_argument(
        "--pretrained_label",
        type=str,
        default="pretrained",
        help='Label for primary pretrained results in table headers, default: "pretrained".',
    )
    parser.add_argument(
        "--pretrained2_glob",
        type=str,
        default="",
        help='Optional second pretrained glob, e.g. "runs/qm9_pretrain*cdft*".',
    )
    parser.add_argument(
        "--pretrained2_prefix",
        type=str,
        default="",
        help='Optional second pretrained exact prefix, e.g. "runs/qm9_pretrain_cdft_". Use only for one fixed naming family.',
    )
    parser.add_argument(
        "--pretrained2_label",
        type=str,
        default="pretrained2",
        help='Label for second pretrained results in table headers, default: "pretrained2".',
    )
    parser.add_argument(
        "--compare_geometric_cdft",
        action="store_true",
        help=(
            "Convenience mode: compare scratch vs geometric and scratch vs CDFT in one table. "
            f'Uses geometric glob "{DEFAULT_GEOMETRIC_GLOB}" '
            f'(fallback legacy "{LEGACY_STAGE1_PREFIX}") '
            f'and CDFT glob "{DEFAULT_CDFT_GLOB}".'
        ),
    )
    parser.add_argument(
        "--compare_geometric_cdft3",
        action="store_true",
        help=(
            "Convenience mode: compare scratch vs geometric/CDFT/reaction in one table. "
            f'Uses geometric glob "{DEFAULT_GEOMETRIC_GLOB}" '
            f'(fallback legacy "{LEGACY_STAGE1_PREFIX}"), '
            f'CDFT glob "{DEFAULT_CDFT_GLOB}", '
            f'and reaction glob "{DEFAULT_REACTION_GLOB}".'
        ),
    )
    parser.add_argument(
        "--pretrained3_glob",
        type=str,
        default="",
        help='Optional third pretrained glob, e.g. "runs/qm9_pretrain*reaction*".',
    )
    parser.add_argument(
        "--pretrained3_prefix",
        type=str,
        default="",
        help='Optional third pretrained exact prefix, e.g. "runs/qm9_pretrain_reaction_". Use only for one fixed naming family.',
    )
    parser.add_argument(
        "--pretrained3_label",
        type=str,
        default="pretrained3",
        help='Label for third pretrained results in table headers, default: "pretrained3".',
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save parsed comparison as JSON.",
    )
    parser.add_argument(
        "--plot_loss_dir",
        type=str,
        default="",
        help=(
            "Optional output directory for per-target loss trend plots. "
            "If empty, plotting is skipped."
        ),
    )
    parser.add_argument(
        "--plot_split",
        type=str,
        default="train",
        choices=("train", "val"),
        help='Which split to plot for trend curves, default: "train".',
    )
    parser.add_argument(
        "--plot_metric",
        type=str,
        default="loss",
        choices=("loss", "mae"),
        help='Which metric to plot for trend curves, default: "loss".',
    )
    parser.add_argument(
        "--plot_y_scale",
        type=str,
        default="linear",
        choices=("linear", "log"),
        help='Y-axis scale for trend curves, default: "linear".',
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=180,
        help="DPI for saved plot images, default: 180.",
    )
    return parser.parse_args()


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _format_float(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.6f}"


def _read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_config_map(run_dir: Path) -> Dict:
    json_path = run_dir / "config.json"
    if json_path.exists():
        return _read_json(json_path)

    yaml_path = run_dir / "config.yaml"
    if yaml_path.exists():
        if yaml is None:
            raise ImportError(
                f"PyYAML is required to read {yaml_path}. Install pyyaml or keep config.json."
            )
        with yaml_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise FileNotFoundError(
        f"Missing config file under {run_dir}. Expected config.json or config.yaml."
    )


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON line at {path}:{line_idx}") from exc
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


_LOG_LINE_RE = re.compile(r"^\[(?P<phase>[^\]]+)\](?: step=(?P<step>\d+))?(?: (?P<body>.*))?$")


def _coerce_log_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in raw and "e" not in lowered:
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def _read_plain_log(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            match = _LOG_LINE_RE.match(raw)
            if not match:
                continue
            record: Dict[str, object] = {"phase": match.group("phase")}
            step = match.group("step")
            if step is not None:
                record["step"] = int(step)
            body = match.group("body") or ""
            for token in body.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                record[key] = _coerce_log_value(value)
            records.append(record)
    return records


def _read_target(run_dir: Path) -> str:
    cfg = _read_config_map(run_dir)

    targets = cfg.get("targets_resolved")
    if isinstance(targets, list) and len(targets) == 1:
        return str(targets[0])
    if isinstance(targets, list) and len(targets) > 1:
        raise RuntimeError(
            f"QM9 compare only supports single-target runs, but {run_dir} contains multiple "
            f"resolved targets: {targets}"
        )

    raw_targets = cfg.get("targets")
    if isinstance(raw_targets, list):
        lowered = [str(item).lower() for item in raw_targets]
        if len(lowered) > 1 or (len(lowered) == 1 and lowered[0] == "all"):
            raise RuntimeError(
                f"QM9 compare only supports single-target runs, but {run_dir} is configured for "
                f"multiple targets: {raw_targets}"
            )

    run_name = run_dir.name.lower()
    out_dir = str(cfg.get("out_dir", "")).lower()
    if run_name.endswith("_multi") or out_dir.endswith("_multi"):
        raise RuntimeError(
            f"QM9 compare only supports single-target runs, but {run_dir} appears to be a multi-target run."
        )

    target = cfg.get("target")
    if isinstance(target, str):
        return target

    raise RuntimeError(f"Cannot determine single target from {run_dir}")


def _infer_family_from_text(text: str) -> Optional[str]:
    lowered = text.lower()
    if "qm9_scratch" in lowered:
        return "scratch"
    if "reaction" in lowered:
        return "pretrain_reaction"
    if "density" in lowered:
        return "pretrain_density"
    if "cdft" in lowered or "electronic" in lowered:
        return "pretrain_cdft"
    if "geometric" in lowered:
        return "pretrain_geometric"
    return None


def _infer_cfg_run_family(cfg: Dict, run_dir: Path) -> str:
    ckpt = cfg.get("pretrained_ckpt")
    if isinstance(ckpt, str) and ckpt:
        inferred = _infer_family_from_text(ckpt)
        if inferred is not None:
            return inferred
    out_dir = cfg.get("out_dir")
    if isinstance(out_dir, str) and out_dir:
        inferred = _infer_family_from_text(out_dir)
        if inferred is not None:
            return inferred
    inferred = _infer_family_from_text(str(run_dir))
    if inferred is not None:
        return inferred
    if not isinstance(ckpt, str) or not ckpt:
        return "scratch"

    ckpt_path = Path(ckpt)
    source = (
        ckpt_path.parent.name
        if ckpt_path.stem in {"best", "latest"} or ckpt_path.stem.startswith("epoch_")
        else ckpt_path.stem
    )
    token = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_") or "custom"
    return f"pretrain_{token}"


def _run_semantic_signature(metrics: Dict) -> Tuple[str, str]:
    return (
        str(metrics.get("split", "")),
        str(metrics.get("run_family", "")),
    )


def _derive_metrics_from_train_log(run_dir: Path, target: str) -> Dict:
    records = _read_plain_log(run_dir / "train.log")
    train_by_epoch: Dict[int, Dict] = {}
    val_by_epoch: Dict[int, Dict] = {}
    best_epoch: Optional[int] = None
    test_record: Optional[Dict] = None

    for item in records:
        phase = item.get("phase")
        if phase == "train" and isinstance(item.get("step"), int):
            train_by_epoch[int(item["step"])] = item
        elif phase == "val" and isinstance(item.get("step"), int):
            val_by_epoch[int(item["step"])] = item
        elif phase == "epoch":
            epoch = item.get("epoch")
            if isinstance(epoch, int) and item.get("is_best") is True:
                best_epoch = epoch
        elif phase == "test":
            test_record = item

    if best_epoch is None:
        candidates = sorted(set(train_by_epoch) | set(val_by_epoch))
        if not candidates:
            raise RuntimeError(f"Cannot derive best epoch from {run_dir / 'train.log'}")
        best_epoch = candidates[-1]

    train_metrics = train_by_epoch.get(best_epoch, {})
    val_metrics = val_by_epoch.get(best_epoch, {})
    return {
        "run_dir": str(run_dir),
        "epoch": best_epoch,
        "target": target,
        "train_mae": _safe_float(train_metrics.get("mae")),
        "val_mae": _safe_float(val_metrics.get("mae")),
        "test_mae": None if test_record is None else _safe_float(test_record.get("mae")),
    }


def _load_run_metrics(run_dir: Path) -> Dict:
    cfg = _read_config_map(run_dir)
    target = _read_target(run_dir)
    run_family = _infer_cfg_run_family(cfg, run_dir)
    split = str(cfg.get("split", ""))
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        result = _derive_metrics_from_train_log(run_dir, target)
        result["split"] = split
        result["run_family"] = run_family
        return result
    record = _read_json(metrics_path)
    train = record.get("train", {}) or {}
    val = record.get("val", {}) or {}
    test = record.get("test", {}) or {}

    return {
        "run_dir": str(run_dir),
        "epoch": int(record.get("best_epoch", -1)),
        "target": target,
        "split": split,
        "run_family": run_family,
        "train_mae": _safe_float(train.get("mae")),
        "val_mae": _safe_float(val.get("mae")) if isinstance(val, dict) else None,
        "test_mae": _safe_float(test.get("mae")) if isinstance(test, dict) else None,
    }


def _target_dir_suffix(target: str) -> str:
    return target.lower()


def _expand_backbone_patterns(pattern: str) -> List[str]:
    replacements = {
        "runs/qm9_scratch_": [
            "runs/qm9_scratch_",
            "runs/qm9_scratch_reacformer_se3_",
            "runs/qm9_scratch_reacformer_so2_",
        ],
        "runs/qm9_scratch_*": [
            "runs/qm9_scratch_*",
            "runs/qm9_scratch_reacformer_se3_*",
            "runs/qm9_scratch_reacformer_so2_*",
        ],
        "runs/qm9_pretrain_geometric_": [
            "runs/qm9_pretrain_geometric_",
            "runs/qm9_pretrain_reacformer_se3_geometric_",
            "runs/qm9_pretrain_reacformer_so2_geometric_",
        ],
        "runs/qm9_pretrain_geometric_*": [
            "runs/qm9_pretrain_geometric_*",
            "runs/qm9_pretrain_reacformer_se3_geometric_*",
            "runs/qm9_pretrain_reacformer_so2_geometric_*",
        ],
        "runs/qm9_pretrain_cdft_": [
            "runs/qm9_pretrain_cdft_",
            "runs/qm9_pretrain_reacformer_se3_cdft_",
            "runs/qm9_pretrain_reacformer_so2_cdft_",
        ],
        "runs/qm9_pretrain_cdft_*": [
            "runs/qm9_pretrain_cdft_*",
            "runs/qm9_pretrain_reacformer_se3_cdft_*",
            "runs/qm9_pretrain_reacformer_so2_cdft_*",
        ],
        "runs/qm9_pretrain_reaction_": [
            "runs/qm9_pretrain_reaction_",
            "runs/qm9_pretrain_reacformer_se3_reaction_",
            "runs/qm9_pretrain_reacformer_so2_reaction_",
        ],
        "runs/qm9_pretrain_reaction_*": [
            "runs/qm9_pretrain_reaction_*",
            "runs/qm9_pretrain_reacformer_se3_reaction_*",
            "runs/qm9_pretrain_reacformer_so2_reaction_*",
        ],
    }
    if pattern in replacements:
        return replacements[pattern]

    patterns = [pattern]
    if "single_mol" in pattern:
        for backbone in BACKBONE_PREFIXES[1:]:
            patterns.append(pattern.replace("single_mol", backbone))
    return patterns


def _select_preferred_run(existing: Dict, candidate: Dict, *, context: str) -> Dict:
    existing_sig = _run_semantic_signature(existing)
    candidate_sig = _run_semantic_signature(candidate)
    if existing_sig != candidate_sig:
        raise ValueError(
            f"Ambiguous QM9 compare selection for {context}: duplicate target "
            f"'{existing['target']}' has incompatible run metadata "
            f"{existing_sig} vs {candidate_sig}. Please pass an explicit prefix/glob."
        )
    existing_path = Path(existing["run_dir"])
    candidate_path = Path(candidate["run_dir"])
    existing_key = (
        existing_path.stat().st_mtime if existing_path.exists() else float("-inf"),
        existing.get("epoch", -1),
        str(existing_path),
    )
    candidate_key = (
        candidate_path.stat().st_mtime if candidate_path.exists() else float("-inf"),
        candidate.get("epoch", -1),
        str(candidate_path),
    )
    preferred = candidate if candidate_key > existing_key else existing
    dropped = existing if preferred is candidate else candidate
    print(
        f"[compare_qm9_mae] duplicate target '{preferred['target']}' in {context}; "
        f"using {preferred['run_dir']} and ignoring {dropped['run_dir']}",
        file=sys.stderr,
    )
    return preferred


def _infer_backbone_family(run_dir: str) -> str:
    if "reacformer_se3" in run_dir:
        return "reacformer_se3"
    if "reacformer_so2" in run_dir:
        return "reacformer_so2"
    return "single_mol"


def _run_preference_key(metrics: Dict) -> Tuple[float, int, str]:
    run_dir = Path(metrics["run_dir"])
    return (
        run_dir.stat().st_mtime if run_dir.exists() else float("-inf"),
        int(metrics.get("epoch", -1)),
        str(run_dir),
    )


def _select_backbone_family(family_runs: Dict[str, Dict[str, Dict]], *, context: str) -> Dict[str, Dict]:
    if not family_runs:
        raise RuntimeError(f"No runs collected for {context}")

    ranked = sorted(
        family_runs.items(),
        key=lambda item: (
            len(item[1]),
            max((_run_preference_key(metrics) for metrics in item[1].values()), default=(float("-inf"), -1, "")),
            item[0],
        ),
        reverse=True,
    )
    top_family, top_runs = ranked[0]
    top_count = len(top_runs)
    tied = [family for family, runs in ranked if len(runs) == top_count]
    if len(tied) > 1:
        raise ValueError(
            f"Ambiguous QM9 compare selection for {context}: multiple backbone families have the same "
            f"target coverage {top_count}: {tied}. Please pass an explicit prefix/glob for one family."
        )
    if len(ranked) > 1:
        print(
            f"[compare_qm9_mae] multiple backbone families found in {context}; "
            f"using family '{top_family}' with {top_count} targets",
            file=sys.stderr,
        )
    return top_runs


def _collect_runs(pattern: str) -> Dict[str, Dict]:
    expanded_patterns = _expand_backbone_patterns(pattern)
    run_dirs = sorted({
        run_dir
        for current_pattern in expanded_patterns
        for run_dir in Path().glob(current_pattern)
    })
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched: {pattern}")

    family_runs: Dict[str, Dict[str, Dict]] = {}
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        metrics = _load_run_metrics(run_dir)
        family = _infer_backbone_family(metrics["run_dir"])
        out = family_runs.setdefault(family, {})
        target = metrics["target"]
        if target in out:
            out[target] = _select_preferred_run(out[target], metrics, context=f"pattern {pattern}")
            continue
        out[target] = metrics
    if not family_runs:
        raise RuntimeError(f"No valid run directories matched: {pattern}")
    return _select_backbone_family(family_runs, context=f"pattern {pattern}")


def _collect_runs_by_prefix(prefix_path: str) -> Dict[str, Dict]:
    family_runs: Dict[str, Dict[str, Dict]] = {}
    for current_prefix in _expand_backbone_patterns(prefix_path):
        for target in QM9_TARGETS:
            suffix = _target_dir_suffix(target)
            candidates = sorted({
                *Path().glob(f"{current_prefix}{suffix}"),
                *Path().glob(f"{current_prefix}*_{suffix}"),
            })
            for run_dir in candidates:
                if not run_dir.is_dir():
                    continue
                metrics = _load_run_metrics(run_dir)
                family = _infer_backbone_family(metrics["run_dir"])
                out = family_runs.setdefault(family, {})
                found_target = metrics["target"]
                if found_target in out:
                    out[found_target] = _select_preferred_run(
                        out[found_target],
                        metrics,
                        context=f"prefix {prefix_path}",
                    )
                    continue
                out[found_target] = metrics

    if not family_runs:
        raise FileNotFoundError(
            f"No run directories matched exact prefix {prefix_path}. "
            "Expected names like "
            f"{prefix_path}{_target_dir_suffix(QM9_TARGETS[0])} or "
            f"{prefix_path}*_{_target_dir_suffix(QM9_TARGETS[0])}."
        )
    return _select_backbone_family(family_runs, context=f"prefix {prefix_path}")


def _collect_geometric_with_fallback() -> Dict[str, Dict]:
    try:
        return _collect_runs(DEFAULT_GEOMETRIC_GLOB)
    except FileNotFoundError:
        return _collect_runs_by_prefix(LEGACY_STAGE1_PREFIX)


def _ordered_targets(*metric_maps: Dict[str, Dict]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    all_targets = set()
    for metric_map in metric_maps:
        all_targets.update(metric_map.keys())

    for target in QM9_TARGETS:
        if target in all_targets and target not in seen:
            ordered.append(target)
            seen.add(target)
    for target in sorted(all_targets):
        if target not in seen:
            ordered.append(target)
            seen.add(target)
    return ordered


def _build_rows(
    scratch: Dict[str, Dict],
    pretrained: Dict[str, Dict],
    pretrained2: Optional[Dict[str, Dict]] = None,
    pretrained3: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    rows: List[Dict] = []
    maps = [scratch, pretrained]
    if pretrained2 is not None:
        maps.append(pretrained2)
    if pretrained3 is not None:
        maps.append(pretrained3)
    for target in _ordered_targets(*maps):
        s = scratch.get(target)
        p = pretrained.get(target)
        p2 = None if pretrained2 is None else pretrained2.get(target)
        p3 = None if pretrained3 is None else pretrained3.get(target)
        s_train = None if s is None else s["train_mae"]
        s_val = None if s is None else s["val_mae"]
        p_train = None if p is None else p["train_mae"]
        p_val = None if p is None else p["val_mae"]
        row = {
            "target": target,
            "scratch_epoch": None if s is None else s["epoch"],
            "scratch_train_mae": s_train,
            "scratch_val_mae": s_val,
            "scratch_test_mae": None if s is None else s["test_mae"],
            "pretrained_epoch": None if p is None else p["epoch"],
            "pretrained_train_mae": p_train,
            "pretrained_val_mae": p_val,
            "pretrained_test_mae": None if p is None else p["test_mae"],
            "delta_train_mae": None if (s_train is None or p_train is None) else (p_train - s_train),
            "delta_val_mae": None if (s_val is None or p_val is None) else (p_val - s_val),
            "delta_test_mae": (
                None
                if (s is None or p is None or s["test_mae"] is None or p["test_mae"] is None)
                else (p["test_mae"] - s["test_mae"])
            ),
        }
        if pretrained2 is not None:
            p2_train = None if p2 is None else p2["train_mae"]
            p2_val = None if p2 is None else p2["val_mae"]
            row.update(
                {
                    "pretrained2_epoch": None if p2 is None else p2["epoch"],
                    "pretrained2_train_mae": p2_train,
                    "pretrained2_val_mae": p2_val,
                    "pretrained2_test_mae": None if p2 is None else p2["test_mae"],
                    "delta2_train_mae": None if (s_train is None or p2_train is None) else (p2_train - s_train),
                    "delta2_val_mae": None if (s_val is None or p2_val is None) else (p2_val - s_val),
                    "delta2_test_mae": (
                        None
                        if (s is None or p2 is None or s["test_mae"] is None or p2["test_mae"] is None)
                        else (p2["test_mae"] - s["test_mae"])
                    ),
                }
            )
        if pretrained3 is not None:
            p3_train = None if p3 is None else p3["train_mae"]
            p3_val = None if p3 is None else p3["val_mae"]
            row.update(
                {
                    "pretrained3_epoch": None if p3 is None else p3["epoch"],
                    "pretrained3_train_mae": p3_train,
                    "pretrained3_val_mae": p3_val,
                    "pretrained3_test_mae": None if p3 is None else p3["test_mae"],
                    "delta3_train_mae": None if (s_train is None or p3_train is None) else (p3_train - s_train),
                    "delta3_val_mae": None if (s_val is None or p3_val is None) else (p3_val - s_val),
                    "delta3_test_mae": (
                        None
                        if (s is None or p3 is None or s["test_mae"] is None or p3["test_mae"] is None)
                        else (p3["test_mae"] - s["test_mae"])
                    ),
                }
            )
        rows.append(row)
    return rows


def _sanitize_header_label(label: str) -> str:
    sanitized = "_".join(label.strip().split())
    return sanitized if sanitized else "pretrained"


def _load_epoch_metric_series(run_dir: Path, split: str, metric: str) -> List[Tuple[int, float]]:
    jsonl_path = run_dir / "train_log.jsonl"
    points: List[Tuple[int, float]] = []
    if jsonl_path.exists():
        records = _read_jsonl(jsonl_path)
        for item in records:
            epoch_raw = item.get("epoch")
            if epoch_raw is None:
                continue
            try:
                epoch = int(epoch_raw)
            except (TypeError, ValueError):
                continue
            split_metrics = item.get(split, {})
            if not isinstance(split_metrics, dict):
                continue
            val = _safe_float(split_metrics.get(metric))
            if val is None:
                continue
            points.append((epoch, val))
    else:
        plain_records = _read_plain_log(run_dir / "train.log")
        expected_phase = "train" if split == "train" else "val"
        for item in plain_records:
            if item.get("phase") != expected_phase:
                continue
            epoch = item.get("step")
            if not isinstance(epoch, int):
                continue
            val = _safe_float(item.get(metric))
            if val is None:
                continue
            points.append((epoch, val))
    points.sort(key=lambda x: x[0])
    return points


def _plot_metric_curves(
    model_maps: Sequence[Tuple[str, Dict[str, Dict]]],
    out_dir: Path,
    split: str,
    metric: str,
    y_scale: str,
    dpi: int,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Please install it in current env."
        ) from exc

    metric_maps = [m for _, m in model_maps]
    targets = _ordered_targets(*metric_maps)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    for target in targets:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for label, metric_map in model_maps:
            run_meta = metric_map.get(target)
            if run_meta is None:
                continue
            run_dir = Path(run_meta["run_dir"])
            try:
                points = _load_epoch_metric_series(run_dir, split=split, metric=metric)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[plot skip] {target} {label}: {exc}")
                continue
            if not points:
                continue
            if y_scale == "log":
                points = [(x, y) for x, y in points if y > 0.0]
                if not points:
                    print(f"[plot skip] {target} {label}: no positive values for log-scale y-axis")
                    continue
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            ax.plot(xs, ys, linewidth=2.0, label=label)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_title(f"QM9 {target}: {split}_{metric} trend")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{split}_{metric}")
        ax.set_yscale(y_scale)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        output_path = out_dir / f"{target}_{split}_{metric}_{y_scale}.png"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        generated.append(str(output_path))
    return generated


def _print_table(
    rows: Sequence[Dict],
    pretrained_label: str = "pretrained",
    pretrained2_label: Optional[str] = None,
    pretrained3_label: Optional[str] = None,
) -> None:
    p1 = _sanitize_header_label(pretrained_label)
    headers = [
        "target",
        "scratch_train",
        "scratch_val",
        "scratch_test",
        f"{p1}_train",
        f"{p1}_val",
        f"{p1}_test",
        f"delta_{p1}_train",
        f"delta_{p1}_val",
        f"delta_{p1}_test",
    ]
    has_pretrained2 = pretrained2_label is not None
    has_pretrained3 = pretrained3_label is not None
    if has_pretrained2:
        p2 = _sanitize_header_label(pretrained2_label)
        headers.extend(
            [
                f"{p2}_train",
                f"{p2}_val",
                f"{p2}_test",
                f"delta_{p2}_train",
                f"delta_{p2}_val",
                f"delta_{p2}_test",
            ]
        )
    if has_pretrained3:
        p3 = _sanitize_header_label(pretrained3_label)
        headers.extend(
            [
                f"{p3}_train",
                f"{p3}_val",
                f"{p3}_test",
                f"delta_{p3}_train",
                f"delta_{p3}_val",
                f"delta_{p3}_test",
            ]
        )

    table_rows: List[List[str]] = []
    for row in rows:
        cells = [
            row["target"],
            _format_float(row["scratch_train_mae"]),
            _format_float(row["scratch_val_mae"]),
            _format_float(row["scratch_test_mae"]),
            _format_float(row["pretrained_train_mae"]),
            _format_float(row["pretrained_val_mae"]),
            _format_float(row["pretrained_test_mae"]),
            _format_float(row["delta_train_mae"]),
            _format_float(row["delta_val_mae"]),
            _format_float(row["delta_test_mae"]),
        ]
        if has_pretrained2:
            cells.extend(
                [
                    _format_float(row.get("pretrained2_train_mae")),
                    _format_float(row.get("pretrained2_val_mae")),
                    _format_float(row.get("pretrained2_test_mae")),
                    _format_float(row.get("delta2_train_mae")),
                    _format_float(row.get("delta2_val_mae")),
                    _format_float(row.get("delta2_test_mae")),
                ]
            )
        if has_pretrained3:
            cells.extend(
                [
                    _format_float(row.get("pretrained3_train_mae")),
                    _format_float(row.get("pretrained3_val_mae")),
                    _format_float(row.get("pretrained3_test_mae")),
                    _format_float(row.get("delta3_train_mae")),
                    _format_float(row.get("delta3_val_mae")),
                    _format_float(row.get("delta3_test_mae")),
                ]
            )
        table_rows.append(cells)

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(cells: Iterable[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_line(headers))
    print(fmt_line("-" * w for w in widths))
    for row in table_rows:
        print(fmt_line(row))


def main() -> None:
    args = parse_args()
    scratch = _collect_runs_by_prefix(args.scratch_prefix) if args.scratch_prefix else _collect_runs(args.scratch_glob)
    if args.compare_geometric_cdft3:
        pretrained = _collect_geometric_with_fallback()
        pretrained2 = _collect_runs(DEFAULT_CDFT_GLOB)
        pretrained3 = _collect_runs(DEFAULT_REACTION_GLOB)
        pretrained_label = "geometric"
        pretrained2_label = "cdft"
        pretrained3_label = "reaction"
    elif args.compare_geometric_cdft:
        pretrained = _collect_geometric_with_fallback()
        pretrained2 = _collect_runs(DEFAULT_CDFT_GLOB)
        pretrained3 = None
        pretrained_label = "geometric"
        pretrained2_label = "cdft"
        pretrained3_label = None
    else:
        pretrained = (
            _collect_runs_by_prefix(args.pretrained_prefix)
            if args.pretrained_prefix
            else _collect_runs(args.pretrained_glob)
        )
        has_pretrained2 = bool(args.pretrained2_prefix or args.pretrained2_glob)
        has_pretrained3 = bool(args.pretrained3_prefix or args.pretrained3_glob)
        pretrained2 = (
            _collect_runs_by_prefix(args.pretrained2_prefix)
            if args.pretrained2_prefix
            else (_collect_runs(args.pretrained2_glob) if has_pretrained2 else None)
        )
        pretrained3 = (
            _collect_runs_by_prefix(args.pretrained3_prefix)
            if args.pretrained3_prefix
            else (_collect_runs(args.pretrained3_glob) if has_pretrained3 else None)
        )
        pretrained_label = args.pretrained_label
        pretrained2_label = args.pretrained2_label if pretrained2 is not None else None
        pretrained3_label = args.pretrained3_label if pretrained3 is not None else None

    rows = _build_rows(scratch, pretrained, pretrained2=pretrained2, pretrained3=pretrained3)
    _print_table(
        rows,
        pretrained_label=pretrained_label,
        pretrained2_label=pretrained2_label,
        pretrained3_label=pretrained3_label,
    )

    generated_plots: List[str] = []
    if args.plot_loss_dir:
        model_maps: List[Tuple[str, Dict[str, Dict]]] = [("scratch", scratch), (pretrained_label, pretrained)]
        if pretrained2 is not None and pretrained2_label is not None:
            model_maps.append((pretrained2_label, pretrained2))
        if pretrained3 is not None and pretrained3_label is not None:
            model_maps.append((pretrained3_label, pretrained3))
        generated_plots = _plot_metric_curves(
            model_maps=model_maps,
            out_dir=Path(args.plot_loss_dir),
            split=args.plot_split,
            metric=args.plot_metric,
            y_scale=args.plot_y_scale,
            dpi=args.plot_dpi,
        )
        print(f"\nSaved {len(generated_plots)} plots to {args.plot_loss_dir}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "compare_geometric_cdft": args.compare_geometric_cdft,
            "compare_geometric_cdft3": args.compare_geometric_cdft3,
            "scratch_glob": args.scratch_glob,
            "pretrained_glob": args.pretrained_glob,
            "scratch_prefix": args.scratch_prefix,
            "pretrained_prefix": args.pretrained_prefix,
            "pretrained_label": pretrained_label,
            "pretrained2_glob": args.pretrained2_glob,
            "pretrained2_prefix": args.pretrained2_prefix,
            "pretrained2_label": pretrained2_label,
            "pretrained3_glob": args.pretrained3_glob,
            "pretrained3_prefix": args.pretrained3_prefix,
            "pretrained3_label": pretrained3_label,
            "plot_loss_dir": args.plot_loss_dir,
            "plot_split": args.plot_split,
            "plot_metric": args.plot_metric,
            "plot_y_scale": args.plot_y_scale,
            "generated_plots": generated_plots,
            "rows": rows,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON to {out_path}")


if __name__ == "__main__":
    main()
