#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ELEMENT2Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract reaction triplets (R/TS/P) from xyz directories and write JSONL manifests. "
            "Expected per-reaction files: RG.xyz, TSG.xyz, PG.xyz (aliases supported)."
        )
    )
    p.add_argument("--roots", nargs="+", required=True, help="Reaction root directories to scan")
    p.add_argument("--output_jsonl", type=str, default="", help="Write all triplets to one JSONL")
    p.add_argument("--output_train_jsonl", type=str, default="", help="Write train split JSONL")
    p.add_argument("--output_val_jsonl", type=str, default="", help="Write val split JSONL")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio when writing train/val JSONL")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--relative_to", type=str, default="", help="Write relative paths to this base (default: absolute)")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any discovered reaction directory is missing one of R/TS/P xyz files",
    )
    return p.parse_args()


def atomic_number_from_token(token: str) -> int:
    s = token.strip()
    if not s:
        raise ValueError("empty atom token")
    if s.isdigit():
        return int(s)
    s = s[0].upper() + s[1:].lower()
    if s not in ELEMENT2Z:
        raise ValueError(f"unsupported element token: {token}")
    return ELEMENT2Z[s]


def parse_xyz_numbers(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"invalid xyz (too short): {path}")
    try:
        n_atoms = int(lines[0].split()[0])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"bad xyz first line: {path}: {lines[0]}") from exc
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"xyz atom count mismatch: {path}, expected={n_atoms}, got={len(atom_lines)}")
    z = np.zeros((n_atoms,), dtype=np.int64)
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"bad xyz atom line: {path}: {ln}")
        z[i] = atomic_number_from_token(parts[0])
    return z


def composition_signature(z: np.ndarray) -> List[Tuple[int, int]]:
    uniq, cnt = np.unique(z, return_counts=True)
    return [(int(u), int(c)) for u, c in zip(uniq.tolist(), cnt.tolist())]


def resolve_xyz(base_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = base_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def maybe_rel(path: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(path.resolve())
    return str(path.resolve().relative_to(base.resolve()))


def collect_triplets(roots: Sequence[str], strict: bool = False) -> Tuple[List[Dict], int]:
    triplets: List[Dict] = []
    strict_miss = 0
    seen_ids = set()

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"root not found: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"root is not a directory: {root_path}")

        for sub in sorted(root_path.iterdir()):
            if not sub.is_dir():
                continue
            r_path = resolve_xyz(sub, ("RG.xyz", "R.xyz", "reactant.xyz"))
            ts_path = resolve_xyz(sub, ("TSG.xyz", "TS.xyz", "transition_state.xyz"))
            p_path = resolve_xyz(sub, ("PG.xyz", "P.xyz", "product.xyz"))
            if r_path is None or ts_path is None or p_path is None:
                if strict:
                    strict_miss += 1
                continue

            z_r = parse_xyz_numbers(r_path)
            z_ts = parse_xyz_numbers(ts_path)
            z_p = parse_xyz_numbers(p_path)
            if z_r.shape[0] != z_ts.shape[0] or z_r.shape[0] != z_p.shape[0]:
                raise ValueError(
                    f"atom count mismatch in {sub}: R={z_r.shape[0]}, TS={z_ts.shape[0]}, P={z_p.shape[0]}"
                )
            sig_r = composition_signature(z_r)
            sig_ts = composition_signature(z_ts)
            sig_p = composition_signature(z_p)
            if sig_r != sig_ts or sig_r != sig_p:
                raise ValueError(
                    f"composition mismatch in {sub}: "
                    f"R={sig_r}, TS={sig_ts}, P={sig_p}"
                )

            rid = sub.name
            if rid in seen_ids:
                rid = f"{root_path.name}:{rid}"
            seen_ids.add(rid)
            triplets.append(
                {
                    "reaction_id": rid,
                    "root": str(root_path.resolve()),
                    "r_path": str(r_path.resolve()),
                    "ts_path": str(ts_path.resolve()),
                    "p_path": str(p_path.resolve()),
                    "n_atoms": int(z_r.shape[0]),
                    "composition": sig_r,
                }
            )

    return triplets, strict_miss


def write_jsonl(records: Sequence[Dict], out_path: Path, relative_base: Optional[Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = dict(rec)
            obj["r_path"] = maybe_rel(Path(obj["r_path"]), relative_base)
            obj["ts_path"] = maybe_rel(Path(obj["ts_path"]), relative_base)
            obj["p_path"] = maybe_rel(Path(obj["p_path"]), relative_base)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def split_train_val(records: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    records = list(records)
    if len(records) < 2 or val_ratio == 0.0:
        return records, []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(records) * val_ratio)))
    n_val = min(n_val, len(records) - 1)
    val_set = set(idx[:n_val].tolist())
    train, val = [], []
    for i, rec in enumerate(records):
        (val if i in val_set else train).append(rec)
    return train, val


def main() -> None:
    args = parse_args()
    if not args.output_jsonl and not (args.output_train_jsonl or args.output_val_jsonl):
        raise ValueError("Please provide --output_jsonl or train/val outputs.")

    relative_base = Path(args.relative_to).resolve() if args.relative_to else None

    triplets, strict_miss = collect_triplets(args.roots, strict=args.strict)
    if args.strict and strict_miss > 0:
        raise RuntimeError(f"Found {strict_miss} reaction directories missing one of R/TS/P xyz files in strict mode.")
    if len(triplets) == 0:
        raise RuntimeError("No valid triplets found.")

    if args.output_jsonl:
        write_jsonl(triplets, Path(args.output_jsonl), relative_base=relative_base)
        print(f"[saved] all={len(triplets)} -> {args.output_jsonl}")

    if args.output_train_jsonl or args.output_val_jsonl:
        train, val = split_train_val(triplets, val_ratio=args.val_ratio, seed=args.seed)
        if args.output_train_jsonl:
            write_jsonl(train, Path(args.output_train_jsonl), relative_base=relative_base)
            print(f"[saved] train={len(train)} -> {args.output_train_jsonl}")
        if args.output_val_jsonl:
            write_jsonl(val, Path(args.output_val_jsonl), relative_base=relative_base)
            print(f"[saved] val={len(val)} -> {args.output_val_jsonl}")

    n_atoms = np.asarray([int(x["n_atoms"]) for x in triplets], dtype=np.int64)
    print(
        "[summary] "
        f"triplets={len(triplets)}, n_atoms(min/mean/max)="
        f"{int(n_atoms.min())}/{float(n_atoms.mean()):.2f}/{int(n_atoms.max())}"
    )


if __name__ == "__main__":
    main()
