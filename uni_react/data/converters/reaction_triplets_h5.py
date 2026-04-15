#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
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
        description="Extract reaction triplets from xyz directories and write reaction-pretraining-ready HDF5."
    )
    p.add_argument("--roots", nargs="+", required=True, help="Reaction root directories to scan")
    p.add_argument("--output_h5", type=str, required=True, help="Output HDF5 path")
    p.add_argument(
        "--compression",
        type=str,
        default="lzf",
        choices=("none", "lzf", "gzip"),
        help="HDF5 compression mode",
    )
    p.add_argument("--strict", action="store_true", help="Fail if any reaction dir misses R/TS/P xyz")
    p.add_argument(
        "--allow_composition_mismatch",
        action="store_true",
        help="Allow R/TS/P with different compositions (default: disallow).",
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


def parse_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
    r = np.zeros((n_atoms, 3), dtype=np.float32)
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"bad xyz atom line: {path}: {ln}")
        z[i] = atomic_number_from_token(parts[0])
        r[i, 0] = float(parts[1])
        r[i, 1] = float(parts[2])
        r[i, 2] = float(parts[3])
    return z, r


def composition_signature(z: np.ndarray) -> List[Tuple[int, int]]:
    uniq, cnt = np.unique(z, return_counts=True)
    return [(int(u), int(c)) for u, c in zip(uniq.tolist(), cnt.tolist())]


def composition_hash(z: np.ndarray) -> int:
    sig = composition_signature(z)
    mod = (1 << 63) - 1
    h = 1469598103934665603
    for u, c in sig:
        h = (h * 1099511628211 + int(u) * 1000003 + int(c) * 9176) % mod
    return int(h)


def resolve_xyz(base_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = base_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def collect_triplets(
    roots: Sequence[str],
    strict: bool,
    allow_composition_mismatch: bool,
) -> Tuple[List[Dict], int]:
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

            z_r, _ = parse_xyz(r_path)
            z_ts, _ = parse_xyz(ts_path)
            z_p, _ = parse_xyz(p_path)
            if z_r.shape[0] != z_ts.shape[0] or z_r.shape[0] != z_p.shape[0]:
                raise ValueError(
                    f"atom count mismatch in {sub}: R={z_r.shape[0]}, TS={z_ts.shape[0]}, P={z_p.shape[0]}"
                )
            if not allow_composition_mismatch:
                sig_r = composition_signature(z_r)
                sig_ts = composition_signature(z_ts)
                sig_p = composition_signature(z_p)
                if sig_r != sig_ts or sig_r != sig_p:
                    raise ValueError(f"composition mismatch in {sub}: R={sig_r}, TS={sig_ts}, P={sig_p}")

            rid = sub.name
            if rid in seen_ids:
                rid = f"{root_path.name}:{rid}"
            seen_ids.add(rid)

            triplets.append(
                {
                    "reaction_id": rid,
                    "r_path": str(r_path.resolve()),
                    "ts_path": str(ts_path.resolve()),
                    "p_path": str(p_path.resolve()),
                    "n_atoms": int(z_r.shape[0]),
                    "comp_hash": int(composition_hash(z_r)),
                }
            )

    return triplets, strict_miss


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression

    triplets, strict_miss = collect_triplets(
        roots=args.roots,
        strict=args.strict,
        allow_composition_mismatch=args.allow_composition_mismatch,
    )
    if args.strict and strict_miss > 0:
        raise RuntimeError(f"Found {strict_miss} dirs missing R/TS/P xyz in strict mode.")
    if not triplets:
        raise RuntimeError("No valid triplets found.")

    n_triplets = len(triplets)
    n_atoms = np.asarray([int(x["n_atoms"]) for x in triplets], dtype=np.int32)
    offsets = np.zeros((n_triplets + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(n_atoms.astype(np.int64))
    total_atoms = int(offsets[-1])

    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as f:
        g = f.create_group("triplets")
        g.create_dataset("offsets", data=offsets, dtype=np.int64)
        g.create_dataset("n_atoms", data=n_atoms, dtype=np.int32)
        g.create_dataset("comp_hash", data=np.asarray([x["comp_hash"] for x in triplets], dtype=np.int64), dtype=np.int64)
        g.create_dataset("reaction_id", data=np.asarray([x["reaction_id"] for x in triplets], dtype=object), dtype=str_dt)

        ds = {}
        for key, shape, dtype in (
            ("r_Z", (total_atoms,), np.int16),
            ("ts_Z", (total_atoms,), np.int16),
            ("p_Z", (total_atoms,), np.int16),
            ("r_R", (total_atoms, 3), np.float32),
            ("ts_R", (total_atoms, 3), np.float32),
            ("p_R", (total_atoms, 3), np.float32),
        ):
            ds[key] = g.create_dataset(
                key,
                shape=shape,
                dtype=dtype,
                compression=compression,
                shuffle=True,
            )

        for i, item in enumerate(triplets):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            z_r, r_r = parse_xyz(Path(item["r_path"]))
            z_ts, r_ts = parse_xyz(Path(item["ts_path"]))
            z_p, r_p = parse_xyz(Path(item["p_path"]))
            ds["r_Z"][start:end] = z_r.astype(np.int16, copy=False)
            ds["ts_Z"][start:end] = z_ts.astype(np.int16, copy=False)
            ds["p_Z"][start:end] = z_p.astype(np.int16, copy=False)
            ds["r_R"][start:end] = r_r.astype(np.float32, copy=False)
            ds["ts_R"][start:end] = r_ts.astype(np.float32, copy=False)
            ds["p_R"][start:end] = r_p.astype(np.float32, copy=False)

        f.attrs["schema"] = "reaction_triplet_xyz_v1"
        f.attrs["num_triplets"] = int(n_triplets)
        f.attrs["total_atoms"] = int(total_atoms)
        f.attrs["roots"] = json.dumps([str(Path(x).resolve()) for x in args.roots], ensure_ascii=False)

    print(f"[saved] {out_path}")
    print(
        "[summary] "
        f"triplets={n_triplets}, total_atoms={total_atoms}, "
        f"n_atoms(min/mean/max)={int(n_atoms.min())}/{float(n_atoms.mean()):.2f}/{int(n_atoms.max())}"
    )


if __name__ == "__main__":
    main()
