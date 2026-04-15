#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stream-convert ED tar.gz archives into sharded HDF5 for density pretraining.

Input (inside each tar.gz sample folder):
  - Mol1_Dt.cube
  - *_Psi4.out

Key constraints:
  - No extraction to disk (stream read tar members).
  - Keep CHON-only molecules by default.

Output schema (variable-size flattened arrays):
  /frames/mol_id                 (str, [n_frames])
  /frames/source_tar             (str, [n_frames])
  /frames/source_prefix          (str, [n_frames])
  /frames/total_charge           (int16, [n_frames])
  /frames/spin_multiplicity      (int16, [n_frames])
  /frames/grid_shape             (int32, [n_frames, 3])
  /frames/grid_origin            (float32, [n_frames, 3])        # Angstrom
  /frames/grid_vectors           (float32, [n_frames, 3, 3])     # Angstrom
  /frames/voxel_volume           (float32, [n_frames])           # Angstrom^3
  /frames/atom_offsets           (int64, [n_frames])
  /frames/n_atoms                (int32, [n_frames])
  /frames/density_offsets        (int64, [n_frames])
  /frames/n_voxels               (int64, [n_frames])
  /atoms/Z                       (uint8, [sum(n_atoms)])
  /atoms/R                       (float32, [sum(n_atoms), 3])    # Angstrom
  /density/target                (float32, [sum(n_voxels)])      # e/Angstrom^3
"""

from __future__ import annotations

import argparse
import glob
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


BOHR_TO_ANG = 0.529177210903
ANG3_PER_BOHR3 = BOHR_TO_ANG ** 3
E_PER_ANG3_PER_E_PER_BOHR3 = 1.0 / ANG3_PER_BOHR3

_SYMBOL_TO_Z = {
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
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


@dataclass
class Sample:
    mol_id: str
    source_tar: str
    source_prefix: str
    total_charge: int
    spin_multiplicity: int
    z: np.ndarray
    r_ang: np.ndarray
    grid_shape: np.ndarray
    grid_origin_ang: np.ndarray
    grid_vectors_ang: np.ndarray
    voxel_volume_ang3: float
    density_target_e_per_ang3: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert ED tar.gz archives to HDF5 shards without extraction.")
    ap.add_argument(
        "--tar_glob",
        nargs="+",
        required=True,
        help="One or more glob patterns for input .tar.gz files.",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for HDF5 shards.")
    ap.add_argument("--prefix", type=str, default="ed_density", help="Output shard prefix.")
    ap.add_argument("--samples_per_shard", type=int, default=8192, help="Max samples per output shard.")
    ap.add_argument(
        "--max_voxels_per_shard",
        type=int,
        default=500_000_000,
        help="Flush shard once total flattened voxel count reaches this threshold.",
    )
    ap.add_argument(
        "--allowed_elements",
        type=str,
        default="H,C,N,O",
        help="Comma-separated allowed elements. Samples containing other elements are dropped.",
    )
    ap.add_argument(
        "--coord_unit_in_cube",
        type=str,
        choices=["bohr", "angstrom", "auto"],
        default="auto",
        help="Coordinate/grid unit in cube files.",
    )
    ap.add_argument(
        "--density_unit_in_cube",
        type=str,
        choices=["e/bohr^3", "e/angstrom^3", "auto"],
        default="auto",
        help="Density unit in cube files.",
    )
    ap.add_argument(
        "--compression",
        type=str,
        choices=["none", "gzip", "lzf"],
        default="gzip",
        help="HDF5 compression for dense datasets.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of kept samples for quick test.")
    ap.add_argument("--verbose_every", type=int, default=500, help="Progress log interval in kept samples.")
    return ap.parse_args()


def _expand_tar_paths(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for pat in patterns:
        for item in sorted(glob.glob(pat)):
            p = Path(item)
            if p.is_file() and p.suffixes[-2:] == [".tar", ".gz"] and str(p) not in seen:
                out.append(p)
                seen.add(str(p))
    if not out:
        raise FileNotFoundError(f"No .tar.gz files matched patterns: {patterns}")
    return out


def _parse_allowed_elements(spec: str) -> Tuple[set, set]:
    syms = [x.strip() for x in spec.split(",") if x.strip()]
    allowed_z = set()
    normalized = set()
    for sym in syms:
        norm = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        z = _SYMBOL_TO_Z.get(norm)
        if z is None:
            raise ValueError(f"Unknown element symbol in --allowed_elements: {sym!r}")
        normalized.add(norm)
        allowed_z.add(int(z))
    if not allowed_z:
        raise ValueError("--allowed_elements cannot be empty")
    return normalized, allowed_z


def _detect_coord_scale(line2: str, mode: str) -> float:
    if mode == "bohr":
        return BOHR_TO_ANG
    if mode == "angstrom":
        return 1.0
    low = line2.lower()
    if "a0" in low or "bohr" in low:
        return BOHR_TO_ANG
    if "angstrom" in low or "ang" in low:
        return 1.0
    # Conservative default for Psi4 cube.
    return BOHR_TO_ANG


def _detect_density_scale(line2: str, mode: str) -> float:
    if mode == "e/bohr^3":
        return E_PER_ANG3_PER_E_PER_BOHR3
    if mode == "e/angstrom^3":
        return 1.0
    low = line2.lower().replace(" ", "")
    if "e/a0^3" in low or "e/bohr^3" in low:
        return E_PER_ANG3_PER_E_PER_BOHR3
    if "e/ang^3" in low or "e/angstrom^3" in low:
        return 1.0
    # Conservative default for Psi4 cube.
    return E_PER_ANG3_PER_E_PER_BOHR3


def _parse_cube(
    cube_bytes: bytes,
    coord_unit_mode: str,
    density_unit_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    text = cube_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 7:
        raise ValueError("Cube file too short.")

    line2 = lines[1].strip() if len(lines) > 1 else ""
    coord_scale = _detect_coord_scale(line2=line2, mode=coord_unit_mode)
    density_scale = _detect_density_scale(line2=line2, mode=density_unit_mode)

    try:
        third = lines[2].split()
        natoms_raw = int(float(third[0]))
        natoms = abs(natoms_raw)
        origin = np.asarray([float(third[1]), float(third[2]), float(third[3])], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to parse cube header line 3: {lines[2]!r}") from exc
    if natoms <= 0:
        raise ValueError(f"Invalid natoms in cube: {natoms_raw}")

    grid_shape = np.empty((3,), dtype=np.int32)
    grid_vectors = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        toks = lines[3 + i].split()
        if len(toks) < 4:
            raise ValueError(f"Bad grid line in cube: {lines[3 + i]!r}")
        grid_shape[i] = abs(int(float(toks[0])))
        grid_vectors[i] = np.asarray([float(toks[1]), float(toks[2]), float(toks[3])], dtype=np.float64)
        if grid_shape[i] <= 0:
            raise ValueError(f"Invalid grid size in cube line: {lines[3 + i]!r}")

    atom_start = 6
    atom_end = atom_start + natoms
    if len(lines) < atom_end:
        raise ValueError("Cube ended before atom block completed.")

    z = np.empty((natoms,), dtype=np.uint8)
    r = np.empty((natoms, 3), dtype=np.float64)
    for i, line in enumerate(lines[atom_start:atom_end]):
        toks = line.split()
        if len(toks) < 5:
            raise ValueError(f"Bad atom line in cube: {line!r}")
        zi = int(round(float(toks[0])))
        if zi <= 0 or zi > 255:
            raise ValueError(f"Unsupported atomic number in cube atom line: {line!r}")
        z[i] = zi
        r[i] = np.asarray([float(toks[2]), float(toks[3]), float(toks[4])], dtype=np.float64)

    density_str = " ".join(lines[atom_end:])
    density = np.fromstring(density_str, sep=" ", dtype=np.float64)
    nvox = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
    if density.size != nvox:
        raise ValueError(f"Density size mismatch in cube: got {density.size}, expected {nvox}")

    origin_ang = (origin * coord_scale).astype(np.float32, copy=False)
    vectors_ang = (grid_vectors * coord_scale).astype(np.float32, copy=False)
    r_ang = (r * coord_scale).astype(np.float32, copy=False)
    density_e_per_ang3 = (density * density_scale).astype(np.float32, copy=False)
    voxel_volume = float(abs(np.linalg.det(vectors_ang.astype(np.float64))))

    return z, r_ang, grid_shape, origin_ang, vectors_ang, voxel_volume, density_e_per_ang3


def _parse_charge_mult(out_bytes: bytes) -> Tuple[int, int]:
    text = out_bytes.decode("utf-8", errors="replace")

    m = re.search(r"charge\s*=\s*([-+]?\d+)\s*,\s*multiplicity\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    m_charge = re.search(r"^\s*Charge\s*=\s*([-+]?\d+)\s*$", text, re.IGNORECASE | re.MULTILINE)
    m_mult = re.search(r"^\s*Multiplicity\s*=\s*(\d+)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m_charge and m_mult:
        return int(m_charge.group(1)), int(m_mult.group(1))

    raise ValueError("Cannot parse charge/multiplicity from Psi4 output.")


def _iter_pairs_from_tar(tar_path: Path) -> Iterable[Tuple[str, bytes, bytes]]:
    pending: Dict[str, Dict[str, bytes]] = {}
    with tarfile.open(tar_path, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name.strip("/")
            if not name:
                continue
            base = Path(name).name
            prefix = str(Path(name).parent)

            # Skip macOS AppleDouble resource-fork files (._*)
            if base.startswith("._"):
                continue

            if base != "Mol1_Dt.cube" and not base.endswith("_Psi4.out"):
                continue

            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            payload = fobj.read()
            entry = pending.setdefault(prefix, {})
            if base == "Mol1_Dt.cube":
                entry["cube"] = payload
            else:
                entry["out"] = payload

            if "cube" in entry and "out" in entry:
                yield prefix, entry["cube"], entry["out"]
                del pending[prefix]


def _write_shard(
    out_path: Path,
    samples: Sequence[Sample],
    compression: Optional[str],
) -> None:
    if not samples:
        return

    n = len(samples)
    atom_counts = np.asarray([s.z.shape[0] for s in samples], dtype=np.int32)
    atom_offsets = np.empty((n,), dtype=np.int64)
    atom_offsets[0] = 0
    if n > 1:
        atom_offsets[1:] = np.cumsum(atom_counts[:-1], dtype=np.int64)

    voxel_counts = np.asarray([s.density_target_e_per_ang3.shape[0] for s in samples], dtype=np.int64)
    voxel_offsets = np.empty((n,), dtype=np.int64)
    voxel_offsets[0] = 0
    if n > 1:
        voxel_offsets[1:] = np.cumsum(voxel_counts[:-1], dtype=np.int64)

    z_all = np.concatenate([s.z for s in samples], axis=0).astype(np.uint8, copy=False)
    r_all = np.concatenate([s.r_ang for s in samples], axis=0).astype(np.float32, copy=False)
    d_all = np.concatenate([s.density_target_e_per_ang3 for s in samples], axis=0).astype(np.float32, copy=False)

    str_dt = h5py.string_dtype(encoding="utf-8")
    comp = None if compression == "none" else compression

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        g_f = f.create_group("frames")
        g_a = f.create_group("atoms")
        g_d = f.create_group("density")

        g_f.create_dataset("mol_id", data=np.asarray([s.mol_id for s in samples], dtype=object), dtype=str_dt)
        g_f.create_dataset("source_tar", data=np.asarray([s.source_tar for s in samples], dtype=object), dtype=str_dt)
        g_f.create_dataset(
            "source_prefix",
            data=np.asarray([s.source_prefix for s in samples], dtype=object),
            dtype=str_dt,
        )
        g_f.create_dataset("total_charge", data=np.asarray([s.total_charge for s in samples], dtype=np.int16))
        g_f.create_dataset(
            "spin_multiplicity",
            data=np.asarray([s.spin_multiplicity for s in samples], dtype=np.int16),
        )
        g_f.create_dataset("grid_shape", data=np.stack([s.grid_shape for s in samples], axis=0), dtype=np.int32)
        g_f.create_dataset("grid_origin", data=np.stack([s.grid_origin_ang for s in samples], axis=0), dtype=np.float32)
        g_f.create_dataset(
            "grid_vectors",
            data=np.stack([s.grid_vectors_ang for s in samples], axis=0),
            dtype=np.float32,
        )
        g_f.create_dataset("voxel_volume", data=np.asarray([s.voxel_volume_ang3 for s in samples], dtype=np.float32))
        g_f.create_dataset("atom_offsets", data=atom_offsets, dtype=np.int64)
        g_f.create_dataset("n_atoms", data=atom_counts, dtype=np.int32)
        g_f.create_dataset("density_offsets", data=voxel_offsets, dtype=np.int64)
        g_f.create_dataset("n_voxels", data=voxel_counts, dtype=np.int64)

        g_a.create_dataset("Z", data=z_all, dtype=np.uint8, compression=comp, shuffle=True)
        g_a.create_dataset("R", data=r_all, dtype=np.float32, compression=comp, shuffle=True)
        g_d.create_dataset("target", data=d_all, dtype=np.float32, compression=comp, shuffle=True)

        f.attrs["format"] = "ed_tar_density_hdf5.flattened.v1"
        f.attrs["coord_unit"] = "angstrom"
        f.attrs["density_unit"] = "e/angstrom^3"
        f.attrs["description"] = "Minimal fields for density pretraining from ED tar.gz (CHON filtered)."


def _flush_if_needed(
    pending: List[Sample],
    shard_id: int,
    args: argparse.Namespace,
    force: bool = False,
) -> Tuple[List[Sample], int]:
    if not pending:
        return pending, shard_id

    total_vox = int(sum(int(s.density_target_e_per_ang3.shape[0]) for s in pending))
    if not force:
        if len(pending) < args.samples_per_shard and total_vox < args.max_voxels_per_shard:
            return pending, shard_id

    out_path = Path(args.out_dir) / f"{args.prefix}_shard_{shard_id:06d}.h5"
    _write_shard(out_path=out_path, samples=pending, compression=args.compression)
    print(f"[write] {out_path}  samples={len(pending)}  voxels={total_vox}")
    return [], shard_id + 1


def main() -> None:
    args = parse_args()
    tar_paths = _expand_tar_paths(args.tar_glob)
    allowed_syms, allowed_z = _parse_allowed_elements(args.allowed_elements)
    print(f"[config] tar_files={len(tar_paths)} allowed_elements={sorted(allowed_syms)}")

    pending_samples: List[Sample] = []
    shard_id = 0

    n_seen = 0
    n_kept = 0
    n_skip_non_chon = 0
    n_skip_parse = 0

    pbar = tqdm(tar_paths, desc="Tar", unit="file")
    for tar_path in pbar:
        for prefix, cube_bytes, out_bytes in _iter_pairs_from_tar(tar_path):
            n_seen += 1
            try:
                z, r_ang, grid_shape, origin_ang, vectors_ang, dV_ang3, density = _parse_cube(
                    cube_bytes=cube_bytes,
                    coord_unit_mode=args.coord_unit_in_cube,
                    density_unit_mode=args.density_unit_in_cube,
                )
                if any(int(v) not in allowed_z for v in z.tolist()):
                    n_skip_non_chon += 1
                    continue
                charge, mult = _parse_charge_mult(out_bytes)
            except Exception:
                n_skip_parse += 1
                continue

            sample = Sample(
                mol_id=Path(prefix).name,
                source_tar=str(tar_path),
                source_prefix=prefix,
                total_charge=int(charge),
                spin_multiplicity=int(mult),
                z=z,
                r_ang=r_ang,
                grid_shape=grid_shape.astype(np.int32, copy=False),
                grid_origin_ang=origin_ang.astype(np.float32, copy=False),
                grid_vectors_ang=vectors_ang.astype(np.float32, copy=False),
                voxel_volume_ang3=float(dV_ang3),
                density_target_e_per_ang3=density.astype(np.float32, copy=False),
            )
            pending_samples.append(sample)
            n_kept += 1

            if args.verbose_every > 0 and (n_kept % args.verbose_every == 0):
                print(
                    f"[progress] seen={n_seen} kept={n_kept} "
                    f"skip_non_chon={n_skip_non_chon} skip_parse={n_skip_parse}"
                )

            pending_samples, shard_id = _flush_if_needed(pending_samples, shard_id, args, force=False)

            if args.limit > 0 and n_kept >= args.limit:
                break
        if args.limit > 0 and n_kept >= args.limit:
            break

    pending_samples, shard_id = _flush_if_needed(pending_samples, shard_id, args, force=True)

    print(
        "[done] "
        f"seen={n_seen} kept={n_kept} skip_non_chon={n_skip_non_chon} "
        f"skip_parse={n_skip_parse} shards={shard_id}"
    )


if __name__ == "__main__":
    main()
