#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert extxyz (GDB13 / GFN2-xTB style) -> sharded HDF5 with flattened atom arrays + frame offsets.

Stores exactly:
  /frames/offsets (int64)  length = n_frames+1, offsets[0]=0, offsets[i+1]=end atom index after frame i
  /frames/n_atoms (int32)
  /frames/energy (float64)
  /frames/dipole (float32, shape (n_frames,3))
  /atoms/Z (uint8)  H/C/N/O
  /atoms/R (float32, shape (n_atoms_total,3))
  /atoms/q (float32, canonical charge alias)
  /atoms/q_mulliken (float32)

Usage example:
  python extxyz_to_hdf5.py --input data.extxyz --out_dir out --prefix gdb13 \
      --frames_per_shard 5000000 --flush_frames 50000 --min_atoms 2 --max_atoms 64 \
      --min_dist 0.0 --compression lzf --chunk_frames 262144 --chunk_atoms_Zq 1048576 \
      --chunk_atoms_R 262144 --max_consecutive_parse_errors 1000000 --write_stats
"""

import argparse
import os
import re
import json
from typing import Dict, Tuple, Optional, List

import numpy as np
import h5py
from tqdm import tqdm


# CHON mapping
E2Z: Dict[str, int] = {"H": 1, "C": 6, "N": 7, "O": 8}

RE_ENERGY = re.compile(r"energy=([-\d.eE+]+)")
RE_DIPOLE = re.compile(r'dipole="([^"]+)"')
RE_PROPERTIES = re.compile(r"Properties=([^\s]+)")


class UnsupportedPropertiesError(ValueError):
    pass


def parse_properties_indices(comment: str) -> Tuple[int, int]:
    """
    Parse the extxyz Properties string and return:
      pos_offset: start index of pos in numeric columns (i.e. after species token)
      q_offset: start index of charge in numeric columns (after species token)
    Example Properties=species:S:1:pos:R:3:forces:R:3:charge:R:1
      numeric columns = pos(3) + forces(3) + charge(1) = 7
      pos_offset=0, q_offset=6
    """
    m = RE_PROPERTIES.search(comment)
    if not m:
        raise ValueError("Missing Properties=... in comment line")

    props_str = m.group(1)  # species:S:1:pos:R:3:...
    parts = props_str.split(":")
    if len(parts) % 3 != 0:
        raise ValueError(f"Malformed Properties field: {props_str}")

    # Require species to be first and a single string column.
    if parts[0] != "species" or parts[1] != "S" or parts[2] != "1":
        raise UnsupportedPropertiesError(
            f"species must be first and encoded as species:S:1, got {parts[:3]}"
        )

    pos_offset = None
    q_offset = None

    numeric_offset = 0  # counts only non-species columns
    for i in range(0, len(parts), 3):
        name = parts[i]
        ptype = parts[i + 1]
        count = int(parts[i + 2])

        if name == "species":
            # species is the first token in atom line, not included in numeric columns
            continue
        if ptype == "S":
            raise UnsupportedPropertiesError(
                f"Unsupported string property: {name}:{ptype}:{count}"
            )
        if ptype not in ("R", "I"):
            raise UnsupportedPropertiesError(
                f"Unsupported property type: {name}:{ptype}:{count}"
            )

        if name == "pos":
            if count != 3:
                raise ValueError(f"pos should have 3 columns, got {count}")
            pos_offset = numeric_offset

        if name == "charge":
            if count != 1:
                raise ValueError(f"charge should have 1 column, got {count}")
            q_offset = numeric_offset

        numeric_offset += count

    if pos_offset is None or q_offset is None:
        raise ValueError(f"Properties missing pos or charge: {props_str}")

    return pos_offset, q_offset


def parse_energy_and_dipole(comment: str) -> Tuple[float, np.ndarray]:
    mE = RE_ENERGY.search(comment)
    if not mE:
        raise ValueError("Missing energy=... in comment line")
    energy = float(mE.group(1))

    mD = RE_DIPOLE.search(comment)
    if not mD:
        raise ValueError('Missing dipole="..." in comment line')
    dip_parts = mD.group(1).strip().split()
    if len(dip_parts) != 3:
        raise ValueError(f"dipole has {len(dip_parts)} components, expected 3")
    dipole = np.array([float(x) for x in dip_parts], dtype=np.float32)
    return energy, dipole


def min_interatomic_distance_ok(R: np.ndarray, min_dist: float) -> bool:
    """
    Check min pairwise distance > min_dist (in same units as R, typically Å).
    R: (N,3)
    """
    N = R.shape[0]
    if N < 2:
        return False
    # O(N^2) but N is small for GDB13; use vectorized computation
    diff = R[:, None, :] - R[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    # ignore diagonal
    np.fill_diagonal(d2, np.inf)
    return float(np.min(d2)) > (min_dist * min_dist)


def create_shard(path: str,
                 compression: Optional[str],
                 chunk_frames: int,
                 chunk_atoms_Zq: int,
                 chunk_atoms_R: int) -> h5py.File:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = h5py.File(path, "w")

    gF = f.create_group("frames")
    gA = f.create_group("atoms")

    # Helper for dataset kwargs
    def ds_kwargs(chunks, shape_rank: int):
        kw = {"chunks": chunks}
        if compression and compression.lower() != "none":
            kw["compression"] = compression
        return kw

    # frames datasets
    # offsets: start with [0]
    gF.create_dataset(
        "offsets",
        data=np.array([0], dtype=np.int64),
        maxshape=(None,),
        dtype=np.int64,
        **ds_kwargs((chunk_frames,), 1),
    )
    gF.create_dataset(
        "n_atoms",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
        **ds_kwargs((chunk_frames,), 1),
    )
    gF.create_dataset(
        "energy",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64,
        **ds_kwargs((chunk_frames,), 1),
    )
    gF.create_dataset(
        "dipole",
        shape=(0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        **ds_kwargs((chunk_frames, 3), 2),
    )

    # atom datasets (flattened)
    gA.create_dataset(
        "Z",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint8,
        **ds_kwargs((chunk_atoms_Zq,), 1),
    )
    gA.create_dataset(
        "R",
        shape=(0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        **ds_kwargs((chunk_atoms_R, 3), 2),
    )
    gA.create_dataset(
        "q",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        **ds_kwargs((chunk_atoms_Zq,), 1),
    )
    gA.create_dataset(
        "q_mulliken",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        **ds_kwargs((chunk_atoms_Zq,), 1),
    )

    # Basic metadata (optional)
    f.attrs["format"] = "extxyz->hdf5.flattened.v1"
    f.attrs["units_R"] = "angstrom"
    f.attrs["units_dipole"] = "au"  # xTB dipole components in atomic units
    f.attrs["elements"] = "CHON"

    return f


def append_batch(f: h5py.File,
                 Z_cat: np.ndarray,
                 R_cat: np.ndarray,
                 q_cat: np.ndarray,
                 n_atoms_arr: np.ndarray,
                 energy_arr: np.ndarray,
                 dipole_arr: np.ndarray):
    """
    Append a batch into an open shard file.
    offsets convention: offsets[0]=0; offsets[i+1]=end atom index after frame i.
    """
    gF = f["frames"]
    gA = f["atoms"]

    # current sizes
    n_frames_cur = gF["n_atoms"].shape[0]
    n_atoms_cur = gA["Z"].shape[0]

    # Append atoms
    n_new_atoms = int(Z_cat.shape[0])
    new_atoms_total = n_atoms_cur + n_new_atoms

    gA["Z"].resize((new_atoms_total,))
    gA["R"].resize((new_atoms_total, 3))
    gA["q"].resize((new_atoms_total,))
    gA["q_mulliken"].resize((new_atoms_total,))

    gA["Z"][n_atoms_cur:new_atoms_total] = Z_cat
    gA["R"][n_atoms_cur:new_atoms_total, :] = R_cat
    gA["q"][n_atoms_cur:new_atoms_total] = q_cat
    gA["q_mulliken"][n_atoms_cur:new_atoms_total] = q_cat

    # Append frames
    n_new_frames = int(n_atoms_arr.shape[0])
    new_frames_total = n_frames_cur + n_new_frames

    gF["n_atoms"].resize((new_frames_total,))
    gF["energy"].resize((new_frames_total,))
    gF["dipole"].resize((new_frames_total, 3))

    gF["n_atoms"][n_frames_cur:new_frames_total] = n_atoms_arr
    gF["energy"][n_frames_cur:new_frames_total] = energy_arr
    gF["dipole"][n_frames_cur:new_frames_total, :] = dipole_arr

    # Append offsets end positions
    # existing offsets length = n_frames_cur+1; last value should be n_atoms_cur
    offsets = gF["offsets"]
    if offsets.shape[0] != n_frames_cur + 1:
        raise RuntimeError("offsets length mismatch with n_frames")

    last_end = int(offsets[-1])
    if last_end != n_atoms_cur:
        raise RuntimeError(f"offsets last_end={last_end} != n_atoms_cur={n_atoms_cur}")

    ends = last_end + np.cumsum(n_atoms_arr.astype(np.int64), dtype=np.int64)  # length n_new_frames
    offsets.resize((offsets.shape[0] + n_new_frames,))
    offsets[-n_new_frames:] = ends


def flush_buffers(f: h5py.File,
                  buf_Z: List[np.ndarray],
                  buf_R: List[np.ndarray],
                  buf_q: List[np.ndarray],
                  buf_n_atoms: List[int],
                  buf_energy: List[float],
                  buf_dipole: List[np.ndarray]):
    if not buf_n_atoms:
        return

    Z_cat = np.concatenate(buf_Z, axis=0).astype(np.uint8, copy=False)
    R_cat = np.concatenate(buf_R, axis=0).astype(np.float32, copy=False)
    q_cat = np.concatenate(buf_q, axis=0).astype(np.float32, copy=False)

    n_atoms_arr = np.asarray(buf_n_atoms, dtype=np.int32)
    energy_arr = np.asarray(buf_energy, dtype=np.float64)
    dipole_arr = np.stack(buf_dipole, axis=0).astype(np.float32, copy=False)

    append_batch(f, Z_cat, R_cat, q_cat, n_atoms_arr, energy_arr, dipole_arr)

    buf_Z.clear()
    buf_R.clear()
    buf_q.clear()
    buf_n_atoms.clear()
    buf_energy.clear()
    buf_dipole.clear()


def open_text_maybe_gz(path: str):
    # Keep it simple; gz support is optional.
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", newline="\n")
    return open(path, "rt", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Input extxyz file(s), optionally .gz")
    ap.add_argument("--out_dir", required=True, help="Output directory for shard HDF5 files")
    ap.add_argument("--prefix", default="gdb13", help="Output shard prefix")
    ap.add_argument("--frames_per_shard", type=int, default=5_000_000)
    ap.add_argument("--flush_frames", type=int, default=50_000, help="Flush buffers every N accepted frames")
    ap.add_argument("--min_atoms", type=int, default=2)
    ap.add_argument("--max_atoms", type=int, default=128)
    ap.add_argument(
        "--min_dist",
        type=float,
        default=0.0,
        help="Min interatomic distance threshold (Å). Default off; enabling is O(N^2) and can be slow.",
    )
    ap.add_argument("--compression", default="lzf", help="HDF5 compression: lzf|gzip|none")
    ap.add_argument("--chunk_frames", type=int, default=262_144)
    ap.add_argument("--chunk_atoms_Zq", type=int, default=1_048_576)
    ap.add_argument("--chunk_atoms_R", type=int, default=262_144)
    ap.add_argument(
        "--max_consecutive_parse_errors",
        type=int,
        default=1_000_000,
        help="Abort if natoms parse fails this many times in a row",
    )
    ap.add_argument("--write_stats", dest="write_stats", action="store_true", help="Write conversion stats JSON")
    ap.add_argument("--no_write_stats", dest="write_stats", action="store_false", help="Disable stats JSON")
    ap.set_defaults(write_stats=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shard_idx = 0
    frames_in_shard = 0

    # stats
    stats = {
        "total_frames_seen": 0,
        "total_frames_kept": 0,
        "filtered_bad_element": 0,
        "filtered_bad_numbers": 0,
        "filtered_unsupported_properties": 0,
        "filtered_n_range": 0,
        "filtered_min_dist": 0,
        "parse_errors": 0,
        "inputs": args.input,
        "config": {
            "frames_per_shard": args.frames_per_shard,
            "flush_frames": args.flush_frames,
            "min_atoms": args.min_atoms,
            "max_atoms": args.max_atoms,
            "min_dist": args.min_dist,
            "compression": args.compression,
            "chunk_frames": args.chunk_frames,
            "chunk_atoms_Zq": args.chunk_atoms_Zq,
            "chunk_atoms_R": args.chunk_atoms_R,
            "max_consecutive_parse_errors": args.max_consecutive_parse_errors,
        },
    }

    def shard_path(i: int) -> str:
        return os.path.join(args.out_dir, f"{args.prefix}_shard_{i:06d}.h5")

    f = create_shard(
        shard_path(shard_idx),
        compression=args.compression,
        chunk_frames=args.chunk_frames,
        chunk_atoms_Zq=args.chunk_atoms_Zq,
        chunk_atoms_R=args.chunk_atoms_R,
    )

    # buffers
    buf_Z: List[np.ndarray] = []
    buf_R: List[np.ndarray] = []
    buf_q: List[np.ndarray] = []
    buf_n_atoms: List[int] = []
    buf_energy: List[float] = []
    buf_dipole: List[np.ndarray] = []

    # cache for properties parsing
    last_props_token = None
    cached_pos_off = None
    cached_q_off = None
    pbar = tqdm(
        total=None,
        unit="frames",
        mininterval=1.0,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt} {unit} [{elapsed}, {rate_fmt}] {postfix}",
    )
    update_every = 1000
    pending_updates = 0
    consecutive_bad_natoms = 0

    for in_path in args.input:
        with open_text_maybe_gz(in_path) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                line = line.strip()
                if line == "":
                    continue

                stats["total_frames_seen"] += 1
                pending_updates += 1
                if pending_updates >= update_every:
                    pbar.update(pending_updates)
                    pbar.set_postfix(
                        kept=stats["total_frames_kept"],
                        shard=shard_idx,
                        in_shard=frames_in_shard,
                    )
                    pending_updates = 0

                # read natoms
                try:
                    natoms = int(line)
                    consecutive_bad_natoms = 0
                except Exception:
                    stats["parse_errors"] += 1
                    consecutive_bad_natoms += 1
                    if consecutive_bad_natoms >= args.max_consecutive_parse_errors:
                        raise RuntimeError(
                            f"Aborting: {consecutive_bad_natoms} consecutive natoms parse failures. "
                            f"Last bad line: {line!r}"
                        )
                    continue

                # read comment line
                comment = fin.readline()
                if not comment:
                    break
                comment = comment.strip()

                try:
                    energy, dipole = parse_energy_and_dipole(comment)

                    # properties parse (cached)
                    m = RE_PROPERTIES.search(comment)
                    if not m:
                        raise ValueError("Missing Properties")
                    props_token = m.group(1)
                    if props_token != last_props_token:
                        pos_off, q_off = parse_properties_indices(comment)
                        last_props_token = props_token
                        cached_pos_off = pos_off
                        cached_q_off = q_off
                    else:
                        pos_off = cached_pos_off
                        q_off = cached_q_off

                except UnsupportedPropertiesError:
                    # skip the natoms lines to resync
                    stats["filtered_unsupported_properties"] += 1
                    for _ in range(natoms):
                        if not fin.readline():
                            break
                    continue
                except Exception:
                    # skip the natoms lines to resync
                    stats["parse_errors"] += 1
                    for _ in range(natoms):
                        if not fin.readline():
                            break
                    continue

                # quick N filter
                if natoms < args.min_atoms or natoms > args.max_atoms:
                    stats["filtered_n_range"] += 1
                    for _ in range(natoms):
                        fin.readline()
                    continue

                Z = np.empty((natoms,), dtype=np.uint8)
                R = np.empty((natoms, 3), dtype=np.float32)
                q = np.empty((natoms,), dtype=np.float32)

                bad = False
                bad_element = False
                i = -1
                for i in range(natoms):
                    atom_line = fin.readline()
                    if not atom_line:
                        bad = True
                        break
                    atom_line = atom_line.strip()
                    if atom_line == "":
                        bad = True
                        break

                    # Split once: species + numeric tail
                    parts = atom_line.split(maxsplit=1)
                    if len(parts) != 2:
                        bad = True
                        break
                    sym = parts[0]
                    z = E2Z.get(sym, 0)
                    if z == 0:
                        bad = True
                        bad_element = True
                        stats["filtered_bad_element"] += 1
                        break

                    nums = np.fromstring(parts[1], sep=" ")
                    # Need at least up to pos_off+3 and q_off+1
                    if nums.size < max(pos_off + 3, q_off + 1):
                        bad = True
                        break

                    coord = nums[pos_off:pos_off + 3]
                    qi = nums[q_off]

                    if not np.isfinite(coord).all() or not np.isfinite(qi):
                        bad = True
                        break

                    Z[i] = z
                    R[i, :] = coord.astype(np.float32, copy=False)
                    q[i] = np.float32(qi)

                if bad:
                    # consume remaining atom lines to keep stream aligned
                    remaining = natoms - (i + 1)
                    for _ in range(max(0, remaining)):
                        if not fin.readline():
                            break
                    if not bad_element:
                        stats["filtered_bad_numbers"] += 1
                    continue

                # min distance filter
                if args.min_dist > 0.0 and (not min_interatomic_distance_ok(R, args.min_dist)):
                    stats["filtered_min_dist"] += 1
                    continue

                # accept frame: buffer it
                buf_Z.append(Z)
                buf_R.append(R)
                buf_q.append(q)
                buf_n_atoms.append(int(natoms))
                buf_energy.append(float(energy))
                buf_dipole.append(dipole)

                stats["total_frames_kept"] += 1
                frames_in_shard += 1

                # flush periodically
                if len(buf_n_atoms) >= args.flush_frames:
                    flush_buffers(f, buf_Z, buf_R, buf_q, buf_n_atoms, buf_energy, buf_dipole)
                    f.flush()

                # rotate shard if needed
                if frames_in_shard >= args.frames_per_shard:
                    # flush remaining buffers first
                    flush_buffers(f, buf_Z, buf_R, buf_q, buf_n_atoms, buf_energy, buf_dipole)
                    f.flush()
                    f.close()

                    shard_idx += 1
                    frames_in_shard = 0
                    f = create_shard(
                        shard_path(shard_idx),
                        compression=args.compression,
                        chunk_frames=args.chunk_frames,
                        chunk_atoms_Zq=args.chunk_atoms_Zq,
                        chunk_atoms_R=args.chunk_atoms_R,
                    )

    # final flush/close
    flush_buffers(f, buf_Z, buf_R, buf_q, buf_n_atoms, buf_energy, buf_dipole)
    f.flush()
    f.close()
    if pending_updates:
        pbar.update(pending_updates)
    pbar.set_postfix(
        kept=stats["total_frames_kept"],
        shard=shard_idx,
        in_shard=frames_in_shard,
    )
    pbar.close()

    if args.write_stats:
        stats_path = os.path.join(args.out_dir, f"{args.prefix}_convert_stats.json")
        with open(stats_path, "w", encoding="utf-8") as w:
            json.dump(stats, w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
