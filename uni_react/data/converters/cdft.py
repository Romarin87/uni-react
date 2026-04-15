#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert paired XYZ + CDFT text files into GDB13-aligned sharded HDF5.

Expected pair:
  foo.xyz
  foo.xyz.CDFT.txt

Output schema (flattened atom arrays + frame offsets):
  /frames/offsets (int64)
  /frames/n_atoms (int32)
  /frames/energy (float64, canonical unit)
  /atoms/Z (uint8)
  /atoms/R (float32, shape (n_atoms_total, 3))
  /atoms/q (float32, canonical charge alias)
  /atoms/q_hirshfeld (float32, condensed charge q(N))

Plus many CDFT labels in /frames/* and /atoms/*.
Canonical unit defaults to eV while alternative units are also preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_EV2 = HARTREE_TO_EV * HARTREE_TO_EV
FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

# Atom-row format in CDFT tables, e.g.:
#   1(C )   0.0594   0.0470 ...
ATOM_ROW_RE = re.compile(r"^\s*(\d+)\(\s*([A-Za-z]{1,3})\s*\)\s+(.+)$")
FLOAT_TOKEN_RE = re.compile(FLOAT_RE)


# Periodic table symbols (1..118)
_PT = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
SYMBOL_TO_Z = {sym: z for z, sym in enumerate(_PT) if sym}


FRAME_SCALAR_KEYS = (
    "energy", "energy_hartree", "energy_ev",
    "vip", "vip_hartree", "vip_ev",
    "vea", "vea_hartree", "vea_ev",
    "vip2_hartree", "vip2_ev",
    "mulliken_electronegativity", "mulliken_electronegativity_hartree", "mulliken_electronegativity_ev",
    "chemical_potential", "chemical_potential_hartree", "chemical_potential_ev",
    "hardness", "hardness_hartree", "hardness_ev",
    "softness", "softness_hartree_inv", "softness_ev_inv",
    "softness2", "softness2_hartree_inv2", "softness2_ev_inv2",
    "electrophilicity_index", "electrophilicity_index_hartree", "electrophilicity_index_ev",
    "nucleophilicity_index", "nucleophilicity_index_hartree", "nucleophilicity_index_ev",
    "cubic_electrophilicity_index", "cubic_electrophilicity_index_hartree", "cubic_electrophilicity_index_ev",
    "electrophilic_descriptor", "electrophilic_descriptor_hartree", "electrophilic_descriptor_ev",
    "e_n_hartree", "e_n_ev",
    "e_nplus1_hartree", "e_nplus1_ev",
    "e_nminus1_hartree", "e_nminus1_ev",
    "e_nminus2_hartree", "e_nminus2_ev",
    "homo_n_hartree", "homo_n_ev",
    "homo_nplus1_hartree", "homo_nplus1_ev",
    "homo_nminus1_hartree", "homo_nminus1_ev",
    "homo_nminus2_hartree", "homo_nminus2_ev",
)

ATOM_SCALAR_KEYS = (
    "q", "q_hirshfeld", "q_n", "q_nplus1", "q_nminus1",
    "f_plus", "f_minus", "f_zero", "cdd",
    "local_electrophilicity", "local_electrophilicity_hartree", "local_electrophilicity_ev",
    "local_nucleophilicity", "local_nucleophilicity_hartree", "local_nucleophilicity_ev",
    "local_cubic_electrophilicity", "local_cubic_electrophilicity_hartree", "local_cubic_electrophilicity_ev",
    "s_plus", "s_minus", "s_zero", "s2",
    "s_plus_hartree_inv", "s_plus_ev_inv",
    "s_minus_hartree_inv", "s_minus_ev_inv",
    "s_zero_hartree_inv", "s_zero_ev_inv",
    "s2_hartree_inv2", "s2_ev_inv2",
    "s_plus_over_s_minus", "s_minus_over_s_plus",
    "rel_electrophilicity", "rel_nucleophilicity",
)

MANDATORY_FRAME_KEYS = ("vip_ev", "vea_ev")
MANDATORY_ATOM_KEYS = ("q_n", "f_plus", "f_minus", "f_zero")


def _canonical_symbol(sym: str) -> str:
    sym = sym.strip()
    if not sym:
        raise ValueError("Empty element symbol")
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


def _unit_maps(canonical_energy_unit: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if canonical_energy_unit not in ("ev", "hartree"):
        raise ValueError(f"Unsupported canonical_energy_unit: {canonical_energy_unit}")
    inv = "eV^-1" if canonical_energy_unit == "ev" else "Hartree^-1"
    inv2 = "eV^-2" if canonical_energy_unit == "ev" else "Hartree^-2"
    eunit = "eV" if canonical_energy_unit == "ev" else "Hartree"

    frame_units = {
        "energy": eunit,
        "energy_hartree": "Hartree",
        "energy_ev": "eV",
        "vip": eunit,
        "vip_hartree": "Hartree",
        "vip_ev": "eV",
        "vea": eunit,
        "vea_hartree": "Hartree",
        "vea_ev": "eV",
        "vip2_hartree": "Hartree",
        "vip2_ev": "eV",
        "mulliken_electronegativity": eunit,
        "mulliken_electronegativity_hartree": "Hartree",
        "mulliken_electronegativity_ev": "eV",
        "chemical_potential": eunit,
        "chemical_potential_hartree": "Hartree",
        "chemical_potential_ev": "eV",
        "hardness": eunit,
        "hardness_hartree": "Hartree",
        "hardness_ev": "eV",
        "softness": inv,
        "softness_hartree_inv": "Hartree^-1",
        "softness_ev_inv": "eV^-1",
        "softness2": inv2,
        "softness2_hartree_inv2": "Hartree^-2",
        "softness2_ev_inv2": "eV^-2",
        "electrophilicity_index": eunit,
        "electrophilicity_index_hartree": "Hartree",
        "electrophilicity_index_ev": "eV",
        "nucleophilicity_index": eunit,
        "nucleophilicity_index_hartree": "Hartree",
        "nucleophilicity_index_ev": "eV",
        "cubic_electrophilicity_index": eunit,
        "cubic_electrophilicity_index_hartree": "Hartree",
        "cubic_electrophilicity_index_ev": "eV",
        "electrophilic_descriptor": eunit,
        "electrophilic_descriptor_hartree": "Hartree",
        "electrophilic_descriptor_ev": "eV",
        "e_n_hartree": "Hartree",
        "e_n_ev": "eV",
        "e_nplus1_hartree": "Hartree",
        "e_nplus1_ev": "eV",
        "e_nminus1_hartree": "Hartree",
        "e_nminus1_ev": "eV",
        "e_nminus2_hartree": "Hartree",
        "e_nminus2_ev": "eV",
        "homo_n_hartree": "Hartree",
        "homo_n_ev": "eV",
        "homo_nplus1_hartree": "Hartree",
        "homo_nplus1_ev": "eV",
        "homo_nminus1_hartree": "Hartree",
        "homo_nminus1_ev": "eV",
        "homo_nminus2_hartree": "Hartree",
        "homo_nminus2_ev": "eV",
    }

    atom_units = {
        "q": "e",
        "q_hirshfeld": "e",
        "q_n": "e",
        "q_nplus1": "e",
        "q_nminus1": "e",
        "f_plus": "e",
        "f_minus": "e",
        "f_zero": "e",
        "cdd": "e",
        "local_electrophilicity": f"e*{eunit}",
        "local_electrophilicity_hartree": "e*Hartree",
        "local_electrophilicity_ev": "e*eV",
        "local_nucleophilicity": f"e*{eunit}",
        "local_nucleophilicity_hartree": "e*Hartree",
        "local_nucleophilicity_ev": "e*eV",
        "local_cubic_electrophilicity": f"e*{eunit}",
        "local_cubic_electrophilicity_hartree": "e*Hartree",
        "local_cubic_electrophilicity_ev": "e*eV",
        "s_plus": inv,
        "s_minus": inv,
        "s_zero": inv,
        "s2": inv2,
        "s_plus_hartree_inv": "Hartree^-1",
        "s_plus_ev_inv": "eV^-1",
        "s_minus_hartree_inv": "Hartree^-1",
        "s_minus_ev_inv": "eV^-1",
        "s_zero_hartree_inv": "Hartree^-1",
        "s_zero_ev_inv": "eV^-1",
        "s2_hartree_inv2": "Hartree^-2",
        "s2_ev_inv2": "eV^-2",
        "s_plus_over_s_minus": "dimensionless",
        "s_minus_over_s_plus": "dimensionless",
        "rel_electrophilicity": "dimensionless",
        "rel_nucleophilicity": "dimensionless",
    }
    return frame_units, atom_units


def _set_hartree_ev(d: Dict[str, float], base: str, hartree: float, ev: float) -> None:
    d[f"{base}_hartree"] = float(hartree)
    d[f"{base}_ev"] = float(ev)


def _set_hartree_only(d: Dict[str, float], base: str, hartree: float) -> None:
    _set_hartree_ev(d, base, hartree=hartree, ev=hartree * HARTREE_TO_EV)


def _read_xyz(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"Empty xyz file: {path}")
        try:
            natoms = int(first.strip())
        except Exception as exc:
            raise ValueError(f"Invalid natoms line in {path}: {first!r}") from exc

        # Comment/title line.
        comment = f.readline()
        if comment == "":
            raise ValueError(f"Missing xyz comment line: {path}")

        symbols: List[str] = []
        z = np.empty((natoms,), dtype=np.uint8)
        r = np.empty((natoms, 3), dtype=np.float32)

        for i in range(natoms):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in xyz atom block: {path}")
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Bad xyz atom line ({path}:{i+3}): {line!r}")

            sym = _canonical_symbol(parts[0])
            zi = SYMBOL_TO_Z.get(sym)
            if zi is None or zi <= 0 or zi > 255:
                raise ValueError(f"Unsupported element symbol in {path}: {sym!r}")
            try:
                x, y, zc = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception as exc:
                raise ValueError(f"Non-numeric xyz coordinate in {path}: {line!r}") from exc

            symbols.append(sym)
            z[i] = np.uint8(zi)
            r[i, :] = (x, y, zc)

    return symbols, z, r


def _parse_atom_row(line: str) -> Optional[Tuple[int, str, List[float]]]:
    m = ATOM_ROW_RE.match(line)
    if not m:
        return None
    atom_idx = int(m.group(1)) - 1
    atom_sym = _canonical_symbol(m.group(2))
    nums = [float(x) for x in FLOAT_TOKEN_RE.findall(m.group(3))]
    return atom_idx, atom_sym, nums


def _parse_cdft(
    path: Path,
    natoms: int,
    xyz_symbols: Sequence[str],
    canonical_energy_unit: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    frame: Dict[str, float] = {}
    atom_raw: Dict[str, np.ndarray] = {
        "q_n": np.full((natoms,), np.nan, dtype=np.float32),
        "q_nplus1": np.full((natoms,), np.nan, dtype=np.float32),
        "q_nminus1": np.full((natoms,), np.nan, dtype=np.float32),
        "f_minus": np.full((natoms,), np.nan, dtype=np.float32),
        "f_plus": np.full((natoms,), np.nan, dtype=np.float32),
        "f_zero": np.full((natoms,), np.nan, dtype=np.float32),
        "cdd": np.full((natoms,), np.nan, dtype=np.float32),
        "local_electrophilicity_ev": np.full((natoms,), np.nan, dtype=np.float32),
        "local_nucleophilicity_ev": np.full((natoms,), np.nan, dtype=np.float32),
        "local_cubic_electrophilicity_ev": np.full((natoms,), np.nan, dtype=np.float32),
        "s_minus_hartree_inv": np.full((natoms,), np.nan, dtype=np.float32),
        "s_plus_hartree_inv": np.full((natoms,), np.nan, dtype=np.float32),
        "s_zero_hartree_inv": np.full((natoms,), np.nan, dtype=np.float32),
        "s_plus_over_s_minus": np.full((natoms,), np.nan, dtype=np.float32),
        "s_minus_over_s_plus": np.full((natoms,), np.nan, dtype=np.float32),
        "s2_hartree_inv2": np.full((natoms,), np.nan, dtype=np.float32),
    }

    # Patterns for global scalar lines.
    re_map_h2e = {
        "vip": re.compile(rf"^First vertical IP:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
        "vip2": re.compile(rf"^Second vertical IP:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
        "vea": re.compile(rf"^First vertical EA:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
        "mulliken_electronegativity": re.compile(
            rf"^Mulliken electronegativity:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
        "chemical_potential": re.compile(
            rf"^Chemical potential:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
        "hardness": re.compile(
            rf"^Hardness \(=fundamental gap\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
        "electrophilicity_index": re.compile(
            rf"^Electrophilicity index:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
        "nucleophilicity_index": re.compile(
            rf"^Nucleophilicity index:\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
        "cubic_electrophilicity_index": re.compile(
            rf"^Cubic electrophilicity index \(w_cubic\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
        "electrophilic_descriptor": re.compile(
            rf"^Electrophilic descriptor \(epsilon\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"
        ),
    }
    re_softness = re.compile(
        rf"^Softness:\s*({FLOAT_RE})\s*Hartree\^-1,\s*({FLOAT_RE})\s*eV\^-1$"
    )
    re_softness2 = re.compile(
        rf"^Softness\^2:\s*({FLOAT_RE})\s*Hartree\^-2,\s*({FLOAT_RE})\s*eV\^-2$"
    )
    re_energy_n = {
        "e_n": re.compile(rf"^E\(N\):\s*({FLOAT_RE})\s*Hartree$"),
        "e_nplus1": re.compile(rf"^E\(N\+1\):\s*({FLOAT_RE})\s*Hartree$"),
        "e_nminus1": re.compile(rf"^E\(N-1\):\s*({FLOAT_RE})\s*Hartree$"),
        "e_nminus2": re.compile(rf"^E\(N-2\):\s*({FLOAT_RE})\s*Hartree$"),
    }
    re_homo_n = {
        "homo_n": re.compile(rf"^E_HOMO\(N\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
        "homo_nplus1": re.compile(rf"^E_HOMO\(N\+1\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
        "homo_nminus1": re.compile(rf"^E_HOMO\(N-1\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
        "homo_nminus2": re.compile(rf"^E_HOMO\(N-2\):\s*({FLOAT_RE})\s*Hartree,\s*({FLOAT_RE})\s*eV$"),
    }

    mode: Optional[str] = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Global scalar lines.
            matched = False
            for base, rx in re_map_h2e.items():
                m = rx.match(line)
                if m:
                    _set_hartree_ev(frame, base, float(m.group(1)), float(m.group(2)))
                    matched = True
                    break
            if matched:
                continue

            m = re_softness.match(line)
            if m:
                frame["softness_hartree_inv"] = float(m.group(1))
                frame["softness_ev_inv"] = float(m.group(2))
                continue
            m = re_softness2.match(line)
            if m:
                frame["softness2_hartree_inv2"] = float(m.group(1))
                frame["softness2_ev_inv2"] = float(m.group(2))
                continue

            found_energy = False
            for base, rx in re_energy_n.items():
                m = rx.match(line)
                if m:
                    _set_hartree_only(frame, base, float(m.group(1)))
                    found_energy = True
                    break
            if found_energy:
                continue

            found_homo = False
            for base, rx in re_homo_n.items():
                m = rx.match(line)
                if m:
                    _set_hartree_ev(frame, base, float(m.group(1)), float(m.group(2)))
                    found_homo = True
                    break
            if found_homo:
                continue

            # Section headers.
            if "q(N)" in line and "q(N+1)" in line and "f-" in line and "f+" in line and "CDD" in line:
                mode = "hirshfeld"
                continue
            if line.startswith("Condensed local electrophilicity/nucleophilicity index"):
                mode = "local_en"
                continue
            if line.startswith("Condensed local cubic electrophilicity index"):
                mode = "local_cubic"
                continue
            if line.startswith("Condensed local softness"):
                mode = "local_softness"
                continue
            if line.startswith("Atom") or line.startswith("Value"):
                continue

            if mode is None:
                continue

            row = _parse_atom_row(line)
            if row is None:
                mode = None
                continue

            atom_idx, atom_sym, nums = row
            if atom_idx < 0 or atom_idx >= natoms:
                raise ValueError(f"Atom index out of range in {path}: {atom_idx + 1}/{natoms}")
            if atom_sym != xyz_symbols[atom_idx]:
                raise ValueError(
                    f"Atom symbol mismatch in {path}: atom {atom_idx + 1}, "
                    f"xyz={xyz_symbols[atom_idx]}, cdft={atom_sym}"
                )

            if mode == "hirshfeld":
                if len(nums) < 7:
                    raise ValueError(f"Bad Hirshfeld row in {path}: {line!r}")
                atom_raw["q_n"][atom_idx] = nums[0]
                atom_raw["q_nplus1"][atom_idx] = nums[1]
                atom_raw["q_nminus1"][atom_idx] = nums[2]
                atom_raw["f_minus"][atom_idx] = nums[3]
                atom_raw["f_plus"][atom_idx] = nums[4]
                atom_raw["f_zero"][atom_idx] = nums[5]
                atom_raw["cdd"][atom_idx] = nums[6]
            elif mode == "local_en":
                if len(nums) < 2:
                    raise ValueError(f"Bad local electrophilicity/nucleophilicity row in {path}: {line!r}")
                atom_raw["local_electrophilicity_ev"][atom_idx] = nums[0]
                atom_raw["local_nucleophilicity_ev"][atom_idx] = nums[1]
            elif mode == "local_cubic":
                if len(nums) < 1:
                    raise ValueError(f"Bad local cubic electrophilicity row in {path}: {line!r}")
                atom_raw["local_cubic_electrophilicity_ev"][atom_idx] = nums[0]
            elif mode == "local_softness":
                if len(nums) < 6:
                    raise ValueError(f"Bad local softness row in {path}: {line!r}")
                atom_raw["s_minus_hartree_inv"][atom_idx] = nums[0]
                atom_raw["s_plus_hartree_inv"][atom_idx] = nums[1]
                atom_raw["s_zero_hartree_inv"][atom_idx] = nums[2]
                atom_raw["s_plus_over_s_minus"][atom_idx] = nums[3]
                atom_raw["s_minus_over_s_plus"][atom_idx] = nums[4]
                atom_raw["s2_hartree_inv2"][atom_idx] = nums[5]

    # Canonical energy from E(N)
    if "e_n_hartree" in frame:
        frame["energy_hartree"] = frame["e_n_hartree"]
        frame["energy_ev"] = frame["e_n_ev"]

    # Canonical aliases.
    ce = canonical_energy_unit
    frame["energy"] = frame.get(f"energy_{ce}", np.nan)
    frame["vip"] = frame.get(f"vip_{ce}", np.nan)
    frame["vea"] = frame.get(f"vea_{ce}", np.nan)
    frame["mulliken_electronegativity"] = frame.get(f"mulliken_electronegativity_{ce}", np.nan)
    frame["chemical_potential"] = frame.get(f"chemical_potential_{ce}", np.nan)
    frame["hardness"] = frame.get(f"hardness_{ce}", np.nan)
    frame["electrophilicity_index"] = frame.get(f"electrophilicity_index_{ce}", np.nan)
    frame["nucleophilicity_index"] = frame.get(f"nucleophilicity_index_{ce}", np.nan)
    frame["cubic_electrophilicity_index"] = frame.get(f"cubic_electrophilicity_index_{ce}", np.nan)
    frame["electrophilic_descriptor"] = frame.get(f"electrophilic_descriptor_{ce}", np.nan)
    frame["softness"] = frame.get("softness_ev_inv" if ce == "ev" else "softness_hartree_inv", np.nan)
    frame["softness2"] = frame.get("softness2_ev_inv2" if ce == "ev" else "softness2_hartree_inv2", np.nan)

    # Atom-side unit conversions and canonical aliases.
    atom = {k: np.full((natoms,), np.nan, dtype=np.float32) for k in ATOM_SCALAR_KEYS}
    for key, arr in atom_raw.items():
        atom[key] = arr.astype(np.float32, copy=False)
    atom["q_hirshfeld"] = atom["q_n"].copy()
    atom["q"] = atom["q_hirshfeld"].copy()

    if np.isfinite(atom["local_electrophilicity_ev"]).any():
        atom["local_electrophilicity_hartree"] = atom["local_electrophilicity_ev"] / HARTREE_TO_EV
    if np.isfinite(atom["local_nucleophilicity_ev"]).any():
        atom["local_nucleophilicity_hartree"] = atom["local_nucleophilicity_ev"] / HARTREE_TO_EV
    if np.isfinite(atom["local_cubic_electrophilicity_ev"]).any():
        atom["local_cubic_electrophilicity_hartree"] = atom["local_cubic_electrophilicity_ev"] / HARTREE_TO_EV

    if np.isfinite(atom["s_plus_hartree_inv"]).any():
        atom["s_plus_ev_inv"] = atom["s_plus_hartree_inv"] / HARTREE_TO_EV
    if np.isfinite(atom["s_minus_hartree_inv"]).any():
        atom["s_minus_ev_inv"] = atom["s_minus_hartree_inv"] / HARTREE_TO_EV
    if np.isfinite(atom["s_zero_hartree_inv"]).any():
        atom["s_zero_ev_inv"] = atom["s_zero_hartree_inv"] / HARTREE_TO_EV
    if np.isfinite(atom["s2_hartree_inv2"]).any():
        atom["s2_ev_inv2"] = atom["s2_hartree_inv2"] / HARTREE_TO_EV2

    atom["local_electrophilicity"] = atom["local_electrophilicity_ev" if ce == "ev" else "local_electrophilicity_hartree"]
    atom["local_nucleophilicity"] = atom["local_nucleophilicity_ev" if ce == "ev" else "local_nucleophilicity_hartree"]
    atom["local_cubic_electrophilicity"] = atom[
        "local_cubic_electrophilicity_ev" if ce == "ev" else "local_cubic_electrophilicity_hartree"
    ]
    atom["s_plus"] = atom["s_plus_ev_inv" if ce == "ev" else "s_plus_hartree_inv"]
    atom["s_minus"] = atom["s_minus_ev_inv" if ce == "ev" else "s_minus_hartree_inv"]
    atom["s_zero"] = atom["s_zero_ev_inv" if ce == "ev" else "s_zero_hartree_inv"]
    atom["s2"] = atom["s2_ev_inv2" if ce == "ev" else "s2_hartree_inv2"]
    atom["rel_electrophilicity"] = atom["s_plus_over_s_minus"].copy()
    atom["rel_nucleophilicity"] = atom["s_minus_over_s_plus"].copy()

    frame_out = {k: np.nan for k in FRAME_SCALAR_KEYS}
    frame_out.update({k: float(v) for k, v in frame.items() if k in frame_out})

    # Required fields for stage-2 pretraining.
    for key in MANDATORY_FRAME_KEYS:
        if not np.isfinite(frame_out.get(key, np.nan)):
            raise ValueError(f"Missing mandatory global CDFT field: {key} ({path})")
    for key in MANDATORY_ATOM_KEYS:
        if np.isnan(atom[key]).any():
            raise ValueError(f"Missing mandatory atom CDFT field: {key} ({path})")

    return frame_out, atom


def _discover_xyz_files(inputs: Sequence[str], recursive: bool) -> List[Path]:
    out: List[Path] = []
    for p in inputs:
        path = Path(p)
        if path.is_file():
            if path.name.endswith(".xyz"):
                out.append(path.resolve())
            continue
        if path.is_dir():
            it = path.rglob("*.xyz") if recursive else path.glob("*.xyz")
            out.extend(x.resolve() for x in it)
            continue
        # wildcard support
        if any(ch in str(path) for ch in "*?[]"):
            parent = path.parent if str(path.parent) else Path(".")
            it = parent.rglob(path.name) if recursive else parent.glob(path.name)
            out.extend(x.resolve() for x in it if x.name.endswith(".xyz"))
    out = sorted(set(out))
    return out


def _geometry_hash(z: np.ndarray, r: np.ndarray, decimals: int) -> bytes:
    scale = float(10 ** int(decimals))
    q = np.rint(r.astype(np.float64) * scale).astype(np.int64)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(z, dtype=np.uint8).tobytes())
    h.update(q.tobytes())
    return h.digest()


def _sample_signature(frame: Dict[str, float], atom: Dict[str, np.ndarray]) -> bytes:
    # Lightweight but stable signature for duplicate-conflict check.
    keys_f = ("vip_ev", "vea_ev", "hardness_ev", "electrophilicity_index_ev", "nucleophilicity_index_ev")
    keys_a = ("q_n", "f_plus", "f_minus", "f_zero")

    h = hashlib.blake2b(digest_size=16)
    fv = np.asarray([frame.get(k, np.nan) for k in keys_f], dtype=np.float64)
    h.update(fv.tobytes())
    for k in keys_a:
        arr = np.asarray(atom[k], dtype=np.float32)
        h.update(arr.tobytes())
    return h.digest()


def _ds_kwargs(chunks: Tuple[int, ...], compression: Optional[str]) -> Dict:
    kw = {"chunks": chunks}
    if compression and compression.lower() != "none":
        kw["compression"] = compression
    return kw


def _create_shard(
    path: Path,
    compression: Optional[str],
    chunk_frames: int,
    chunk_atoms_scalar: int,
    chunk_atoms_coords: int,
    canonical_energy_unit: str,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    g_f = f.create_group("frames")
    g_a = f.create_group("atoms")

    frame_units, atom_units = _unit_maps(canonical_energy_unit)

    g_f.create_dataset(
        "offsets",
        data=np.asarray([0], dtype=np.int64),
        maxshape=(None,),
        dtype=np.int64,
        **_ds_kwargs((chunk_frames,), compression),
    )
    g_f.create_dataset(
        "n_atoms",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
        **_ds_kwargs((chunk_frames,), compression),
    )
    for key in FRAME_SCALAR_KEYS:
        ds = g_f.create_dataset(
            key,
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            **_ds_kwargs((chunk_frames,), compression),
        )
        unit = frame_units.get(key)
        if unit:
            ds.attrs["units"] = unit

    g_a.create_dataset(
        "Z",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint8,
        **_ds_kwargs((chunk_atoms_scalar,), compression),
    )
    g_a.create_dataset(
        "R",
        shape=(0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        **_ds_kwargs((chunk_atoms_coords, 3), compression),
    )
    for key in ATOM_SCALAR_KEYS:
        ds = g_a.create_dataset(
            key,
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            **_ds_kwargs((chunk_atoms_scalar,), compression),
        )
        unit = atom_units.get(key)
        if unit:
            ds.attrs["units"] = unit

    f.attrs["format"] = "cdft_xyz->hdf5.flattened.v1"
    f.attrs["canonical_energy_unit"] = canonical_energy_unit
    f.attrs["base_layout"] = "gdb13_compatible_frames_atoms_offsets"
    f.attrs["units_R"] = "angstrom"
    f.attrs["dedup"] = "geometry_hash(symbols+coords_quantized)"
    return f


def _append_batch(
    f: h5py.File,
    samples: Sequence[Dict],
) -> None:
    if not samples:
        return

    g_f = f["frames"]
    g_a = f["atoms"]

    n_frames_cur = int(g_f["n_atoms"].shape[0])
    n_atoms_cur = int(g_a["Z"].shape[0])
    n_new_frames = len(samples)

    n_atoms_arr = np.asarray([s["num_atoms"] for s in samples], dtype=np.int32)
    n_new_atoms = int(n_atoms_arr.sum())
    new_frames_total = n_frames_cur + n_new_frames
    new_atoms_total = n_atoms_cur + n_new_atoms

    # Concatenate batch payload.
    z_cat = np.concatenate([s["Z"] for s in samples], axis=0).astype(np.uint8, copy=False)
    r_cat = np.concatenate([s["R"] for s in samples], axis=0).astype(np.float32, copy=False)
    atom_cat: Dict[str, np.ndarray] = {}
    for key in ATOM_SCALAR_KEYS:
        atom_cat[key] = np.concatenate([s["atom"][key] for s in samples], axis=0).astype(np.float32, copy=False)
    frame_arr: Dict[str, np.ndarray] = {}
    for key in FRAME_SCALAR_KEYS:
        frame_arr[key] = np.asarray([s["frame"][key] for s in samples], dtype=np.float64)

    # Resize.
    g_a["Z"].resize((new_atoms_total,))
    g_a["R"].resize((new_atoms_total, 3))
    for key in ATOM_SCALAR_KEYS:
        g_a[key].resize((new_atoms_total,))

    g_f["n_atoms"].resize((new_frames_total,))
    for key in FRAME_SCALAR_KEYS:
        g_f[key].resize((new_frames_total,))

    # Write atoms.
    g_a["Z"][n_atoms_cur:new_atoms_total] = z_cat
    g_a["R"][n_atoms_cur:new_atoms_total, :] = r_cat
    for key in ATOM_SCALAR_KEYS:
        g_a[key][n_atoms_cur:new_atoms_total] = atom_cat[key]

    # Write frames.
    g_f["n_atoms"][n_frames_cur:new_frames_total] = n_atoms_arr
    for key in FRAME_SCALAR_KEYS:
        g_f[key][n_frames_cur:new_frames_total] = frame_arr[key]

    # Update offsets.
    offsets = g_f["offsets"]
    if offsets.shape[0] != n_frames_cur + 1:
        raise RuntimeError("offsets length mismatch")
    last_end = int(offsets[-1])
    if last_end != n_atoms_cur:
        raise RuntimeError(f"offsets[-1]={last_end} != n_atoms_cur={n_atoms_cur}")
    new_ends = last_end + np.cumsum(n_atoms_arr.astype(np.int64), dtype=np.int64)
    offsets.resize((offsets.shape[0] + n_new_frames,))
    offsets[-n_new_frames:] = new_ends


def _flush_buffer(f: h5py.File, buf: List[Dict]) -> None:
    if not buf:
        return
    _append_batch(f, buf)
    buf.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert XYZ + CDFT pairs to GDB13-aligned HDF5 shards")
    ap.add_argument("--input", nargs="+", required=True, help="Input file/dir/glob containing .xyz files")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--prefix", default="cdft", help="Output shard prefix")
    ap.add_argument("--recursive", dest="recursive", action="store_true", help="Recursively scan directories")
    ap.add_argument("--no_recursive", dest="recursive", action="store_false", help="Disable recursive scan")
    ap.set_defaults(recursive=True)

    ap.add_argument("--frames_per_shard", type=int, default=2_000_000)
    ap.add_argument("--flush_frames", type=int, default=20_000)
    ap.add_argument("--compression", default="lzf", help="lzf|gzip|none")
    ap.add_argument("--chunk_frames", type=int, default=131_072)
    ap.add_argument("--chunk_atoms_scalar", type=int, default=524_288)
    ap.add_argument("--chunk_atoms_coords", type=int, default=262_144)

    ap.add_argument("--canonical_energy_unit", choices=("ev", "hartree"), default="ev")
    ap.add_argument("--no_dedup", action="store_true", help="Disable geometry deduplication")
    ap.add_argument("--dedup_decimals", type=int, default=6, help="Coordinate quantization decimals for dedup")
    ap.add_argument("--conflict_policy", choices=("keep_first", "error"), default="keep_first")

    ap.add_argument("--strict_pairing", action="store_true", help="Fail if any .xyz has no matching .CDFT.txt")
    ap.add_argument("--on_error", choices=("skip", "raise"), default="skip", help="Per-file parse failure policy")
    ap.add_argument("--write_stats", dest="write_stats", action="store_true")
    ap.add_argument("--no_write_stats", dest="write_stats", action="store_false")
    ap.set_defaults(write_stats=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_files = _discover_xyz_files(args.input, recursive=args.recursive)
    if not xyz_files:
        raise ValueError("No .xyz files found")

    pairs: List[Tuple[Path, Path]] = []
    missing_pairs = 0
    for xyz in xyz_files:
        cdft = Path(str(xyz) + ".CDFT.txt")
        if cdft.exists():
            pairs.append((xyz, cdft))
        else:
            missing_pairs += 1
            if args.strict_pairing:
                raise FileNotFoundError(f"Missing paired CDFT file for {xyz}: expected {cdft}")

    if not pairs:
        raise ValueError("No valid xyz+CDFT pairs found")

    stats = {
        "inputs": [str(x) for x in args.input],
        "total_xyz_found": len(xyz_files),
        "missing_cdft_pairs": missing_pairs,
        "total_pairs": len(pairs),
        "processed_pairs": 0,
        "kept_frames": 0,
        "parse_errors": 0,
        "duplicates_removed": 0,
        "duplicate_conflicts": 0,
        "n_shards": 0,
        "canonical_energy_unit": args.canonical_energy_unit,
        "dedup_enabled": (not args.no_dedup),
        "dedup_decimals": int(args.dedup_decimals),
        "conflict_policy": args.conflict_policy,
        "errors": [],
    }

    def shard_path(i: int) -> Path:
        return out_dir / f"{args.prefix}_shard_{i:06d}.h5"

    shard_idx = 0
    frames_in_shard = 0
    f = _create_shard(
        path=shard_path(shard_idx),
        compression=args.compression,
        chunk_frames=args.chunk_frames,
        chunk_atoms_scalar=args.chunk_atoms_scalar,
        chunk_atoms_coords=args.chunk_atoms_coords,
        canonical_energy_unit=args.canonical_energy_unit,
    )
    stats["n_shards"] = 1

    buf: List[Dict] = []
    dedup_map: Dict[bytes, bytes] = {}
    pbar = tqdm(
        total=len(pairs),
        unit="mol",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}] {postfix}",
    )

    for xyz_path, cdft_path in pairs:
        stats["processed_pairs"] += 1
        try:
            xyz_symbols, z, r = _read_xyz(xyz_path)
            frame, atom = _parse_cdft(
                path=cdft_path,
                natoms=int(z.shape[0]),
                xyz_symbols=xyz_symbols,
                canonical_energy_unit=args.canonical_energy_unit,
            )

            if not args.no_dedup:
                ghash = _geometry_hash(z, r, decimals=args.dedup_decimals)
                sig = _sample_signature(frame, atom)
                prev = dedup_map.get(ghash)
                if prev is not None:
                    stats["duplicates_removed"] += 1
                    if sig != prev:
                        stats["duplicate_conflicts"] += 1
                        if args.conflict_policy == "error":
                            raise RuntimeError(
                                f"Duplicate geometry with conflicting CDFT labels: {xyz_path}"
                            )
                    pbar.update(1)
                    pbar.set_postfix(
                        kept=stats["kept_frames"],
                        dup=stats["duplicates_removed"],
                        err=stats["parse_errors"],
                        shard=shard_idx,
                    )
                    continue
                dedup_map[ghash] = sig

            sample = {
                "num_atoms": int(z.shape[0]),
                "Z": z,
                "R": r,
                "frame": frame,
                "atom": atom,
            }
            buf.append(sample)
            stats["kept_frames"] += 1
            frames_in_shard += 1

            if len(buf) >= args.flush_frames:
                _flush_buffer(f, buf)
                f.flush()

            if frames_in_shard >= args.frames_per_shard:
                _flush_buffer(f, buf)
                f.flush()
                f.close()
                shard_idx += 1
                frames_in_shard = 0
                f = _create_shard(
                    path=shard_path(shard_idx),
                    compression=args.compression,
                    chunk_frames=args.chunk_frames,
                    chunk_atoms_scalar=args.chunk_atoms_scalar,
                    chunk_atoms_coords=args.chunk_atoms_coords,
                    canonical_energy_unit=args.canonical_energy_unit,
                )
                stats["n_shards"] += 1

        except Exception as exc:
            stats["parse_errors"] += 1
            if len(stats["errors"]) < 200:
                stats["errors"].append(
                    {"xyz": str(xyz_path), "cdft": str(cdft_path), "error": str(exc)}
                )
            if args.on_error == "raise":
                pbar.close()
                f.close()
                raise

        pbar.update(1)
        pbar.set_postfix(
            kept=stats["kept_frames"],
            dup=stats["duplicates_removed"],
            err=stats["parse_errors"],
            shard=shard_idx,
        )

    _flush_buffer(f, buf)
    f.flush()
    f.close()
    pbar.close()

    if args.write_stats:
        stats_path = out_dir / f"{args.prefix}_convert_stats.json"
        with stats_path.open("w", encoding="utf-8") as w:
            json.dump(stats, w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
