#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
ATOM_ROW_RE = re.compile(r"^\s*(\d+)\(\s*([A-Za-z]{1,3})\s*\)\s+(.+)$")
FLOAT_TOKEN_RE = re.compile(FLOAT_RE)

PAIR_KEYS = ("q_n", "q_nplus1", "q_nminus1", "f_minus", "f_plus", "f_zero", "cdd")

# Periodic table symbols (1..118)
_PT = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
SYMBOL_TO_Z = {sym: z for z, sym in enumerate(_PT) if sym}


def _canonical_symbol(sym: str) -> str:
    sym = sym.strip()
    if not sym:
        raise ValueError("Empty element symbol")
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


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
        if any(ch in str(path) for ch in "*?[]"):
            parent = path.parent if str(path.parent) else Path(".")
            it = parent.rglob(path.name) if recursive else parent.glob(path.name)
            out.extend(x.resolve() for x in it if x.name.endswith(".xyz"))
    return sorted(set(out))


def _read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
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
            symbols.append(sym)
            z[i] = np.uint8(zi)
    return symbols, z


def _parse_cdft_qf_table(cdft_path: Path, xyz_symbols: Sequence[str], natoms: int) -> Dict[str, np.ndarray]:
    atom: Dict[str, np.ndarray] = {
        "q_n": np.full((natoms,), np.nan, dtype=np.float64),
        "q_nplus1": np.full((natoms,), np.nan, dtype=np.float64),
        "q_nminus1": np.full((natoms,), np.nan, dtype=np.float64),
        "f_minus": np.full((natoms,), np.nan, dtype=np.float64),
        "f_plus": np.full((natoms,), np.nan, dtype=np.float64),
        "f_zero": np.full((natoms,), np.nan, dtype=np.float64),
        "cdd": np.full((natoms,), np.nan, dtype=np.float64),
    }

    in_qf_table = False
    any_row = False

    with cdft_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_qf_table:
                if (
                    "q(N)" in line
                    and "q(N+1)" in line
                    and "q(N-1)" in line
                    and "f-" in line
                    and "f+" in line
                    and "f0" in line
                    and "CDD" in line
                ):
                    in_qf_table = True
                continue

            row = line.strip()
            if not row:
                if any_row:
                    break
                continue
            if set(row) <= {"-", "=", " "}:
                continue

            m = ATOM_ROW_RE.match(line)
            if m is None:
                if any_row:
                    break
                continue

            atom_idx = int(m.group(1)) - 1
            atom_sym = _canonical_symbol(m.group(2))
            nums = [float(x) for x in FLOAT_TOKEN_RE.findall(m.group(3))]

            if atom_idx < 0 or atom_idx >= natoms:
                raise ValueError(f"Atom index out of range in {cdft_path}: {atom_idx + 1}")
            if atom_sym != xyz_symbols[atom_idx]:
                raise ValueError(
                    f"Atom symbol mismatch in {cdft_path}: idx={atom_idx + 1}, "
                    f"cdft={atom_sym}, xyz={xyz_symbols[atom_idx]}"
                )
            if len(nums) < 7:
                raise ValueError(f"Incomplete q/f row in {cdft_path}: {line.strip()!r}")

            atom["q_n"][atom_idx] = nums[0]
            atom["q_nplus1"][atom_idx] = nums[1]
            atom["q_nminus1"][atom_idx] = nums[2]
            atom["f_minus"][atom_idx] = nums[3]
            atom["f_plus"][atom_idx] = nums[4]
            atom["f_zero"][atom_idx] = nums[5]
            atom["cdd"][atom_idx] = nums[6]
            any_row = True

    if not in_qf_table or not any_row:
        raise ValueError(f"Cannot find q/f(CDD) atom table in {cdft_path}")

    for k, arr in atom.items():
        if np.isnan(arr).any():
            raise ValueError(f"Missing atom values in {cdft_path}: key={k}")
    return atom


def _pair_cdft_file(xyz_path: Path) -> Path:
    return Path(str(xyz_path) + ".CDFT.txt")


def _all_exact_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b)


def _all_exact_zero(a: np.ndarray) -> bool:
    return np.array_equal(a, np.zeros_like(a))


def _max_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _check_single_pair(
    xyz_path: Path,
    cdft_path: Path,
    check_relations: bool,
    atol: float,
    rtol: float,
) -> List[str]:
    issues: List[str] = []

    xyz_symbols, z = _read_xyz(xyz_path)
    atom = _parse_cdft_qf_table(cdft_path=cdft_path, xyz_symbols=xyz_symbols, natoms=int(z.shape[0]))

    arr: Dict[str, np.ndarray] = {}
    for k in PAIR_KEYS:
        arr[k] = np.asarray(atom[k], dtype=np.float64)

    n = int(arr["q_n"].shape[0])

    if _all_exact_equal(arr["q_n"], arr["q_nminus1"]):
        issues.append("q_n and q_nminus1 are exactly identical for all atoms")
    if _all_exact_equal(arr["q_n"], arr["q_nplus1"]):
        issues.append("q_n and q_nplus1 are exactly identical for all atoms")
    if _all_exact_zero(arr["f_minus"]):
        issues.append("f_minus is exactly zero for all atoms")
    if _all_exact_zero(arr["f_plus"]):
        issues.append("f_plus is exactly zero for all atoms")
    if _all_exact_zero(arr["f_zero"]):
        issues.append("f_zero is exactly zero for all atoms")

    for i in range(len(PAIR_KEYS)):
        ki = PAIR_KEYS[i]
        for j in range(i + 1, len(PAIR_KEYS)):
            kj = PAIR_KEYS[j]
            if _all_exact_equal(arr[ki], arr[kj]):
                issues.append(f"{ki} and {kj} are exactly identical for all atoms")

    if check_relations:
        # Optional soft check: compare magnitudes to avoid sign-convention ambiguity.
        d_f_minus = np.abs(arr["q_n"] - arr["q_nminus1"])
        d_f_plus = np.abs(arr["q_nplus1"] - arr["q_n"])
        d_f_zero = 0.5 * (np.abs(arr["f_plus"]) + np.abs(arr["f_minus"]))

        if not np.allclose(np.abs(arr["f_minus"]), d_f_minus, atol=atol, rtol=rtol):
            issues.append(
                "f_minus magnitude mismatch with |q_n - q_nminus1|, "
                f"max_abs_diff={_max_abs(np.abs(arr['f_minus']) - d_f_minus):.6g}"
            )
        if not np.allclose(np.abs(arr["f_plus"]), d_f_plus, atol=atol, rtol=rtol):
            issues.append(
                "f_plus magnitude mismatch with |q_nplus1 - q_n|, "
                f"max_abs_diff={_max_abs(np.abs(arr['f_plus']) - d_f_plus):.6g}"
            )
        if not np.allclose(np.abs(arr["f_zero"]), d_f_zero, atol=atol, rtol=rtol):
            issues.append(
                "f_zero magnitude mismatch with (|f_plus| + |f_minus|)/2, "
                f"max_abs_diff={_max_abs(np.abs(arr['f_zero']) - d_f_zero):.6g}"
            )

    if issues:
        issues.insert(0, f"atoms={n}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check CDFT txt values for suspicious identical columns and "
            "consistency of f-/f+/f0 against q(N)/q(N+1)/q(N-1)."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input .xyz file(s), directories, or globs. The script expects paired *.xyz.CDFT.txt files.",
    )
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="Recursively scan directories")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false", help="Disable recursive scan")
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--check_relations",
        action="store_true",
        help="Also check f-/f+/f0 against q(N)/q(N+1)/q(N-1) by magnitude (optional).",
    )
    parser.add_argument("--atol", type=float, default=5e-4, help="Absolute tolerance for f-/f+/f0 consistency checks")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for f-/f+/f0 consistency checks")
    parser.add_argument(
        "--strict_pairing",
        action="store_true",
        help="Fail immediately when any discovered .xyz does not have a paired .CDFT.txt file.",
    )
    args = parser.parse_args()

    xyz_files = _discover_xyz_files(args.input, recursive=args.recursive)
    if not xyz_files:
        print("[error] no .xyz files found from --input")
        return 2

    total = 0
    bad = 0
    missing = 0

    for xyz_path in xyz_files:
        cdft_path = _pair_cdft_file(xyz_path)
        if not cdft_path.exists():
            missing += 1
            msg = f"[missing] {xyz_path} -> {cdft_path}"
            if args.strict_pairing:
                print(msg)
                return 2
            print(msg)
            continue

        total += 1
        try:
            issues = _check_single_pair(
                xyz_path=xyz_path,
                cdft_path=cdft_path,
                check_relations=args.check_relations,
                atol=args.atol,
                rtol=args.rtol,
            )
        except Exception as exc:
            bad += 1
            print(f"[error] {cdft_path}: parse/check failed: {exc}")
            continue

        if issues:
            bad += 1
            print(f"[suspect] {cdft_path}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[ok] {cdft_path}")

    print(
        "[summary] "
        f"checked_pairs={total}, suspect_or_error={bad}, missing_pairs={missing}, "
        f"inputs_xyz={len(xyz_files)}"
    )
    return 1 if bad > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
