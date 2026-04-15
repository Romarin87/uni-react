"""Reaction triplet HDF5 dataset for stage-3 pretraining."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore


class ReactionTripletH5Dataset(Dataset):
    """Stage-3 reaction-triplet dataset backed by a single HDF5 file.

    Required HDF5 schema under ``/triplets``:

    - ``offsets``  int64  [n_triplets + 1]
    - ``r_Z``, ``ts_Z``, ``p_Z``   integer  [total_atoms]
    - ``r_R``, ``ts_R``, ``p_R``   float32  [total_atoms, 3]

    Optional:

    - ``n_atoms``   int32   [n_triplets] (inferred from offsets if absent)
    - ``comp_hash`` int64   [n_triplets] (defaults to 0 if absent)

    Each ``__getitem__`` returns a dict with keys:
    ``R``, ``TS``, ``P`` (positive triplet) and
    ``R_cons``, ``TS_cons``, ``P_cons`` (same or negative triplet),
    plus ``cons_label`` (1.0 = consistent / 0.0 = negative).
    """

    def __init__(
        self,
        h5_path: str,
        neg_ratio: float = 0.5,
        hard_negative: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if h5py is None:
            raise ImportError("h5py is required for ReactionTripletH5Dataset. pip install h5py")
        self._handles: Dict[int, "h5py.File"] = {}
        if not (0.0 <= neg_ratio <= 1.0):
            raise ValueError("neg_ratio must be in [0, 1]")

        self.h5_path = str(Path(h5_path).resolve())
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            if "triplets" not in f:
                raise ValueError(f"Missing group '/triplets' in {self.h5_path}")
            g = f["triplets"]
            for k in ("offsets", "r_Z", "ts_Z", "p_Z", "r_R", "ts_R", "p_R"):
                if k not in g:
                    raise ValueError(f"Missing '/triplets/{k}' in {self.h5_path}")

            offsets = np.asarray(g["offsets"][:], dtype=np.int64)
            if offsets.ndim != 1 or offsets.shape[0] < 2:
                raise ValueError("/triplets/offsets must be 1-D with length >= 2")
            self.offsets = offsets
            self.n_triplets = int(offsets.shape[0] - 1)

            n_atoms = (
                np.asarray(g["n_atoms"][:], dtype=np.int32)
                if "n_atoms" in g
                else np.diff(offsets).astype(np.int32)
            )
            if n_atoms.shape[0] != self.n_triplets:
                raise ValueError("/triplets/n_atoms length mismatch")
            self.n_atoms = n_atoms

            comp_hash = (
                np.asarray(g["comp_hash"][:], dtype=np.int64)
                if "comp_hash" in g
                else np.zeros((self.n_triplets,), dtype=np.int64)
            )
            self.comp_hash = comp_hash

        self.neg_ratio    = float(neg_ratio)
        self.hard_negative = bool(hard_negative)
        self.seed         = int(seed)

        self._pool_exact:  Dict[Tuple[int, int], List[int]] = {}
        self._pool_natoms: Dict[int, List[int]] = {}
        for i in range(self.n_triplets):
            nat = int(self.n_atoms[i])
            ch  = int(self.comp_hash[i])
            self._pool_exact.setdefault((nat, ch), []).append(i)
            self._pool_natoms.setdefault(nat, []).append(i)

    def __len__(self) -> int:
        return self.n_triplets

    def _worker_id(self) -> int:
        info = torch.utils.data.get_worker_info()
        return 0 if info is None else int(info.id)

    def _h5(self) -> "h5py.File":
        wid = self._worker_id()
        h = self._handles.get(wid)
        if h is None:
            h = h5py.File(self.h5_path, "r")
            self._handles[wid] = h
        return h

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + idx * 7919 + self._worker_id() * 104729)

    def _sample_negative_index(self, idx: int, rng: np.random.Generator) -> Optional[int]:
        natoms = int(self.n_atoms[idx])
        ch     = int(self.comp_hash[idx])
        exact  = [x for x in self._pool_exact.get((natoms, ch), []) if x != idx]
        if exact and self.hard_negative:
            return int(exact[int(rng.integers(0, len(exact)))])
        same_n = [x for x in self._pool_natoms.get(natoms, []) if x != idx]
        if same_n:
            return int(same_n[int(rng.integers(0, len(same_n)))])
        any_other = [x for x in range(self.n_triplets) if x != idx]
        return int(any_other[int(rng.integers(0, len(any_other)))]) if any_other else None

    def _slice_mol(self, g: "h5py.Group", prefix: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = int(self.offsets[idx])
        end   = int(self.offsets[idx + 1])
        z = np.asarray(g[f"{prefix}_Z"][start:end], dtype=np.int64)
        r = np.asarray(g[f"{prefix}_R"][start:end], dtype=np.float32)
        return z, r

    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        rng = self._rng_for_index(idx)
        g   = self._h5()["triplets"]

        z_r,  r_r  = self._slice_mol(g, "r",  idx)
        z_ts, r_ts = self._slice_mol(g, "ts", idx)
        z_p,  r_p  = self._slice_mol(g, "p",  idx)

        # Build consistency pair (positive or negative)
        if rng.random() < self.neg_ratio:
            neg_idx = self._sample_negative_index(idx, rng)
            if neg_idx is None:
                z_r_c,  r_r_c  = z_r,  r_r
                z_ts_c, r_ts_c = z_ts, r_ts
                z_p_c,  r_p_c  = z_p,  r_p
                cons_label = 1.0
            else:
                z_r_c,  r_r_c  = self._slice_mol(g, "r",  neg_idx)
                z_ts_c, r_ts_c = self._slice_mol(g, "ts", neg_idx)
                z_p_c,  r_p_c  = self._slice_mol(g, "p",  neg_idx)
                cons_label = 0.0
        else:
            z_r_c,  r_r_c  = z_r,  r_r
            z_ts_c, r_ts_c = z_ts, r_ts
            z_p_c,  r_p_c  = z_p,  r_p
            cons_label = 1.0

        return {
            "R":      (z_r,  r_r),
            "TS":     (z_ts, r_ts),
            "P":      (z_p,  r_p),
            "R_cons":  (z_r_c,  r_r_c),
            "TS_cons": (z_ts_c, r_ts_c),
            "P_cons":  (z_p_c,  r_p_c),
            "cons_label": cons_label,
        }

    def __del__(self) -> None:
        for h in getattr(self, "_handles", {}).values():
            try:
                h.close()
            except Exception:
                pass


def _pad_molecules(
    mols: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz      = len(mols)
    max_atoms = max(int(z.shape[0]) for z, _ in mols)
    z_pad       = torch.zeros((bsz, max_atoms),    dtype=torch.long)
    r_pad       = torch.zeros((bsz, max_atoms, 3), dtype=torch.float32)
    atom_padding = torch.ones((bsz, max_atoms),    dtype=torch.bool)
    for i, (z, r) in enumerate(mols):
        n = int(z.shape[0])
        z_pad[i, :n]       = torch.from_numpy(z.astype(np.int64,   copy=False))
        r_pad[i, :n]       = torch.from_numpy(r.astype(np.float32, copy=False))
        atom_padding[i, :n] = False
    return z_pad, r_pad, atom_padding


def collate_reaction_triplet(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of reaction-triplet samples into a padded batch dict."""
    out: Dict[str, torch.Tensor] = {}
    for role in ("R", "TS", "P", "R_cons", "TS_cons", "P_cons"):
        mols = [item[role] for item in batch]
        z_pad, r_pad, atom_padding = _pad_molecules(mols)
        out[f"{role}_atomic_numbers"] = z_pad
        out[f"{role}_coords"]         = r_pad
        out[f"{role}_padding"]        = atom_padding
    out["cons_label"] = torch.tensor(
        [float(item["cons_label"]) for item in batch], dtype=torch.float32
    )
    return out


def split_dataset(
    dataset: Dataset, val_ratio: float, seed: int
) -> Tuple[Dataset, Optional[Dataset]]:
    """Split a dataset into train and validation subsets."""
    n = len(dataset)
    if n < 2 or val_ratio <= 0.0:
        return dataset, None
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val   = min(max(1, int(round(n * val_ratio))), n - 1)
    return Subset(dataset, idx[n_val:].tolist()), Subset(dataset, idx[:n_val].tolist())
