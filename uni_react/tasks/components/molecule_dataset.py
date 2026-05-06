import bisect
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


PathLike = Union[str, Path]


class H5SingleMolPretrainDataset(Dataset):
    """
    Single-molecule pretraining dataset from HDF5 shards.

    Supported schemas:
    1) stable_gen shard schema:
       - /frames/offsets
       - /atoms/Z, /atoms/R, /atoms/q (optional canonical alias)
       - typed charge aliases are also accepted when /atoms/q is absent:
         /atoms/q_mulliken or /atoms/q_hirshfeld
    2) extxyz_to_hdf5 schema:
       - /mol_offsets
       - /atom_numbers, /coords, /charges (optional)

    Optional electronic-structure labels (direct regression):
      - global targets (per-frame), e.g. VIP/VEA
      - atom targets (per-atom), e.g. f+/f-/f0
    """

    def __init__(
        self,
        h5_files: Sequence[PathLike],
        mask_ratio: float = 0.15,
        mask_token_id: int = 94,
        atom_vocab_size: Optional[int] = None,
        min_masked: int = 1,
        max_masked: Optional[int] = None,
        noise_std: float = 0.02,
        center_coords: bool = True,
        recenter_noisy: bool = True,
        deterministic: bool = False,
        seed: int = 0,
        return_ids: bool = False,
        require_reactivity: bool = False,
        reactivity_global_keys: Optional[Sequence[str]] = None,
        reactivity_atom_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if not h5_files:
            raise ValueError("h5_files is empty")
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0, 1]")
        if min_masked < 0:
            raise ValueError("min_masked must be >= 0")
        if max_masked is not None and max_masked < 0:
            raise ValueError("max_masked must be >= 0")
        if atom_vocab_size is not None and atom_vocab_size <= 0:
            raise ValueError("atom_vocab_size must be > 0")
        if atom_vocab_size is not None and mask_token_id >= atom_vocab_size:
            raise ValueError(
                f"mask_token_id ({mask_token_id}) must be < atom_vocab_size ({atom_vocab_size})"
            )

        self.h5_files = [str(Path(p)) for p in h5_files]
        self.mask_ratio = float(mask_ratio)
        self.mask_token_id = int(mask_token_id)
        self.atom_vocab_size = atom_vocab_size
        self.min_masked = int(min_masked)
        self.max_masked = max_masked
        self.noise_std = float(noise_std)
        self.center_coords = bool(center_coords)
        self.recenter_noisy = bool(recenter_noisy)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.return_ids = bool(return_ids)
        self.require_reactivity = bool(require_reactivity)
        self.reactivity_global_keys = (
            ("vip", "vea") if reactivity_global_keys is None else tuple(reactivity_global_keys)
        )
        self.reactivity_atom_keys = (
            ("f_plus", "f_minus", "f_zero")
            if reactivity_atom_keys is None
            else tuple(reactivity_atom_keys)
        )
        if len(set(self.reactivity_global_keys)) != len(self.reactivity_global_keys):
            raise ValueError(f"Duplicate reactivity_global_keys: {self.reactivity_global_keys}")
        if len(set(self.reactivity_atom_keys)) != len(self.reactivity_atom_keys):
            raise ValueError(f"Duplicate reactivity_atom_keys: {self.reactivity_atom_keys}")

        # Per-file metadata:
        # {
        #   "n_frames": int,
        #   "schema": str,
        #   "reactivity_global_paths": Dict[str, str],
        #   "reactivity_atom_paths": Dict[str, str],
        # }
        self._file_meta: List[Dict] = []
        for path in self.h5_files:
            with h5py.File(path, "r") as h5:
                n_frames, schema = self._inspect_file(h5)
                meta = {
                    "n_frames": n_frames,
                    "schema": schema,
                    "reactivity_global_paths": {},
                    "reactivity_atom_paths": {},
                }
                if self.require_reactivity:
                    global_paths = self._resolve_reactivity_paths(
                        h5=h5,
                        schema=schema,
                        keys=self.reactivity_global_keys,
                        kind="global",
                    )
                    atom_paths = self._resolve_reactivity_paths(
                        h5=h5,
                        schema=schema,
                        keys=self.reactivity_atom_keys,
                        kind="atom",
                    )
                    self._validate_reactivity_shapes(
                        h5=h5,
                        schema=schema,
                        n_frames=n_frames,
                        global_paths=global_paths,
                        atom_paths=atom_paths,
                        file_path=path,
                    )
                    meta["reactivity_global_paths"] = global_paths
                    meta["reactivity_atom_paths"] = atom_paths
                self._file_meta.append(meta)

        self._cum_frames = np.zeros(len(self._file_meta) + 1, dtype=np.int64)
        for i, meta in enumerate(self._file_meta):
            n_frames = int(meta["n_frames"])
            self._cum_frames[i + 1] = self._cum_frames[i] + n_frames

        self._handles: Dict[str, h5py.File] = {}
        self._worker_rngs: Dict[int, np.random.Generator] = {}

    @staticmethod
    def _inspect_file(h5: h5py.File) -> Tuple[int, str]:
        if "frames" in h5 and "offsets" in h5["frames"]:
            n_frames = int(h5["frames"]["offsets"].shape[0]) - 1
            if n_frames < 0:
                raise ValueError("invalid /frames/offsets")
            return n_frames, "stable_gen"

        if "mol_offsets" in h5:
            n_frames = int(h5["mol_offsets"].shape[0]) - 1
            if n_frames < 0:
                raise ValueError("invalid /mol_offsets")
            return n_frames, "extxyz"

        raise ValueError("unsupported h5 schema: require /frames/offsets or /mol_offsets")

    @staticmethod
    def _dataset_exists(h5: h5py.File, path: str) -> bool:
        try:
            h5[path]
            return True
        except KeyError:
            return False

    @classmethod
    def _resolve_reactivity_path(
        cls,
        h5: h5py.File,
        schema: str,
        key: str,
        kind: str,
    ) -> Optional[str]:
        # Explicit HDF5 path support, e.g. "/frames/vip" or "frames/vip".
        if "/" in key:
            path = key.lstrip("/")
            return path if cls._dataset_exists(h5, path) else None

        if kind not in ("global", "atom"):
            raise ValueError(f"Invalid reactivity kind: {kind}")

        candidates: List[str] = []
        if schema == "stable_gen":
            if kind == "global":
                candidates = [f"frames/{key}", key]
            else:
                candidates = [f"atoms/{key}", key]
        else:
            # extxyz schema often keeps atom arrays at root.
            if kind == "global":
                candidates = [f"frames/{key}", key]
            else:
                candidates = [key, f"atoms/{key}"]

        for path in candidates:
            if cls._dataset_exists(h5, path):
                return path
        return None

    @classmethod
    def _resolve_reactivity_paths(
        cls,
        h5: h5py.File,
        schema: str,
        keys: Sequence[str],
        kind: str,
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        missing: List[str] = []
        for key in keys:
            path = cls._resolve_reactivity_path(h5=h5, schema=schema, key=key, kind=kind)
            if path is None:
                missing.append(key)
            else:
                out[key] = path
        if missing:
            raise ValueError(
                f"Missing required reactivity {kind} datasets: {missing}. "
                f"schema={schema}, keys={list(keys)}"
            )
        return out

    @staticmethod
    def _validate_reactivity_shapes(
        h5: h5py.File,
        schema: str,
        n_frames: int,
        global_paths: Dict[str, str],
        atom_paths: Dict[str, str],
        file_path: str,
    ) -> None:
        for key, path in global_paths.items():
            ds = h5[path]
            if ds.shape[0] != n_frames:
                raise ValueError(
                    f"Reactivity global dataset shape mismatch for {file_path}: "
                    f"{key} -> {path}, len={ds.shape[0]}, expected_frames={n_frames}"
                )

        if schema == "stable_gen":
            n_atoms_total = int(h5["atoms"]["Z"].shape[0])
        else:
            n_atoms_total = int(h5["atom_numbers"].shape[0])

        for key, path in atom_paths.items():
            ds = h5[path]
            if ds.shape[0] != n_atoms_total:
                raise ValueError(
                    f"Reactivity atom dataset shape mismatch for {file_path}: "
                    f"{key} -> {path}, len={ds.shape[0]}, expected_atoms={n_atoms_total}"
                )

    def __len__(self) -> int:
        return int(self._cum_frames[-1])

    def _index_to_file(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_i = bisect.bisect_right(self._cum_frames, idx) - 1
        local_idx = int(idx - self._cum_frames[file_i])
        return file_i, local_idx

    def _get_handle(self, path: str) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _read_frame(self, h5: h5py.File, schema: str, frame_idx: int):
        if schema == "stable_gen":
            offsets = h5["frames"]["offsets"]
            start = int(offsets[frame_idx])
            end = int(offsets[frame_idx + 1])
            z = np.asarray(h5["atoms"]["Z"][start:end], dtype=np.int64)
            coords = np.asarray(h5["atoms"]["R"][start:end], dtype=np.float32)
            q = None
            atoms = h5["atoms"]
            if "q" in atoms:
                q = np.asarray(atoms["q"][start:end], dtype=np.float32)
            else:
                has_mulliken = "q_mulliken" in atoms
                has_hirshfeld = "q_hirshfeld" in atoms
                if has_mulliken and has_hirshfeld:
                    raise ValueError(
                        "Found both /atoms/q_mulliken and /atoms/q_hirshfeld without /atoms/q. "
                        "Please provide a canonical /atoms/q to avoid ambiguous charge semantics."
                    )
                if has_mulliken:
                    q = np.asarray(atoms["q_mulliken"][start:end], dtype=np.float32)
                elif has_hirshfeld:
                    q = np.asarray(atoms["q_hirshfeld"][start:end], dtype=np.float32)
            return z, coords, q, start, end

        offsets = h5["mol_offsets"]
        start = int(offsets[frame_idx])
        end = int(offsets[frame_idx + 1])
        z = np.asarray(h5["atom_numbers"][start:end], dtype=np.int64)
        coords = np.asarray(h5["coords"][start:end], dtype=np.float32)
        q = None
        if "charges" in h5:
            q = np.asarray(h5["charges"][start:end], dtype=np.float32)
        return z, coords, q, start, end

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + idx)
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        rng = self._worker_rngs.get(worker_id)
        if rng is None:
            base_seed = (self.seed + 1009 * (worker_id + 1) + 7919 * os.getpid()) & 0xFFFFFFFF
            rng = np.random.default_rng(base_seed)
            self._worker_rngs[worker_id] = rng
        return rng

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_i, local_idx = self._index_to_file(idx)
        path = self.h5_files[file_i]
        meta = self._file_meta[file_i]
        schema = meta["schema"]

        h5 = self._get_handle(path)
        atomic_numbers, coords, charges, atom_start, atom_end = self._read_frame(h5, schema, local_idx)
        num_atoms = int(atomic_numbers.shape[0])
        if num_atoms <= 0:
            raise RuntimeError(f"empty frame: {path}#{local_idx}")

        if self.atom_vocab_size is not None:
            if atomic_numbers.min() < 0 or atomic_numbers.max() >= self.atom_vocab_size:
                raise ValueError(
                    f"atomic number out of range in {path}#{local_idx}: "
                    f"min={int(atomic_numbers.min())}, max={int(atomic_numbers.max())}, "
                    f"atom_vocab_size={self.atom_vocab_size}"
                )

        if self.center_coords:
            coords = coords - coords.mean(axis=0, keepdims=True)

        rng = self._rng(idx)

        n_masked = int(round(self.mask_ratio * num_atoms))
        if self.mask_ratio > 0:
            n_masked = max(self.min_masked, n_masked)
        if self.max_masked is not None:
            n_masked = min(n_masked, self.max_masked)
        n_masked = max(0, min(num_atoms, n_masked))

        mask_positions = np.zeros((num_atoms,), dtype=bool)
        if n_masked > 0:
            masked_idx = rng.choice(num_atoms, size=n_masked, replace=False)
            mask_positions[masked_idx] = True

        input_atomic_numbers = atomic_numbers.copy()
        input_atomic_numbers[mask_positions] = self.mask_token_id

        if self.noise_std > 0:
            noise = rng.normal(0.0, self.noise_std, size=coords.shape).astype(np.float32)
            coords_noisy = coords + noise
        else:
            noise = np.zeros_like(coords, dtype=np.float32)
            coords_noisy = coords.copy()

        if self.recenter_noisy:
            coords_noisy = coords_noisy - coords_noisy.mean(axis=0, keepdims=True)

        if charges is None:
            charge_values = np.zeros((num_atoms,), dtype=np.float32)
            charge_valid = np.zeros((num_atoms,), dtype=bool)
        else:
            charge_values = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            charge_valid = np.isfinite(charges)

        sample = {
            "atomic_numbers": torch.from_numpy(atomic_numbers).long(),
            "input_atomic_numbers": torch.from_numpy(input_atomic_numbers).long(),
            "coords": torch.from_numpy(coords).float(),
            "coords_noisy": torch.from_numpy(coords_noisy).float(),
            "noise": torch.from_numpy(noise).float(),
            "mask_positions": torch.from_numpy(mask_positions),
            "charges": torch.from_numpy(charge_values).float(),
            "charge_valid": torch.from_numpy(charge_valid),
        }
        if self.require_reactivity:
            reactivity_global = np.zeros((len(self.reactivity_global_keys),), dtype=np.float32)
            for i, key in enumerate(self.reactivity_global_keys):
                ds_path = meta["reactivity_global_paths"][key]
                reactivity_global[i] = np.float32(h5[ds_path][local_idx])

            reactivity_atom = np.zeros((num_atoms, len(self.reactivity_atom_keys)), dtype=np.float32)
            for i, key in enumerate(self.reactivity_atom_keys):
                ds_path = meta["reactivity_atom_paths"][key]
                reactivity_atom[:, i] = np.asarray(h5[ds_path][atom_start:atom_end], dtype=np.float32)

            if (not np.isfinite(reactivity_global).all()) or (not np.isfinite(reactivity_atom).all()):
                raise ValueError(
                    f"Non-finite reactivity label found in {path}#{local_idx}. "
                    f"global_keys={self.reactivity_global_keys}, atom_keys={self.reactivity_atom_keys}"
                )
            sample["reactivity_global"] = torch.from_numpy(reactivity_global).float()
            sample["reactivity_atom"] = torch.from_numpy(reactivity_atom).float()

        if self.return_ids:
            sample["sample_id"] = f"{Path(path).name}#{local_idx}"
        return sample

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def collate_fn_pretrain(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_atoms = max(item["atomic_numbers"].shape[0] for item in batch)
    has_reactivity = "reactivity_global" in batch[0]

    atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    input_atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    coords_noisy = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    noise = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    charges = torch.zeros((batch_size, max_atoms), dtype=torch.float32)

    mask_positions = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    charge_valid = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    atom_padding = torch.ones((batch_size, max_atoms), dtype=torch.bool)

    num_atoms = torch.zeros((batch_size,), dtype=torch.long)
    reactivity_global = None
    reactivity_atom = None
    reactivity_atom_valid = None
    if has_reactivity:
        n_global = int(batch[0]["reactivity_global"].shape[0])
        n_atom = int(batch[0]["reactivity_atom"].shape[1])
        reactivity_global = torch.zeros((batch_size, n_global), dtype=torch.float32)
        reactivity_atom = torch.zeros((batch_size, max_atoms, n_atom), dtype=torch.float32)
        reactivity_atom_valid = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    sample_ids: List[str] = []
    has_ids = "sample_id" in batch[0]

    for i, item in enumerate(batch):
        n = item["atomic_numbers"].shape[0]
        num_atoms[i] = n
        atom_padding[i, :n] = False

        atomic_numbers[i, :n] = item["atomic_numbers"]
        input_atomic_numbers[i, :n] = item["input_atomic_numbers"]
        coords[i, :n] = item["coords"]
        coords_noisy[i, :n] = item["coords_noisy"]
        noise[i, :n] = item["noise"]
        mask_positions[i, :n] = item["mask_positions"]
        charges[i, :n] = item["charges"]
        charge_valid[i, :n] = item["charge_valid"]
        if has_reactivity:
            reactivity_global[i] = item["reactivity_global"]
            reactivity_atom[i, :n] = item["reactivity_atom"]
            reactivity_atom_valid[i, :n] = True

        if has_ids:
            sample_ids.append(item["sample_id"])

    out = {
        "atomic_numbers": atomic_numbers,
        "input_atomic_numbers": input_atomic_numbers,
        "coords": coords,
        "coords_noisy": coords_noisy,
        "noise": noise,
        "mask_positions": mask_positions,
        "charges": charges,
        "charge_valid": charge_valid,
        "atom_padding": atom_padding,
        "num_atoms": num_atoms,
    }
    if has_reactivity:
        out["reactivity_global"] = reactivity_global
        out["reactivity_atom"] = reactivity_atom
        out["reactivity_atom_valid"] = reactivity_atom_valid
    if has_ids:
        out["sample_ids"] = sample_ids
    return out
