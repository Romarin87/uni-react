"""Density pretraining dataset utilities."""
from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5DensityPretrainDataset(Dataset):
    """Point-sampled density dataset from flattened ED HDF5 shards."""

    def __init__(
        self,
        h5_files: Sequence[str],
        num_query_points: int = 2048,
        center_coords: bool = True,
        deterministic: bool = False,
        seed: int = 42,
        return_ids: bool = False,
    ) -> None:
        super().__init__()
        if not h5_files:
            raise ValueError("h5_files is empty")
        if num_query_points <= 0:
            raise ValueError("num_query_points must be > 0")

        self.h5_files = [str(Path(p)) for p in h5_files]
        self.num_query_points = int(num_query_points)
        self.center_coords = bool(center_coords)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.return_ids = bool(return_ids)

        self._file_meta: List[Dict[str, int]] = []
        self._cum_frames = np.zeros((len(self.h5_files) + 1,), dtype=np.int64)
        for i, path in enumerate(self.h5_files):
            with h5py.File(path, "r") as h5:
                self._validate_schema(path=path, h5=h5)
                n_frames = int(h5["frames"]["n_atoms"].shape[0])
            self._file_meta.append({"n_frames": n_frames})
            self._cum_frames[i + 1] = self._cum_frames[i] + n_frames

        self._handles: Dict[str, h5py.File] = {}
        self._worker_rngs: Dict[int, np.random.Generator] = {}

    @staticmethod
    def _validate_schema(path: str, h5: h5py.File) -> None:
        req = [
            "frames/atom_offsets",
            "frames/n_atoms",
            "frames/density_offsets",
            "frames/n_voxels",
            "frames/grid_shape",
            "frames/grid_origin",
            "frames/grid_vectors",
            "frames/total_charge",
            "frames/spin_multiplicity",
            "atoms/Z",
            "atoms/R",
            "density/target",
        ]
        missing = [k for k in req if k not in h5]
        if missing:
            raise ValueError(f"Missing required datasets in {path}: {missing}")

        n_frames = int(h5["frames"]["n_atoms"].shape[0])
        for k in ("atom_offsets", "density_offsets", "n_voxels", "total_charge", "spin_multiplicity"):
            if int(h5["frames"][k].shape[0]) != n_frames:
                raise ValueError(
                    f"{path}: frames/{k} length mismatch. got {h5['frames'][k].shape[0]} expected {n_frames}"
                )
        if tuple(h5["frames"]["grid_shape"].shape) != (n_frames, 3):
            raise ValueError(f"{path}: frames/grid_shape must have shape (n_frames, 3)")
        if tuple(h5["frames"]["grid_origin"].shape) != (n_frames, 3):
            raise ValueError(f"{path}: frames/grid_origin must have shape (n_frames, 3)")
        if tuple(h5["frames"]["grid_vectors"].shape) != (n_frames, 3, 3):
            raise ValueError(f"{path}: frames/grid_vectors must have shape (n_frames, 3, 3)")

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
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + idx)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        rng = self._worker_rngs.get(worker_id)
        if rng is None:
            base_seed = (self.seed + 10007 * (worker_id + 1) + 7919 * os.getpid()) & 0xFFFFFFFF
            rng = np.random.default_rng(base_seed)
            self._worker_rngs[worker_id] = rng
        return rng

    @staticmethod
    def _sample_points_from_grid(
        rng: np.random.Generator,
        num_query_points: int,
        density_all: np.ndarray,
        grid_shape: np.ndarray,
        grid_origin: np.ndarray,
        grid_vectors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = [int(v) for v in grid_shape.tolist()]
        n_vox = int(nx * ny * nz)
        if n_vox <= 0:
            raise ValueError(f"Invalid n_vox from grid shape: {grid_shape.tolist()}")
        if density_all.shape[0] != n_vox:
            raise ValueError(f"density size mismatch: got {density_all.shape[0]} expected {n_vox}")

        replace = n_vox < num_query_points
        sampled = rng.choice(n_vox, size=num_query_points, replace=replace)

        yz = ny * nz
        i = sampled // yz
        rem = sampled - i * yz
        j = rem // nz
        k = rem - j * nz
        points = (
            grid_origin[None, :]
            + i[:, None] * grid_vectors[0][None, :]
            + j[:, None] * grid_vectors[1][None, :]
            + k[:, None] * grid_vectors[2][None, :]
        ).astype(np.float32, copy=False)
        targets = density_all[sampled].astype(np.float32, copy=False)
        return points, targets

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_i, local_idx = self._index_to_file(idx)
        path = self.h5_files[file_i]
        h5 = self._get_handle(path)
        f = h5["frames"]
        a = h5["atoms"]
        d = h5["density"]

        atom_start = int(f["atom_offsets"][local_idx])
        n_atoms = int(f["n_atoms"][local_idx])
        atom_end = atom_start + n_atoms
        if n_atoms <= 0:
            raise RuntimeError(f"empty frame: {path}#{local_idx}")

        den_start = int(f["density_offsets"][local_idx])
        n_vox = int(f["n_voxels"][local_idx])
        den_end = den_start + n_vox
        if n_vox <= 0:
            raise RuntimeError(f"empty density: {path}#{local_idx}")

        z = np.asarray(a["Z"][atom_start:atom_end], dtype=np.int64)
        coords = np.asarray(a["R"][atom_start:atom_end], dtype=np.float32)
        grid_shape = np.asarray(f["grid_shape"][local_idx], dtype=np.int64)
        grid_origin = np.asarray(f["grid_origin"][local_idx], dtype=np.float32)
        grid_vectors = np.asarray(f["grid_vectors"][local_idx], dtype=np.float32)
        density_all = np.asarray(d["target"][den_start:den_end], dtype=np.float32)
        total_charge = float(f["total_charge"][local_idx])
        spin_mult = float(f["spin_multiplicity"][local_idx])

        rng = self._rng(idx)
        query_points, density_target = self._sample_points_from_grid(
            rng=rng,
            num_query_points=self.num_query_points,
            density_all=density_all,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            grid_vectors=grid_vectors,
        )

        if self.center_coords:
            center = coords.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
            coords = coords - center
            query_points = query_points - center

        out: Dict[str, torch.Tensor] = {
            "atomic_numbers": torch.from_numpy(z).long(),
            "coords": torch.from_numpy(coords).float(),
            "query_points": torch.from_numpy(query_points).float(),
            "density_target": torch.from_numpy(density_target).float(),
            "total_charge": torch.tensor(total_charge, dtype=torch.float32),
            "spin_multiplicity": torch.tensor(spin_mult, dtype=torch.float32),
        }
        if self.return_ids:
            sid = f"{Path(path).name}#{local_idx}"
            out["sample_id"] = sid  # type: ignore[assignment]
        return out


def collate_fn_density(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_atoms = max(int(x["atomic_numbers"].shape[0]) for x in batch)
    n_query = int(batch[0]["query_points"].shape[0])

    atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    atom_padding = torch.ones((batch_size, max_atoms), dtype=torch.bool)
    num_atoms = torch.zeros((batch_size,), dtype=torch.long)
    query_points = torch.zeros((batch_size, n_query, 3), dtype=torch.float32)
    density_target = torch.zeros((batch_size, n_query), dtype=torch.float32)
    total_charge = torch.zeros((batch_size,), dtype=torch.float32)
    spin_multiplicity = torch.zeros((batch_size,), dtype=torch.float32)

    for i, item in enumerate(batch):
        n = int(item["atomic_numbers"].shape[0])
        atomic_numbers[i, :n] = item["atomic_numbers"]
        coords[i, :n] = item["coords"]
        atom_padding[i, :n] = False
        num_atoms[i] = n
        query_points[i] = item["query_points"]
        density_target[i] = item["density_target"]
        total_charge[i] = item["total_charge"]
        spin_multiplicity[i] = item["spin_multiplicity"]

    return {
        "atomic_numbers": atomic_numbers,
        "coords": coords,
        "atom_padding": atom_padding,
        "num_atoms": num_atoms,
        "query_points": query_points,
        "density_target": density_target,
        "total_charge": total_charge,
        "spin_multiplicity": spin_multiplicity,
    }
