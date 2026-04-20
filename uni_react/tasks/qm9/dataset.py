import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


QM9_TARGETS = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
)

# Align energy targets with the common QM9 benchmark setup:
# U0/U/H/G read the atomization-energy columns from PyG QM9 (indices 12-15),
# while the user-facing target names stay unchanged.
QM9_TARGET_INDEX = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "U0": 12,
    "U": 13,
    "H": 14,
    "G": 15,
    "Cv": 11,
}
GOTENNET_QM9_TARGET_INDEX = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "U0": 7,
    "U": 8,
    "H": 9,
    "G": 10,
    "Cv": 11,
}
QM9_TARGET_INDEX_VARIANTS = {
    "default": QM9_TARGET_INDEX,
    "gotennet": GOTENNET_QM9_TARGET_INDEX,
}
QM9_SPLIT_MODES = ("egnn", "dimenet", "gotennet")
QM9_SPLIT_SPECS = {
    # Common Cormorant / EGNN split.
    "egnn": {"seed": 0, "sizes": (100000, 17748, 13083)},
    # Common DimeNet / DimeNet++ split.
    "dimenet": {"seed": 0, "sizes": (110000, 10000, 10831)},
    # Official GotenNet QM9 split.
    "gotennet": {"seed": 1, "sizes": (110000, 10000, 10831)},
}


def _resolve_targets(target: str, targets: Optional[Sequence[str]]) -> List[str]:
    if targets is None or len(targets) == 0:
        targets = [target]
    out = list(targets)
    if len(out) == 1 and out[0].lower() == "all":
        return list(QM9_TARGETS)
    if any(name.lower() == "all" for name in out):
        raise ValueError("`all` cannot be combined with other targets. Use only `--targets all`.")
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate targets are not allowed: {out}")
    bad = [name for name in out if name not in QM9_TARGETS]
    if bad:
        raise ValueError(f"Unsupported targets {bad}, available: {QM9_TARGETS}")
    return out


def get_qm9_target_index_map(variant: str = "default") -> Dict[str, int]:
    if variant not in QM9_TARGET_INDEX_VARIANTS:
        raise ValueError(
            f"Unsupported QM9 target index variant '{variant}', choose from {tuple(QM9_TARGET_INDEX_VARIANTS)}"
        )
    return QM9_TARGET_INDEX_VARIANTS[variant]


# Module-level class so it can be pickled by multiprocessing DataLoader workers.
class _QM9ProcessedOnly:
    """Placeholder – replaced at runtime once torch_geometric is imported."""
    pass


def _make_qm9_class():
    """Return a picklable QM9 subclass (imported lazily to keep torch_geometric optional)."""
    from torch_geometric.data import Data, download_url, extract_zip
    from torch_geometric.datasets import QM9
    from torch_geometric.io import fs

    class QM9ProcessedOnly(QM9):
        @property
        def raw_file_names(self):
            return ["qm9_v3.pt"]

        def download(self) -> None:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        def process(self) -> None:
            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, self.processed_paths[0])

    # Register at module level so pickle can find it by name.
    import uni_react.tasks.qm9.dataset as _mod
    _mod.QM9ProcessedOnly = QM9ProcessedOnly
    return QM9ProcessedOnly


def load_pyg_qm9(root: str, force_reload: bool = False):
    try:
        cls = _make_qm9_class()
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required for QM9 finetuning. Install PyG before running "
            "`uni_react/train_finetune_qm9.py`."
        ) from exc
    return cls(root=str(Path(root)), force_reload=force_reload)


def get_qm9_atomref(
    root: str,
    target: str,
    max_z: int = 100,
    force_reload: bool = False,
    target_index_variant: str = "default",
) -> Optional[torch.Tensor]:
    base_dataset = load_pyg_qm9(root=root, force_reload=force_reload)
    target_idx = get_qm9_target_index_map(target_index_variant)[target]
    atomref = base_dataset.atomref(target_idx)
    if atomref is None:
        return None
    if atomref.size(0) != max_z:
        tmp = torch.zeros(max_z, 1, dtype=atomref.dtype)
        idx = min(max_z, atomref.size(0))
        tmp[:idx] = atomref[:idx]
        return tmp
    return atomref


def build_qm9_split_indices(num_samples: int, split_mode: str) -> Dict[str, np.ndarray]:
    if split_mode not in QM9_SPLIT_MODES:
        raise ValueError(f"Unsupported split_mode '{split_mode}', choose from {QM9_SPLIT_MODES}")

    spec = QM9_SPLIT_SPECS[split_mode]
    n_train, n_valid, n_test = spec["sizes"]
    expected = n_train + n_valid + n_test
    if num_samples != expected:
        raise ValueError(
            f"PyG QM9 size mismatch for split '{split_mode}': got {num_samples}, expected {expected}. "
            "This code assumes the standard processed QM9 size."
        )

    rng = np.random.RandomState(spec["seed"])
    perm = rng.permutation(num_samples)
    train_end = n_train
    valid_end = n_train + n_valid
    return {
        "train": perm[:train_end].astype(np.int64),
        "valid": perm[train_end:valid_end].astype(np.int64),
        "test": perm[valid_end:].astype(np.int64),
    }


class QM9PyGDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices: np.ndarray,
        target: str = "gap",
        targets: Optional[Sequence[str]] = None,
        target_index_variant: str = "default",
        center_coords: bool = True,
        atom_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        resolved_targets = _resolve_targets(target, targets)

        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = tuple(resolved_targets)
        self.target = self.targets[0]
        self.target_index_variant = target_index_variant
        self.target_index_map = get_qm9_target_index_map(target_index_variant)
        self.target_indices = torch.as_tensor(
            [self.target_index_map[name] for name in self.targets],
            dtype=torch.long,
        )
        self.num_targets = len(self.targets)
        self.center_coords = bool(center_coords)
        self.atom_vocab_size = atom_vocab_size

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _read_item(self, idx: int):
        base_idx = int(self.indices[idx])
        return base_idx, self.base_dataset[base_idx]

    def _extract_targets(self, data) -> torch.Tensor:
        y = data.y.reshape(-1)
        return y.index_select(0, self.target_indices).to(dtype=torch.float32)

    def get_targets(self, idx: int) -> np.ndarray:
        _, data = self._read_item(idx)
        return self._extract_targets(data).detach().cpu().numpy().astype(np.float64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        base_idx, data = self._read_item(idx)
        atomic_numbers = data.z.detach().cpu().to(dtype=torch.long)
        coords = data.pos.detach().cpu().to(dtype=torch.float32)

        if self.atom_vocab_size is not None and int(atomic_numbers.max().item()) >= self.atom_vocab_size:
            raise ValueError(
                f"Atomic number out of vocab in sample {base_idx}: "
                f"max={int(atomic_numbers.max().item())}, atom_vocab_size={self.atom_vocab_size}"
            )

        if self.center_coords:
            coords = coords - coords.mean(dim=0, keepdim=True)

        sample = {
            "atomic_numbers": atomic_numbers,
            "coords": coords,
            "y": self._extract_targets(data),
        }
        return sample


def build_qm9_pyg_splits(
    root: str,
    target: str = "gap",
    targets: Optional[Sequence[str]] = None,
    split_mode: str = "egnn",
    target_index_variant: str = "default",
    center_coords: bool = True,
    atom_vocab_size: Optional[int] = None,
    force_reload: bool = False,
) -> Dict[str, QM9PyGDataset]:
    base_dataset = load_pyg_qm9(root=root, force_reload=force_reload)
    split_indices = build_qm9_split_indices(len(base_dataset), split_mode)
    return {
        split_name: QM9PyGDataset(
            base_dataset=base_dataset,
            indices=indices,
            target=target,
            targets=targets,
            target_index_variant=target_index_variant,
            center_coords=center_coords,
            atom_vocab_size=atom_vocab_size,
        )
        for split_name, indices in split_indices.items()
    }


def collate_fn_qm9(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_atoms = max(item["atomic_numbers"].shape[0] for item in batch)
    target_dim = int(batch[0]["y"].shape[0])

    atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    atom_padding = torch.ones((batch_size, max_atoms), dtype=torch.bool)
    y = torch.zeros((batch_size, target_dim), dtype=torch.float32)
    num_atoms = torch.zeros((batch_size,), dtype=torch.long)

    for i, item in enumerate(batch):
        n = int(item["atomic_numbers"].shape[0])
        num_atoms[i] = n
        atom_padding[i, :n] = False
        atomic_numbers[i, :n] = item["atomic_numbers"]
        coords[i, :n] = item["coords"]
        y[i] = item["y"]

    return {
        "atomic_numbers": atomic_numbers,
        "coords": coords,
        "atom_padding": atom_padding,
        "num_atoms": num_atoms,
        "y": y,
    }
