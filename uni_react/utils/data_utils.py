from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader

from .dataset import H5SingleMolPretrainDataset, collate_fn_pretrain
from .types import PathLike  # noqa: F401 – re-exported for backward compat


def expand_h5_files(paths: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    """
    Expand file/dir/glob inputs into a sorted list of HDF5 file paths.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out: List[str] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(str(x) for x in sorted(p.glob("*.h5")))
            continue

        # Treat wildcard patterns as globs.
        if any(ch in str(p) for ch in "*?[]"):
            out.extend(str(x) for x in sorted(p.parent.glob(p.name)))
            continue

        if p.exists() and p.is_file():
            out.append(str(p))

    out = sorted(set(out))
    if not out:
        raise ValueError("No .h5 files found from provided paths")
    return out


def split_h5_files(
    h5_files: Sequence[PathLike],
    val_ratio: float = 0.02,
    test_ratio: float = 0.02,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """
    File-level split. Prefer shard-level split for very large datasets.
    """
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Require val_ratio >= 0, test_ratio >= 0 and val_ratio + test_ratio < 1")

    files = expand_h5_files(h5_files)
    n = len(files)
    if n < 3:
        return {"train": files, "val": [], "test": []}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    return {
        "train": [files[i] for i in train_idx],
        "val": [files[i] for i in val_idx],
        "test": [files[i] for i in test_idx],
    }


def build_pretrain_dataset(
    h5_files: Sequence[PathLike],
    **dataset_kwargs,
) -> H5SingleMolPretrainDataset:
    files = expand_h5_files(h5_files)
    return H5SingleMolPretrainDataset(files, **dataset_kwargs)


def build_pretrain_dataloader(
    h5_files: Sequence[PathLike],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
    dataset_kwargs: Optional[Dict] = None,
) -> DataLoader:
    dataset_kwargs = dataset_kwargs or {}
    dataset = build_pretrain_dataset(h5_files, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=drop_last,
        collate_fn=collate_fn_pretrain,
    )


def build_pretrain_dataloaders(
    train_h5_files: Sequence[PathLike],
    val_h5_files: Optional[Sequence[PathLike]] = None,
    test_h5_files: Optional[Sequence[PathLike]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last_train: bool = True,
    dataset_kwargs: Optional[Dict] = None,
) -> Dict[str, DataLoader]:
    dataset_kwargs = dataset_kwargs or {}

    loaders = {
        "train": build_pretrain_dataloader(
            train_h5_files,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last_train,
            dataset_kwargs=dataset_kwargs,
        )
    }

    if val_h5_files:
        loaders["val"] = build_pretrain_dataloader(
            val_h5_files,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
            dataset_kwargs=dataset_kwargs,
        )

    if test_h5_files:
        loaders["test"] = build_pretrain_dataloader(
            test_h5_files,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
            dataset_kwargs=dataset_kwargs,
        )

    return loaders
