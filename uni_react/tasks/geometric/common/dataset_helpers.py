from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
from .dataset import H5SingleMolPretrainDataset

PathLike = Union[str, Path]


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
    file_limit: int = 0,
    **dataset_kwargs,
) -> H5SingleMolPretrainDataset:
    files = expand_h5_files(h5_files)
    if file_limit > 0:
        files = files[:file_limit]
    return H5SingleMolPretrainDataset(files, **dataset_kwargs)
