#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inspect an HDF5 file structure and summarize datasets without loading data."""

import argparse
from typing import Any

import h5py


def _fmt_bytes(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(nbytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PB"


def _print_attrs(name: str, attrs: h5py.AttributeManager, indent: str) -> None:
    if len(attrs) == 0:
        return
    print(f"{indent}{name} attrs:")
    for k in attrs.keys():
        v = attrs[k]
        print(f"{indent}  - {k}: {v}")


def _describe_dataset(path: str, dset: h5py.Dataset, indent: str) -> None:
    shape = dset.shape
    dtype = dset.dtype
    chunks = dset.chunks
    compression = dset.compression
    nbytes = dset.size * dtype.itemsize

    print(f"{indent}{path}")
    print(f"{indent}  shape: {shape}")
    print(f"{indent}  dtype: {dtype}")
    print(f"{indent}  size: {_fmt_bytes(nbytes)}")
    print(f"{indent}  chunks: {chunks}")
    print(f"{indent}  compression: {compression}")


def _walk(name: str, obj: Any) -> None:
    if isinstance(obj, h5py.Group):
        print(f"{name}/")
        _print_attrs("group", obj.attrs, "  ")
    elif isinstance(obj, h5py.Dataset):
        _describe_dataset(name, obj, "  ")
        _print_attrs("dataset", obj.attrs, "  ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect HDF5 file structure")
    ap.add_argument("path", help="Path to .h5 file")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"File: {args.path}")
        _print_attrs("file", f.attrs, "")
        print("\nTree:")
        f.visititems(_walk)


if __name__ == "__main__":
    main()
