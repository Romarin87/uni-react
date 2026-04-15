#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Check integrity of .tar.gz archives by streaming and fully reading contents.

This detects:
1) archive open/read errors (truncated/corrupted gzip stream),
2) member-level extraction/read errors.
"""

from __future__ import annotations

import argparse
import glob
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence


def _expand_tar_paths(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if not p.endswith(".tar.gz"):
                continue
            if os.path.isfile(p) and p not in seen:
                out.append(p)
                seen.add(p)
    if not out:
        raise FileNotFoundError(f"No .tar.gz files matched patterns: {patterns}")
    return out


def _check_one_tar(path: str, chunk_size: int = 4 * 1024 * 1024) -> Dict[str, object]:
    t0 = time.time()
    members = 0
    files = 0
    total_bytes = 0
    try:
        # Stream mode: never extracts to disk, validates decompression sequentially.
        with tarfile.open(path, mode="r|gz") as tf:
            for member in tf:
                members += 1
                if not member.isfile():
                    continue
                files += 1
                fobj = tf.extractfile(member)
                if fobj is None:
                    raise RuntimeError(f"extractfile returned None for file member: {member.name}")
                while True:
                    chunk = fobj.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
        return {
            "path": path,
            "ok": True,
            "error": "",
            "members": members,
            "files": files,
            "bytes": total_bytes,
            "seconds": time.time() - t0,
        }
    except Exception as exc:
        return {
            "path": path,
            "ok": False,
            "error": repr(exc),
            "members": members,
            "files": files,
            "bytes": total_bytes,
            "seconds": time.time() - t0,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check integrity of tar.gz archives and report bad files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tar_glob",
        nargs="+",
        required=True,
        help="One or more glob patterns for .tar.gz files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes.",
    )
    p.add_argument(
        "--chunk_size_mb",
        type=int,
        default=4,
        help="Chunk size (MB) when reading each member stream.",
    )
    p.add_argument(
        "--bad_list",
        type=str,
        default="",
        help="Optional path to write bad tar paths (one per line).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tar_paths = _expand_tar_paths(args.tar_glob)
    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_size_mb)) * 1024 * 1024

    print(f"[config] tars={len(tar_paths)} workers={workers} chunk_size={chunk_size // (1024 * 1024)}MB")

    bad: List[Dict[str, object]] = []
    good = 0
    t0 = time.time()

    if workers == 1:
        for i, path in enumerate(tar_paths, start=1):
            result = _check_one_tar(path, chunk_size)
            ok = bool(result["ok"])
            secs = float(result["seconds"])
            files = int(result["files"])
            members = int(result["members"])
            read_gb = float(result["bytes"]) / (1024 ** 3)
            if ok:
                good += 1
                print(f"[ok]   {path} | members={members} files={files} read={read_gb:.2f}GB time={secs:.1f}s")
            else:
                bad.append(result)
                print(f"[bad]  {path} | members={members} files={files} read={read_gb:.2f}GB time={secs:.1f}s")
                print(f"       error: {result['error']}")
            if i % 20 == 0 or i == len(tar_paths):
                print(f"[progress] checked={i}/{len(tar_paths)} good={good} bad={len(bad)}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_check_one_tar, p, chunk_size) for p in tar_paths]
            for i, fut in enumerate(as_completed(futs), start=1):
                result = fut.result()
                path = str(result["path"])
                ok = bool(result["ok"])
                secs = float(result["seconds"])
                files = int(result["files"])
                members = int(result["members"])
                read_gb = float(result["bytes"]) / (1024 ** 3)
                if ok:
                    good += 1
                    print(f"[ok]   {path} | members={members} files={files} read={read_gb:.2f}GB time={secs:.1f}s")
                else:
                    bad.append(result)
                    print(f"[bad]  {path} | members={members} files={files} read={read_gb:.2f}GB time={secs:.1f}s")
                    print(f"       error: {result['error']}")
                if i % 20 == 0 or i == len(tar_paths):
                    print(f"[progress] checked={i}/{len(tar_paths)} good={good} bad={len(bad)}")

    elapsed = time.time() - t0
    print(f"[done] total={len(tar_paths)} good={good} bad={len(bad)} elapsed={elapsed / 60.0:.1f} min")

    if bad and args.bad_list:
        bad_path = Path(args.bad_list)
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        with bad_path.open("w", encoding="utf-8") as f:
            for item in bad:
                f.write(str(item["path"]) + "\n")
        print(f"[write] bad list -> {bad_path}")


if __name__ == "__main__":
    main()
