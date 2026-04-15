#!/usr/bin/env python3
"""Convert a legacy geometric pretraining checkpoint to the current format.

The current code already supports many legacy key remappings inside
``SingleMolPretrainNet.load_state_dict``. This script makes that compatibility
explicit by:

1. Loading the old checkpoint
2. Building the current model
3. Loading legacy weights through the model's remapping logic
4. Saving a new checkpoint in the current layout

Usage:
    python scripts/convert_legacy_pretrain_ckpt.py \
      --input pretrain_v2.pt \
      --output pretrain_v2_converted.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from uni_react.encoders import SingleMolPretrainNet


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert legacy pretrain checkpoint")
    p.add_argument("--input", required=True, help="Path to legacy checkpoint")
    p.add_argument("--output", required=True, help="Path to converted checkpoint")
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--inv-layer", type=int, default=2)
    p.add_argument("--se3-layer", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--atom-vocab-size", type=int, default=128)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--num-kernel", type=int, default=128)
    return p


def main() -> None:
    args = build_parser().parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    file_size_gb = in_path.stat().st_size / (1024 ** 3)
    print(f"[1/4] Input checkpoint: {in_path}")
    print(f"      Size: {file_size_gb:.3f} GB")

    t0 = time.time()
    print("[2/4] Loading legacy checkpoint with torch.load(...) ...")
    ckpt = torch.load(in_path, map_location="cpu")
    print(f"      Done in {time.time() - t0:.2f}s")

    t1 = time.time()
    print("[3/4] Building current model and remapping state_dict ...")
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError("Unsupported checkpoint format: expected dict or dict['model']")

    print(f"      Top-level object type: {type(ckpt).__name__}")
    print(f"      Model/state_dict keys: {len(state_dict)}")

    # Building the current model may trigger TorchScript compilation for some
    # helper layers (e.g. RBFEmb). That is unnecessary for checkpoint
    # conversion and can fail on some local environments, so temporarily turn
    # torch.jit.script into a no-op while constructing the model.
    orig_script = torch.jit.script
    try:
        torch.jit.script = lambda obj, *a, **k: obj  # type: ignore[assignment]
        model = SingleMolPretrainNet(
            emb_dim=args.emb_dim,
            inv_layer=args.inv_layer,
            se3_layer=args.se3_layer,
            heads=args.heads,
            atom_vocab_size=args.atom_vocab_size,
            cutoff=args.cutoff,
            num_kernel=args.num_kernel,
            enable_electronic_structure_task=False,
        )
    finally:
        torch.jit.script = orig_script  # type: ignore[assignment]

    # First remap legacy keys through the model helper.
    remapped = {}
    for key, value in state_dict.items():
        new_key = model._remap_legacy_key(key)
        if new_key in remapped and new_key != key:
            continue
        remapped[new_key] = value

    # Drop keys whose tensor shapes no longer match in the current model.
    current_state = model.state_dict()
    skipped_shape_mismatch = []
    filtered = {}
    for key, value in remapped.items():
        if key in current_state and hasattr(value, "shape") and hasattr(current_state[key], "shape"):
            if tuple(value.shape) != tuple(current_state[key].shape):
                skipped_shape_mismatch.append(
                    (key, tuple(value.shape), tuple(current_state[key].shape))
                )
                continue
        filtered[key] = value

    incompatible = model.load_state_dict(filtered, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f"      Remap/load done in {time.time() - t1:.2f}s")
    print(f"      Missing keys: {len(missing)}")
    print(f"      Unexpected keys: {len(unexpected)}")

    converted = {
        "epoch": ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0,
        "model": model.state_dict(),
        "best_val": ckpt.get("best_val", float("inf")) if isinstance(ckpt, dict) else float("inf"),
        "global_step": ckpt.get("global_step", 0) if isinstance(ckpt, dict) else 0,
        "converted_from": str(in_path),
        "missing_keys_during_conversion": list(missing),
        "unexpected_keys_during_conversion": list(unexpected),
        "skipped_shape_mismatch": [
            {"key": k, "old_shape": old, "new_shape": new}
            for k, old, new in skipped_shape_mismatch
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.time()
    print(f"[4/4] Saving converted checkpoint to: {out_path}")
    torch.save(converted, out_path)
    print(f"      Save done in {time.time() - t2:.2f}s")

    print(f"Converted checkpoint saved to: {out_path}")
    print(f"Missing keys during conversion: {len(missing)}")
    if missing:
        for k in missing[:20]:
            print(f"  MISSING: {k}")
    print(f"Unexpected keys during conversion: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:20]:
            print(f"  UNEXPECTED: {k}")
    print(f"Shape-mismatch keys skipped: {len(skipped_shape_mismatch)}")
    if skipped_shape_mismatch:
        for k, old, new in skipped_shape_mismatch[:20]:
            print(f"  SHAPE_MISMATCH: {k} old={old} new={new}")

    total = time.time() - t0
    print(f"[done] Total conversion time: {total:.2f}s")


if __name__ == "__main__":
    main()
