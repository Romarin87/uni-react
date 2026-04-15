#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


from ..encoders import SingleMolPretrainNet


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"], True
    if isinstance(payload, dict):
        return payload, False
    raise TypeError("Unsupported checkpoint type. Expect dict or dict with key 'model'.")


def _remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    out: Dict[str, torch.Tensor] = {}
    changed = 0
    for key, value in state_dict.items():
        new_key = SingleMolPretrainNet._remap_legacy_key(key)
        if new_key != key:
            changed += 1
        # Prefer already-new keys when both old/new map to same name.
        if new_key in out and new_key != key:
            continue
        out[new_key] = value
    return out, changed


def _normalize_args(raw_args: Dict) -> Dict:
    if not isinstance(raw_args, dict):
        return raw_args

    out = dict(raw_args)

    key_renames = {
        "reactivity_global_weight": "vip_vea_weight",
        "reactivity_atom_weight": "fukui_weight",
        "reactivity_global_keys": "vip_vea_keys",
        "reactivity_atom_keys": "fukui_keys",
        "enable_reactivity_task": "enable_electronic_structure_task",
        "reactivity_vip_vea_dim": "electronic_structure_vip_vea_dim",
        "reactivity_fukui_dim": "electronic_structure_fukui_dim",
    }
    for old_key, new_key in key_renames.items():
        if old_key in out and new_key not in out:
            out[new_key] = out[old_key]

    train_mode = out.get("train_mode", None)
    if isinstance(train_mode, str):
        mode = train_mode.strip().lower()
        # "stage1" / "stage2" kept as legacy aliases for old checkpoints.
        if mode in {"geometric", "geometric_structure", "geom", "stage1"}:
            out["train_mode"] = "geometric_structure"
        elif mode in {"electronic", "electronic_structure", "reactivity", "stage2"}:
            out["train_mode"] = "electronic_structure"

    return out


def _infer_train_mode_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    electronic_prefixes = (
        "tasks.electronic_structure.",
        "tasks.reactivity.",
        "vip_vea_head.",
        "fukui_head.",
        "reactivity_global_head.",
        "reactivity_atom_head.",
    )
    for key in state_dict.keys():
        if key.startswith(electronic_prefixes):
            return "electronic_structure"
    return "geometric_structure"


def _resolve_train_mode(args_dict: Dict, remapped_state_dict: Dict[str, torch.Tensor]) -> str:
    train_mode = args_dict.get("train_mode", None)
    if train_mode in {"geometric_structure", "electronic_structure"}:
        return train_mode
    if bool(args_dict.get("enable_electronic_structure_task", False)):
        return "electronic_structure"
    return _infer_train_mode_from_state_dict(remapped_state_dict)


def _build_model_from_args(
    args_dict: Dict,
    remapped_state_dict: Dict[str, torch.Tensor],
    forced_train_mode: Optional[str] = None,
) -> SingleMolPretrainNet:
    args_dict = args_dict or {}
    train_mode = forced_train_mode or _resolve_train_mode(args_dict, remapped_state_dict)

    vip_vea_keys = args_dict.get("vip_vea_keys", ["vip", "vea"])
    fukui_keys = args_dict.get("fukui_keys", ["f_plus", "f_minus", "f_zero"])

    return SingleMolPretrainNet(
        emb_dim=int(args_dict.get("emb_dim", 256)),
        inv_layer=int(args_dict.get("inv_layer", 2)),
        se3_layer=int(args_dict.get("se3_layer", 4)),
        heads=int(args_dict.get("heads", 8)),
        atom_vocab_size=int(args_dict.get("atom_vocab_size", 128)),
        cutoff=float(args_dict.get("cutoff", 5.0)),
        path_dropout=float(args_dict.get("path_dropout", 0.1)),
        activation_dropout=float(args_dict.get("activation_dropout", 0.1)),
        attn_dropout=float(args_dict.get("attn_dropout", 0.1)),
        num_kernel=int(args_dict.get("num_kernel", 128)),
        enable_electronic_structure_task=(train_mode == "electronic_structure"),
        electronic_structure_vip_vea_dim=int(
            args_dict.get("electronic_structure_vip_vea_dim", len(vip_vea_keys))
        ),
        electronic_structure_fukui_dim=int(
            args_dict.get("electronic_structure_fukui_dim", len(fukui_keys))
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert old pretrain checkpoint keys/config to current uni_react format."
    )
    parser.add_argument("--in_ckpt", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--out_ckpt", type=str, required=True, help="Output checkpoint path")
    parser.add_argument(
        "--drop_optimizer",
        action="store_true",
        help="Drop optimizer state in output checkpoint",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip loading converted state into current model for validation",
    )
    args = parser.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_ckpt)
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(str(in_path), map_location="cpu")
    state_dict, has_wrapper = _extract_state_dict(payload)
    new_state_dict, changed = _remap_state_dict(state_dict)

    if has_wrapper:
        new_payload = dict(payload)
        new_payload["model"] = new_state_dict
        if isinstance(new_payload.get("args", None), dict):
            new_payload["args"] = _normalize_args(new_payload["args"])
            # Extremely old checkpoints may not have train_mode at all.
            if "train_mode" not in new_payload["args"]:
                new_payload["args"]["train_mode"] = _resolve_train_mode(new_payload["args"], new_state_dict)
        if args.drop_optimizer and "optimizer" in new_payload:
            new_payload["optimizer"] = None
    else:
        new_payload = new_state_dict

    if not args.no_validate:
        cfg = {}
        train_mode = None
        if has_wrapper and isinstance(new_payload.get("args", None), dict):
            cfg = new_payload["args"]
            train_mode = cfg.get("train_mode", None)
        model = _build_model_from_args(
            args_dict=cfg,
            remapped_state_dict=new_state_dict,
            forced_train_mode=train_mode,
        )
        load_ret = model.load_state_dict(new_state_dict, strict=False)
        missing = len(getattr(load_ret, "missing_keys", []))
        unexpected = len(getattr(load_ret, "unexpected_keys", []))
        print(f"[validate] missing_keys={missing}, unexpected_keys={unexpected}")

    if has_wrapper and isinstance(new_payload.get("args", None), dict):
        print(f"[info] resolved train_mode={new_payload['args'].get('train_mode')}")

    torch.save(new_payload, str(out_path))
    print(f"[done] converted checkpoint saved to: {out_path}")
    print(f"[done] remapped keys: {changed}")


if __name__ == "__main__":
    main()
