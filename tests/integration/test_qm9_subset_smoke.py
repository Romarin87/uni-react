"""Smoke test for QM9 fine-tuning using a tiny subset of real QM9 data.

This test reuses an already-downloaded PyG QM9 dataset under
``QM9_SMOKE_ROOT`` (or ``/tmp/uni_react_train_smoke/qm9_data`` by default),
selects a tiny deterministic subset, and runs one epoch of training plus test
evaluation. It skips cleanly when the local QM9 data is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch


_DEFAULT_QM9_ROOT = Path("/tmp/uni_react_train_smoke/qm9_data")


@pytest.fixture(scope="module")
def qm9_root() -> str:
    root = Path(os.environ.get("QM9_SMOKE_ROOT", str(_DEFAULT_QM9_ROOT)))
    processed = root / "processed" / "data_v3.pt"
    if not processed.exists():
        pytest.skip(
            f"Local QM9 data not found at {processed}. "
            "Download QM9 first or set QM9_SMOKE_ROOT."
        )
    return str(root)


@pytest.mark.parametrize("encoder_type", ["single_mol", "reacformer_se3", "reacformer_so2"])
def test_qm9_subset_trainer_smoke(monkeypatch, tmp_path, qm9_root, encoder_type):
    from uni_react.configs import FinetuneQM9Config
    from uni_react.encoders import QM9FineTuneNet
    from uni_react.training.pretrain_builders import build_pretrain_encoder
    from uni_react.trainers.finetune_qm9 import FinetuneQM9Trainer
    from uni_react.utils.qm9_dataset import QM9PyGDataset, load_pyg_qm9

    base = load_pyg_qm9(root=qm9_root, force_reload=False)
    splits = {
        "train": QM9PyGDataset(
            base, np.arange(32, dtype=np.int64),
            target="gap", center_coords=True, atom_vocab_size=16,
        ),
        "valid": QM9PyGDataset(
            base, np.arange(32, 40, dtype=np.int64),
            target="gap", center_coords=True, atom_vocab_size=16,
        ),
        "test": QM9PyGDataset(
            base, np.arange(40, 48, dtype=np.int64),
            target="gap", center_coords=True, atom_vocab_size=16,
        ),
    }
    monkeypatch.setattr(
        "uni_react.trainers.finetune_qm9.build_qm9_pyg_splits",
        lambda **kwargs: splits,
    )

    cfg = FinetuneQM9Config(
        data_root=qm9_root,
        device="cpu",
        epochs=1,
        batch_size=8,
        num_workers=0,
        encoder_type=encoder_type,
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_kernel=16,
        head_hidden_dim=32,
        out_dir=str(tmp_path / f"qm9_subset_smoke_{encoder_type}"),
        log_interval=1,
        save_every=1,
    )

    model = QM9FineTuneNet(
        emb_dim=cfg.emb_dim,
        inv_layer=cfg.inv_layer,
        se3_layer=cfg.se3_layer,
        heads=cfg.heads,
        atom_vocab_size=cfg.atom_vocab_size,
        cutoff=cfg.cutoff,
        num_kernel=cfg.num_kernel,
        head_hidden_dim=cfg.head_hidden_dim,
        num_targets=1,
        descriptor=build_pretrain_encoder(cfg),
    )
    backbone_params = list(model.descriptor.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("descriptor.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": cfg.head_lr, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
    )

    trainer = FinetuneQM9Trainer(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        targets=["gap"],
        scheduler=None,
        distributed=False,
        device=torch.device("cpu"),
    )
    trainer.fit(start_epoch=1)
    test_metrics = trainer.eval_test()

    assert torch.isfinite(torch.tensor(test_metrics["loss"]))
    assert torch.isfinite(torch.tensor(test_metrics["mae"]))
    assert (Path(cfg.out_dir) / "latest.pt").exists()
