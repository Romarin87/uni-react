"""Smoke tests for density/reaction/QM9 training paths."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


class TinyDensityDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        n_atoms = 3
        n_points = 8
        return {
            "atomic_numbers": torch.tensor([1, 6, 8], dtype=torch.long),
            "coords": torch.randn(n_atoms, 3, dtype=torch.float32),
            "atom_padding": torch.zeros(n_atoms, dtype=torch.bool),
            "query_points": torch.randn(n_points, 3, dtype=torch.float32),
            "density_target": torch.randn(n_points, dtype=torch.float32),
            "total_charge": torch.tensor(0.0, dtype=torch.float32),
            "spin_multiplicity": torch.tensor(1.0, dtype=torch.float32),
        }


class TinyQM9Dataset(Dataset):
    def __init__(self, size: int = 6):
        self.size = size

    def __len__(self):
        return self.size

    def get_targets(self, idx: int):
        return np.asarray([float(idx)], dtype=np.float64)

    def __getitem__(self, idx):
        return {
            "atomic_numbers": torch.tensor([1, 6, 8], dtype=torch.long),
            "coords": torch.randn(3, 3, dtype=torch.float32),
            "y": torch.tensor([float(idx)], dtype=torch.float32),
        }


def _make_reaction_h5(path: Path, n_triplets: int = 3, n_atoms: int = 3) -> None:
    total_atoms = n_triplets * n_atoms
    offsets = np.arange(0, total_atoms + 1, n_atoms, dtype=np.int64)
    z = np.tile(np.asarray([1, 6, 8], dtype=np.int64), n_triplets)
    coords = np.random.default_rng(0).normal(size=(total_atoms, 3)).astype(np.float32)
    comp_hash = np.arange(n_triplets, dtype=np.int64)

    with h5py.File(path, "w") as h5:
        g = h5.create_group("triplets")
        g.create_dataset("offsets", data=offsets)
        g.create_dataset("n_atoms", data=np.full((n_triplets,), n_atoms, dtype=np.int32))
        g.create_dataset("comp_hash", data=comp_hash)
        for prefix in ("r", "ts", "p"):
            g.create_dataset(f"{prefix}_Z", data=z)
            g.create_dataset(f"{prefix}_R", data=coords)


def test_density_trainer_smoke(tmp_path):
    from uni_react.configs import DensityPretrainConfig
    from uni_react.encoders import DensityPretrainNet
    from uni_react.schedulers.cosine import WarmupCosineScheduler
    from uni_react.trainers import DensityPretrainTrainer

    cfg = DensityPretrainConfig(
        batch_size=2,
        num_workers=0,
        epochs=2,
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_kernel=16,
        point_hidden_dim=16,
        cond_hidden_dim=8,
        head_hidden_dim=32,
        out_dir=str(tmp_path / "density"),
    )
    train_loader = DataLoader(TinyDensityDataset(), batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(TinyDensityDataset(), batch_size=cfg.batch_size, shuffle=False)

    model = DensityPretrainNet(
        encoder_type=cfg.encoder_type,
        emb_dim=cfg.emb_dim,
        inv_layer=cfg.inv_layer,
        se3_layer=cfg.se3_layer,
        heads=cfg.heads,
        atom_vocab_size=cfg.atom_vocab_size,
        cutoff=cfg.cutoff,
        num_kernel=cfg.num_kernel,
        point_hidden_dim=cfg.point_hidden_dim,
        cond_hidden_dim=cfg.cond_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        radial_sigma=cfg.radial_sigma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=0, total_steps=1)
    trainer = DensityPretrainTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        scheduler=scheduler,
        distributed=False,
        device=torch.device("cpu"),
    )

    train_metrics = trainer.train_epoch(1)
    val_metrics = trainer.eval_epoch(1)
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert torch.isfinite(torch.tensor(val_metrics["loss"]))
    assert scheduler.state_dict()["total_steps"] == cfg.epochs * len(train_loader)
    assert scheduler.state_dict()["step_count"] > 0

    trainer.save_checkpoint(epoch=1, tag="density_smoke")
    new_model = DensityPretrainNet(
        encoder_type=cfg.encoder_type,
        emb_dim=cfg.emb_dim,
        inv_layer=cfg.inv_layer,
        se3_layer=cfg.se3_layer,
        heads=cfg.heads,
        atom_vocab_size=cfg.atom_vocab_size,
        cutoff=cfg.cutoff,
        num_kernel=cfg.num_kernel,
        point_hidden_dim=cfg.point_hidden_dim,
        cond_hidden_dim=cfg.cond_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        radial_sigma=cfg.radial_sigma,
    )
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = WarmupCosineScheduler(new_optimizer, warmup_steps=0, total_steps=1)
    resumed = DensityPretrainTrainer(
        model=new_model,
        optimizer=new_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        scheduler=new_scheduler,
        distributed=False,
        device=torch.device("cpu"),
    )
    assert resumed.load_checkpoint(str(Path(cfg.out_dir) / "density_smoke.pt")) == 2


@pytest.mark.parametrize("encoder_type", ["single_mol", "reacformer_se3", "reacformer_so2"])
def test_reaction_trainer_smoke(tmp_path, encoder_type):
    from uni_react.configs import ReactionPretrainConfig
    from uni_react.encoders.reaction_model import ReactionPretrainNet
    from uni_react.training.pretrain_builders import build_pretrain_encoder
    from uni_react.trainers.pretrain_reaction import ReactionPretrainTrainer

    h5_path = tmp_path / "reaction.h5"
    _make_reaction_h5(h5_path)

    cfg = ReactionPretrainConfig(
        train_h5=str(h5_path),
        val_h5=str(h5_path),
        batch_size=2,
        num_workers=0,
        epochs=2,
        encoder_type=encoder_type,
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_kernel=16,
        head_hidden_dim=32,
        out_dir=str(tmp_path / f"reaction_{encoder_type}"),
    )
    model = ReactionPretrainNet(
        descriptor=build_pretrain_encoder(cfg),
        emb_dim=cfg.emb_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        teacher_momentum=cfg.teacher_momentum,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ReactionPretrainTrainer(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        distributed=False,
        device=torch.device("cpu"),
    )

    train_metrics = trainer.train_epoch(1)
    val_metrics = trainer.eval_epoch(1)
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert torch.isfinite(torch.tensor(val_metrics["loss"]))

    trainer.save_checkpoint(epoch=1, tag="reaction_smoke")
    resumed_model = ReactionPretrainNet(
        descriptor=build_pretrain_encoder(cfg),
        emb_dim=cfg.emb_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        teacher_momentum=cfg.teacher_momentum,
    )
    resumed = ReactionPretrainTrainer(
        model=resumed_model,
        cfg=cfg,
        optimizer=torch.optim.AdamW(resumed_model.parameters(), lr=1e-3),
        distributed=False,
        device=torch.device("cpu"),
    )
    assert resumed.load_checkpoint(str(Path(cfg.out_dir) / "reaction_smoke.pt")) == 2


@pytest.mark.parametrize("encoder_type", ["single_mol", "reacformer_se3", "reacformer_so2"])
def test_qm9_trainer_smoke(monkeypatch, tmp_path, encoder_type):
    from uni_react.configs import FinetuneQM9Config
    from uni_react.encoders import QM9FineTuneNet
    from uni_react.training.pretrain_builders import build_pretrain_encoder
    from uni_react.schedulers.cosine import WarmupCosineScheduler
    from uni_react.trainers.finetune_qm9 import FinetuneQM9Trainer

    tiny_splits = {
        "train": TinyQM9Dataset(size=6),
        "valid": TinyQM9Dataset(size=4),
        "test": TinyQM9Dataset(size=4),
    }
    monkeypatch.setattr(
        "uni_react.trainers.finetune_qm9.build_qm9_pyg_splits",
        lambda **kwargs: tiny_splits,
    )

    cfg = FinetuneQM9Config(
        batch_size=2,
        num_workers=0,
        epochs=2,
        encoder_type=encoder_type,
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_kernel=16,
        head_hidden_dim=32,
        out_dir=str(tmp_path / "qm9"),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=0, total_steps=1)
    trainer = FinetuneQM9Trainer(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        targets=["gap"],
        scheduler=scheduler,
        distributed=False,
        device=torch.device("cpu"),
    )

    train_metrics = trainer.train_epoch(1)
    val_metrics = trainer.eval_epoch(1)
    test_metrics = trainer.eval_test()
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert torch.isfinite(torch.tensor(val_metrics["mae"]))
    assert torch.isfinite(torch.tensor(test_metrics["loss"]))
    assert scheduler.state_dict()["total_steps"] == cfg.epochs * len(trainer._train_loader)

    trainer.save_checkpoint(epoch=1, tag="qm9_smoke")
    resumed_model = QM9FineTuneNet(
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
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = WarmupCosineScheduler(resumed_optimizer, warmup_steps=0, total_steps=1)
    resumed = FinetuneQM9Trainer(
        model=resumed_model,
        cfg=cfg,
        optimizer=resumed_optimizer,
        targets=["gap"],
        scheduler=resumed_scheduler,
        distributed=False,
        device=torch.device("cpu"),
    )
    assert resumed.load_checkpoint(str(Path(cfg.out_dir) / "qm9_smoke.pt")) == 2
