"""Unit tests for trainer utilities."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TestBaseTrainer:
    """Tests for BaseTrainer class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 10)
    
    @pytest.fixture
    def simple_optimizer(self, simple_model):
        """Create a simple optimizer for testing."""
        return torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    
    @pytest.fixture
    def trainer(self, simple_model, simple_optimizer, tmp_path):
        """Create a BaseTrainer instance for testing."""
        from uni_react.training import BaseTrainer
        
        return BaseTrainer(
            model=simple_model,
            optimizer=simple_optimizer,
            out_dir=str(tmp_path),
            epochs=10,
            save_every=2,
            distributed=False,
            rank=0,
            world_size=1,
        )
    
    def test_trainer_initialization(self, trainer, simple_model):
        """Test that trainer initializes correctly."""
        assert trainer.raw_model is simple_model
        assert trainer.model is simple_model  # Not wrapped in DDP
        assert trainer.epochs == 10
        assert trainer.save_every == 2
        assert trainer.distributed is False
        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert trainer.best_val == float('inf')
        assert trainer.global_step == 0
    
    def test_trainer_out_dir_created(self, trainer):
        """Test that output directory is created."""
        assert trainer.out_dir.exists()
        assert trainer.out_dir.is_dir()
    
    def test_reduce_metrics_single_process(self, trainer):
        """Test metric reduction in single process mode."""
        meters = {"loss": 10.0, "acc": 0.8, "mae": 2.5}
        steps = 5
        
        result = trainer.reduce_metrics(meters, steps)
        
        assert result["loss"] == pytest.approx(2.0)  # 10.0 / 5
        assert result["acc"] == pytest.approx(0.16)  # 0.8 / 5
        assert result["mae"] == pytest.approx(0.5)   # 2.5 / 5
    
    def test_reduce_metrics_zero_steps(self, trainer):
        """Test metric reduction with zero steps."""
        meters = {"loss": 10.0, "acc": 0.8}
        steps = 0
        
        result = trainer.reduce_metrics(meters, steps)
        
        # Should return original meters when steps is 0
        assert result == meters
    
    def test_reduce_metrics_empty(self, trainer):
        """Test metric reduction with empty meters."""
        meters = {}
        steps = 5
        
        result = trainer.reduce_metrics(meters, steps)
        
        assert result == {}
    
    def test_save_checkpoint_creates_file(self, trainer):
        """Test that save_checkpoint creates a file."""
        trainer.save_checkpoint(epoch=1, tag="test")
        
        ckpt_path = trainer.out_dir / "test.pt"
        assert ckpt_path.exists()
    
    def test_save_checkpoint_content(self, trainer):
        """Test that checkpoint contains expected keys."""
        trainer.best_val = 0.5
        trainer.global_step = 100
        trainer.save_checkpoint(epoch=5, tag="test")
        
        ckpt_path = trainer.out_dir / "test.pt"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        assert "epoch" in ckpt
        assert "model" in ckpt
        assert "optimizer" in ckpt
        assert "best_val" in ckpt
        assert "global_step" in ckpt
        assert "world_size" in ckpt
        assert "time" in ckpt
        
        assert ckpt["epoch"] == 5
        assert ckpt["best_val"] == 0.5
        assert ckpt["global_step"] == 100
        assert ckpt["world_size"] == 1
    
    def test_save_checkpoint_without_optimizer(self, trainer):
        """Test saving checkpoint without optimizer state."""
        trainer.save_checkpoint(epoch=1, tag="test", include_optimizer=False)
        
        ckpt_path = trainer.out_dir / "test.pt"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        assert "model" in ckpt
        assert "optimizer" not in ckpt

    def test_save_checkpoint_respects_default_save_optimizer_flag(self, tmp_path):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        from uni_react.training import BaseTrainer
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            out_dir=str(tmp_path),
            distributed=False,
            save_optimizer=False,
        )

        trainer.save_checkpoint(epoch=1, tag="no_opt")
        ckpt = torch.load(tmp_path / "no_opt.pt", map_location="cpu")
        assert "optimizer" not in ckpt
    
    def test_load_checkpoint_restores_state(self, trainer):
        """Test that load_checkpoint restores model and optimizer state."""
        # Save initial state
        trainer.best_val = 0.5
        trainer.global_step = 100
        trainer.save_checkpoint(epoch=5, tag="test")
        
        # Create new trainer with different model
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        
        from uni_react.training import BaseTrainer
        new_trainer = BaseTrainer(
            model=new_model,
            optimizer=new_optimizer,
            out_dir=str(trainer.out_dir),
            distributed=False,
        )
        
        # Load checkpoint
        ckpt_path = trainer.out_dir / "test.pt"
        epoch = new_trainer.load_checkpoint(str(ckpt_path))
        
        # Verify state restored
        assert epoch == 6  # Should resume from next epoch
        assert new_trainer.best_val == pytest.approx(0.5)
        assert new_trainer.global_step == 100

    def test_load_checkpoint_resumes_same_epoch_for_mid_epoch_checkpoint(self, tmp_path):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        from uni_react.training import BaseTrainer
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            out_dir=str(tmp_path),
            distributed=False,
        )
        trainer.save_checkpoint(epoch=5, tag="step_ckpt", step_in_epoch=7)

        resumed_model = nn.Linear(10, 10)
        resumed = BaseTrainer(
            model=resumed_model,
            optimizer=torch.optim.Adam(resumed_model.parameters(), lr=1e-3),
            out_dir=str(tmp_path),
            distributed=False,
        )

        start_epoch = resumed.load_checkpoint(str(tmp_path / "step_ckpt.pt"))
        assert start_epoch == 5
        assert resumed.resume_step_in_epoch == 7

    def test_reaction_trainer_respects_save_optimizer_flag(self, tmp_path):
        from uni_react.configs import ReactionConfig
        from uni_react.tasks.reaction.common import ReactionPretrainNet
        from uni_react.models import build_model_spec
        from uni_react.tasks.reaction.common.trainer import ReactionPretrainTrainer

        h5_path = tmp_path / "reaction.h5"
        import h5py
        import numpy as np

        total_atoms = 6
        offsets = np.asarray([0, 3, 6], dtype=np.int64)
        z = np.asarray([1, 6, 8, 1, 6, 8], dtype=np.int64)
        coords = np.zeros((total_atoms, 3), dtype=np.float32)
        with h5py.File(h5_path, "w") as h5:
            g = h5.create_group("triplets")
            g.create_dataset("offsets", data=offsets)
            g.create_dataset("n_atoms", data=np.asarray([3, 3], dtype=np.int32))
            g.create_dataset("comp_hash", data=np.asarray([0, 1], dtype=np.int64))
            for prefix in ("r", "ts", "p"):
                g.create_dataset(f"{prefix}_Z", data=z)
                g.create_dataset(f"{prefix}_R", data=coords)

        cfg = ReactionConfig(
            train_h5=str(h5_path),
            val_h5=str(h5_path),
            batch_size=1,
            num_workers=0,
            epochs=1,
            save_optimizer=False,
            emb_dim=32,
            inv_layer=1,
            se3_layer=1,
            heads=4,
            atom_vocab_size=128,
            cutoff=3.0,
            num_kernel=16,
            head_hidden_dim=32,
            out_dir=str(tmp_path / "reaction_out"),
        )
        model = ReactionPretrainNet(
            descriptor=build_model_spec(cfg.model_name).build_backbone(cfg),
            emb_dim=cfg.emb_dim,
            head_hidden_dim=cfg.head_hidden_dim,
            teacher_momentum=cfg.teacher_momentum,
        )
        trainer = ReactionPretrainTrainer(
            model=model,
            cfg=cfg,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            distributed=False,
            device=torch.device("cpu"),
        )
        trainer.save_checkpoint(epoch=1, tag="reaction_no_opt")
        ckpt = torch.load(Path(cfg.out_dir) / "reaction_no_opt.pt", map_location="cpu")
        assert "optimizer" not in ckpt

    def test_qm9_trainer_respects_save_optimizer_flag(self, monkeypatch, tmp_path):
        from uni_react.configs import QM9Config
        from uni_react.models import build_model_spec
        from uni_react.tasks import build_qm9_model, resolve_qm9_task_spec
        from uni_react.tasks.qm9.common.trainer import FinetuneQM9Trainer

        class TinyQM9Dataset(Dataset):
            def __init__(self, size: int = 4):
                self.size = size

            def __len__(self):
                return self.size

            def get_targets(self, idx: int):
                return np.asarray([float(idx)], dtype=np.float64)

            def __getitem__(self, idx):
                return {
                    "atomic_numbers": torch.tensor([1, 6, 8], dtype=torch.long),
                    "coords": torch.zeros(3, 3, dtype=torch.float32),
                    "y": torch.tensor([float(idx)], dtype=torch.float32),
                }

        monkeypatch.setattr(
            "uni_react.tasks.qm9.common.trainer.build_qm9_pyg_splits",
            lambda **kwargs: {
                "train": TinyQM9Dataset(size=4),
                "valid": TinyQM9Dataset(size=2),
                "test": TinyQM9Dataset(size=2),
            },
        )

        cfg = QM9Config(
            model_name="single_mol",
            batch_size=2,
            num_workers=0,
            epochs=1,
            save_optimizer=False,
            emb_dim=32,
            inv_layer=1,
            se3_layer=1,
            heads=4,
            atom_vocab_size=128,
            cutoff=3.0,
            num_kernel=16,
            head_hidden_dim=32,
            out_dir=str(tmp_path / "qm9_out"),
        )
        task_spec = resolve_qm9_task_spec(cfg)
        model = build_qm9_model(
            cfg,
            build_model_spec(cfg.model_name),
            ["gap"],
            task_spec,
        )
        trainer = FinetuneQM9Trainer(
            model=model,
            cfg=cfg,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            targets=["gap"],
            distributed=False,
            device=torch.device("cpu"),
        )
        trainer.save_checkpoint(epoch=1, tag="qm9_no_opt")
        ckpt = torch.load(Path(cfg.out_dir) / "qm9_no_opt.pt", map_location="cpu")
        assert "optimizer" not in ckpt
    
    def test_load_checkpoint_model_weights(self, trainer):
        """Test that model weights are correctly loaded."""
        # Set some specific weights
        with torch.no_grad():
            trainer.raw_model.weight.fill_(1.0)
            trainer.raw_model.bias.fill_(2.0)
        
        trainer.save_checkpoint(epoch=1, tag="test")
        
        # Create new model with different weights
        new_model = nn.Linear(10, 10)
        with torch.no_grad():
            new_model.weight.fill_(0.0)
            new_model.bias.fill_(0.0)
        
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        from uni_react.training import BaseTrainer
        new_trainer = BaseTrainer(
            model=new_model,
            optimizer=new_optimizer,
            out_dir=str(trainer.out_dir),
            distributed=False,
        )
        
        # Load checkpoint
        ckpt_path = trainer.out_dir / "test.pt"
        new_trainer.load_checkpoint(str(ckpt_path))
        
        # Verify weights restored
        assert torch.allclose(new_trainer.raw_model.weight, torch.ones(10, 10))
        assert torch.allclose(new_trainer.raw_model.bias, torch.ones(10) * 2.0)

    def test_save_checkpoint_uses_wrapped_model_when_distributed(self, trainer):
        """Distributed save should serialize the wrapped DDP module state."""
        trainer.distributed = True

        class WrappedModel:
            def __init__(self, module):
                self.module = module

        trainer.model = WrappedModel(trainer.raw_model)
        trainer.save_checkpoint(epoch=1, tag="ddp_test")

        ckpt = torch.load(trainer.out_dir / "ddp_test.pt", map_location="cpu")
        assert "model" in ckpt
        assert set(ckpt["model"]) == set(trainer.raw_model.state_dict())

    def test_load_checkpoint_restores_scheduler_state(self, trainer):
        """Scheduler state should be restored on resume when available."""

        class DummyScheduler:
            def __init__(self):
                self.state = {"steps": 0}

            def state_dict(self):
                return dict(self.state)

            def load_state_dict(self, state):
                self.state = dict(state)

        trainer.scheduler = DummyScheduler()
        trainer.scheduler.state = {"steps": 12}
        trainer.save_checkpoint(epoch=3, tag="scheduler")

        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_scheduler = DummyScheduler()

        from uni_react.training import BaseTrainer
        new_trainer = BaseTrainer(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            out_dir=str(trainer.out_dir),
            distributed=False,
        )

        new_trainer.load_checkpoint(str(trainer.out_dir / "scheduler.pt"))
        assert new_scheduler.state == {"steps": 12}

    def test_load_checkpoint_validates_saved_config(self, tmp_path):
        """Restart config mismatches should raise unless explicitly ignored."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        from uni_react.training import BaseTrainer
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            out_dir=str(tmp_path),
            distributed=False,
            checkpoint_config={"train_mode": "geometric_structure", "batch_size": 32},
        )
        trainer.save_checkpoint(epoch=2, tag="cfg")

        new_trainer = BaseTrainer(
            model=nn.Linear(10, 10),
            optimizer=torch.optim.Adam(nn.Linear(10, 10).parameters(), lr=1e-3),
            out_dir=str(tmp_path),
            distributed=False,
            checkpoint_config={"train_mode": "cdft", "batch_size": 32},
        )
        with pytest.raises(ValueError, match="Restart config mismatch"):
            new_trainer.load_checkpoint(str(tmp_path / "cfg.pt"))

        ignored_trainer = BaseTrainer(
            model=nn.Linear(10, 10),
            optimizer=torch.optim.Adam(nn.Linear(10, 10).parameters(), lr=1e-3),
            out_dir=str(tmp_path),
            distributed=False,
            checkpoint_config={"train_mode": "cdft", "batch_size": 32},
        )
        epoch = ignored_trainer.load_checkpoint(
            str(tmp_path / "cfg.pt"),
            ignore_config_mismatch=True,
        )
        assert epoch == 3
    
    def test_reduce_bag_uses_public_methods(self, trainer):
        """Test that _reduce_bag uses public methods instead of private attributes."""
        from uni_react.training import MetricBag
        
        bag = MetricBag(['loss', 'mae'])
        bag.update('loss', 1.0, weight=10)
        bag.update('mae', 0.5, weight=10)
        
        result = trainer._reduce_bag(bag)
        
        assert 'loss' in result
        assert 'mae' in result
        assert result['loss'] == pytest.approx(1.0)  # 10.0 / 10
        assert result['mae'] == pytest.approx(0.5)   # 5.0 / 10
    
    def test_reduce_bag_empty(self, trainer):
        """Test _reduce_bag with empty bag."""
        from uni_react.training import MetricBag
        
        bag = MetricBag([])
        result = trainer._reduce_bag(bag)
        
        assert result == {}


class TestCheckpointUtils:
    """Tests for checkpoint utility functions."""
    
    def test_build_checkpoint_dict(self, tmp_path):
        """Test building checkpoint dictionary."""
        from uni_react.training.checkpoint import build_checkpoint_dict
        import argparse
        
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        args = argparse.Namespace(
            lr=1e-3,
            batch_size=32,
            epochs=100,
        )
        
        train_metrics = {"loss": 1.0, "acc": 0.9}
        val_metrics = {"loss": 1.2, "acc": 0.85}
        
        ckpt = build_checkpoint_dict(
            model=model,
            optimizer=optimizer,
            args=args,
            distributed=False,
            world_size=1,
            epoch=5,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val=1.0,
        )
        
        assert "epoch" in ckpt
        assert "model" in ckpt
        assert "optimizer" in ckpt
        assert "args" in ckpt
        assert "train" in ckpt
        assert "val" in ckpt
        assert "world_size" in ckpt
        assert "time" in ckpt
        assert "best_val" in ckpt
        
        assert ckpt["epoch"] == 5
        assert ckpt["world_size"] == 1
        assert ckpt["best_val"] == 1.0
        assert ckpt["train"] == train_metrics
        assert ckpt["val"] == val_metrics
    
    def test_build_checkpoint_dict_without_optimizer(self):
        """Test building checkpoint without optimizer."""
        from uni_react.training.checkpoint import build_checkpoint_dict
        import argparse
        
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        args = argparse.Namespace(lr=1e-3)
        
        ckpt = build_checkpoint_dict(
            model=model,
            optimizer=optimizer,
            args=args,
            distributed=False,
            world_size=1,
            epoch=5,
            train_metrics={},
            val_metrics=None,
            include_optimizer=False,
        )
        
        assert "model" in ckpt
        assert "optimizer" not in ckpt
    
    def test_load_restart_checkpoint(self, tmp_path):
        """Test loading restart checkpoint."""
        from uni_react.training.checkpoint import (
            build_checkpoint_dict,
            load_restart_checkpoint
        )
        import argparse
        
        # Create and save a checkpoint
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        args = argparse.Namespace(lr=1e-3, batch_size=32)
        
        ckpt = build_checkpoint_dict(
            model=model,
            optimizer=optimizer,
            args=args,
            distributed=False,
            world_size=1,
            epoch=5,
            train_metrics={"loss": 1.0},
            val_metrics={"loss": 1.2},
            best_val=1.0,
        )
        
        ckpt_path = tmp_path / "test.pt"
        torch.save(ckpt, ckpt_path)
        
        # Load checkpoint
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        start_epoch, step_in_epoch, best_val, ckpt_args, optimizer_loaded, ckpt_world_size = \
            load_restart_checkpoint(
                restart_path=str(ckpt_path),
                model=new_model,
                optimizer=new_optimizer,
                device=torch.device('cpu'),
                distributed=False,
            )
        
        assert start_epoch == 6  # epoch + 1
        assert step_in_epoch == 0
        assert best_val == 1.0
        assert ckpt_args is not None
        assert optimizer_loaded is True
        assert ckpt_world_size == 1

    def test_validate_restart_config_accepts_mapping(self):
        """Restart validation should work with dict-based configs used by trainers."""
        from uni_react.training.checkpoint import validate_restart_config

        with pytest.raises(ValueError, match="Restart config mismatch"):
            validate_restart_config(
                ckpt_args={"train_mode": "geometric_structure"},
                cur_args={"train_mode": "cdft"},
                ignore_config_mismatch=False,
                rank=0,
            )

    def test_validate_restart_config_catches_qm9_specific_mismatch(self):
        """QM9-specific restart keys should be validated too."""
        from uni_react.training.checkpoint import validate_restart_config

        with pytest.raises(ValueError, match="Restart config mismatch"):
            validate_restart_config(
                ckpt_args={
                    "target": "gap",
                    "split": "egnn",
                    "head_hidden_dim": 256,
                    "head_dropout": 0.1,
                },
                cur_args={
                    "target": "homo",
                    "split": "dimenet",
                    "head_hidden_dim": 512,
                    "head_dropout": 0.2,
                },
                ignore_config_mismatch=False,
                rank=0,
            )


class TestSchedulers:
    def test_cosine_scheduler_roundtrip_state(self):
        from uni_react.training.scheduler import WarmupCosineScheduler

        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1, total_steps=4)
        scheduler.step()
        scheduler.step()
        saved = scheduler.state_dict()

        new_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
        restored = WarmupCosineScheduler(new_optimizer, warmup_steps=1, total_steps=1)
        restored.load_state_dict(saved)

        assert restored.state_dict()["step_count"] == 2
        assert restored.state_dict()["total_steps"] == 4

    def test_pretrain_trainer_updates_scheduler_total_steps(self, monkeypatch, tmp_path):
        from uni_react.configs import GeometricConfig
        from uni_react.tasks.geometric.common.trainer import PretrainTrainer

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "atomic_numbers": torch.tensor([1, 6], dtype=torch.long),
                    "input_atomic_numbers": torch.tensor([1, 94], dtype=torch.long),
                    "coords": torch.zeros(2, 3),
                    "coords_noisy": torch.zeros(2, 3),
                    "noise": torch.zeros(2, 3),
                    "mask_positions": torch.tensor([False, True]),
                    "charges": torch.zeros(2),
                    "charge_valid": torch.tensor([False, False]),
                }

        class DummyLoss:
            def metric_keys(self):
                return ["loss"]

        class DummyScheduler:
            def __init__(self):
                self.total_steps = None

            def set_total_steps(self, total_steps):
                self.total_steps = total_steps

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(1, 1)

            def forward(self, **kwargs):
                return {"loss": self.proj.weight.sum().unsqueeze(0)}

        monkeypatch.setattr(
            "uni_react.tasks.geometric.common.trainer.build_pretrain_dataset",
            lambda *args, **kwargs: DummyDataset(),
        )

        cfg = GeometricConfig(
            train_h5=["dummy.h5"],
            batch_size=4,
            num_workers=0,
            epochs=3,
            out_dir=str(tmp_path),
        )
        scheduler = DummyScheduler()
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        PretrainTrainer(
            model=model,
            loss_fn=DummyLoss(),
            optimizer=optimizer,
            cfg=cfg,
            scheduler=scheduler,
            distributed=False,
            device=torch.device("cpu"),
        )

        # drop_last=True, so len(loader)=floor(10/4)=2 and total steps = 3 * 2
        assert scheduler.total_steps == 6

    def test_density_trainer_advances_distributed_sampler_epoch(self, tmp_path):
        from uni_react.configs import DensityConfig
        from uni_react.tasks.density.common.trainer import DensityPretrainTrainer

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "atomic_numbers": torch.tensor([1, 6], dtype=torch.long),
                    "coords": torch.zeros(2, 3),
                    "atom_padding": torch.tensor([False, False]),
                    "query_points": torch.zeros(3, 3),
                    "density_target": torch.zeros(3),
                    "total_charge": torch.tensor(0.0),
                    "spin_multiplicity": torch.tensor(1.0),
                }

        class DummySampler:
            def __init__(self):
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

            def __iter__(self):
                return iter(range(4))

            def __len__(self):
                return 4

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, **kwargs):
                batch = kwargs["query_points"].shape[0]
                points = kwargs["query_points"].shape[1]
                pred = self.scale * torch.ones((batch, points))
                return {"density_pred": pred}

        sampler = DummySampler()
        loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2, sampler=sampler)
        cfg = DensityConfig(
            train_h5=["dummy.h5"],
            batch_size=2,
            num_workers=0,
            epochs=2,
            out_dir=str(tmp_path),
        )
        model = DummyModel()
        trainer = DensityPretrainTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            train_loader=loader,
            val_loader=None,
            cfg=cfg,
            distributed=False,
            device=torch.device("cpu"),
        )
        trainer._train_sampler = sampler

        trainer.train_epoch(3)
        assert sampler.epochs == [3]

    def test_qm9_trainer_advances_distributed_sampler_epoch(self, monkeypatch, tmp_path):
        from uni_react.configs import QM9Config
        from uni_react.tasks.qm9.common.trainer import FinetuneQM9Trainer

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def get_targets(self, idx):
                return [0.0]

            def __getitem__(self, idx):
                return {
                    "atomic_numbers": torch.tensor([1, 6], dtype=torch.long),
                    "coords": torch.zeros(2, 3),
                    "y": torch.tensor([0.0]),
                }

        class DummySampler:
            def __init__(self, dataset, shuffle, drop_last):
                self.dataset = dataset
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

            def __iter__(self):
                return iter(range(len(self.dataset)))

            def __len__(self):
                return len(self.dataset)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.descriptor = nn.Linear(1, 1)
                self.head = nn.Linear(1, 1)

            def forward(self, atomic_numbers, coords, atom_padding):
                batch = atomic_numbers.shape[0]
                pred = self.head(self.descriptor.weight.view(1, 1)).expand(batch, 1)
                return {"pred": pred.squeeze(-1)}

        monkeypatch.setattr(
            "uni_react.tasks.qm9.common.trainer.build_qm9_pyg_splits",
            lambda **kwargs: {"train": DummyDataset(), "valid": DummyDataset(), "test": DummyDataset()},
        )
        monkeypatch.setattr("uni_react.tasks.qm9.common.trainer.DistributedSampler", DummySampler)

        cfg = QM9Config(
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
            head_hidden_dim=32,
            out_dir=str(tmp_path),
        )
        model = DummyModel()
        trainer = FinetuneQM9Trainer(
            model=model,
            cfg=cfg,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            targets=["gap"],
            distributed=False,
            device=torch.device("cpu"),
        )
        trainer._train_sampler = DummySampler(trainer._train_loader.dataset, shuffle=True, drop_last=True)

        trainer.train_epoch(5)
        assert trainer._train_sampler.epochs == [5]


class TestDistributedUtils:
    """Tests for distributed training utilities."""
    
    def test_is_main_process(self):
        """Test is_main_process function."""
        from uni_react.training.distributed import is_main_process
        
        assert is_main_process(0) is True
        assert is_main_process(1) is False
        assert is_main_process(2) is False


class TestOptimizerUtils:
    """Tests for optimizer utilities."""

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.descriptor = nn.Linear(10, 20)
            self.head = nn.Linear(20, 5)

        def forward(self, x):
            return self.head(self.descriptor(x))
    
    def test_build_optimizer_single_lr(self):
        """Test building optimizer with single learning rate."""
        from uni_react.training.optimizer import build_optimizer
        
        model = self._TinyModel()
        
        optimizer = build_optimizer(
            model=model,
            distributed=False,
            lr_default=1e-3,
            weight_decay=1e-4,
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2
        assert {pg["name"] for pg in optimizer.param_groups} == {"descriptor", "tasks"}
        assert all(pg['lr'] == 1e-3 for pg in optimizer.param_groups)
        assert all(pg['weight_decay'] == 1e-4 for pg in optimizer.param_groups)
    
    def test_build_optimizer_layered_lr(self):
        """Test building optimizer with different learning rates for different layers."""
        from uni_react.training.optimizer import build_optimizer
        
        model = self._TinyModel()
        
        optimizer = build_optimizer(
            model=model,
            distributed=False,
            lr_default=1e-3,
            weight_decay=1e-4,
            descriptor_lr=1e-4,
            task_lr=1e-3,
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2
        
        lrs = {pg['name']: pg['lr'] for pg in optimizer.param_groups}
        assert lrs["descriptor"] == 1e-4
        assert lrs["tasks"] == 1e-3


class TestSeedUtils:
    """Tests for random seed utilities."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        from uni_react.training.seed import set_seed
        
        # Set seed and generate random numbers
        set_seed(42)
        rand1 = torch.rand(10)
        
        # Set same seed again
        set_seed(42)
        rand2 = torch.rand(10)
        
        # Should be identical
        assert torch.allclose(rand1, rand2)
    
    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        from uni_react.training.seed import set_seed
        
        set_seed(42)
        rand1 = torch.rand(10)
        
        set_seed(123)
        rand2 = torch.rand(10)
        
        # Should be different
        assert not torch.allclose(rand1, rand2)
