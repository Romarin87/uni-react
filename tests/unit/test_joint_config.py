import pytest


def _base_joint_config(**updates):
    from uni_react.configs import JointConfig

    data = dict(
        run={"device": "cpu", "seed": 1, "out_dir": "runs/test", "num_workers": 0},
        model={
            "name": "single_mol",
            "emb_dim": 16,
            "inv_layer": 1,
            "se3_layer": 1,
            "heads": 4,
            "atom_vocab_size": 16,
            "cutoff": 3.0,
            "num_kernel": 8,
        },
        tasks={
            "atom_mask": {
                "enabled": True,
                "train_h5": ["dummy.h5"],
                "batch_size": 2,
                "params": {"mask_token_id": 15},
            },
            "coord_denoise": {
                "enabled": True,
                "train_h5": ["dummy.h5"],
                "batch_size": 2,
            },
        },
        schedule={"sample_prob": {"atom_mask": 1.0, "coord_denoise": 0.0}},
        loss_weights={
            "initial": {"atom_mask": 1.0, "coord_denoise": 1.0},
            "final": {"atom_mask": 1.0, "coord_denoise": 1.0},
        },
        learning_rates={
            "descriptor": {"atom_mask": 1e-5, "coord_denoise": 1e-5},
            "head": {"atom_mask": 1e-4, "coord_denoise": 1e-4},
        },
        optimization={"train_unit": "steps", "max_steps": 10},
    )
    data.update(updates)
    return JointConfig(**data)


def test_joint_config_allows_zero_probability_ablation():
    cfg = _base_joint_config()
    assert cfg.active_train_tasks == ["atom_mask"]


def test_joint_config_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unknown joint task"):
        _base_joint_config(tasks={"not_a_task": {"enabled": True}})


def test_joint_config_epochs_requires_reference_task():
    with pytest.raises(ValueError, match="epoch_reference_task"):
        _base_joint_config(optimization={"train_unit": "epochs", "epochs": 2})


def test_joint_config_rejects_unknown_eval_mode():
    with pytest.raises(ValueError, match="evaluation.eval_tasks"):
        _base_joint_config(evaluation={"eval_tasks": "sometimes"})


def test_distributed_task_sampling_uses_broadcast(monkeypatch):
    import torch

    from uni_react.tasks.joint.trainer import JointTrainer

    trainer = object.__new__(JointTrainer)
    trainer.distributed = True
    trainer.rank = 1
    trainer.device = torch.device("cpu")
    trainer.active_tasks = ["atom_mask", "electron_density"]
    trainer.sample_probs = torch.tensor([0.0, 1.0])
    trainer._rng = torch.Generator(device="cpu").manual_seed(1)
    calls = []

    monkeypatch.setattr("uni_react.tasks.joint.trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("uni_react.tasks.joint.trainer.dist.is_initialized", lambda: True)

    def broadcast(tensor, src):
        calls.append(src)
        tensor.fill_(0)

    monkeypatch.setattr("uni_react.tasks.joint.trainer.dist.broadcast", broadcast)

    assert trainer._sample_task() == "atom_mask"
    assert calls == [0]


def test_atom_mask_zero_mask_returns_zero_loss():
    import torch

    from uni_react.tasks.atom_mask import AtomMaskAdapter

    adapter = AtomMaskAdapter(
        "atom_mask",
        {"batch_size": 2, "params": {"mask_token_id": 15}},
        {"seed": 1},
        {"atom_vocab_size": 16},
        {},
    )
    logits = torch.randn(2, 3, 16, requires_grad=True)
    batch = {
        "atomic_numbers": torch.tensor([[1, 6, 0], [8, 1, 0]], dtype=torch.long),
        "atom_padding": torch.tensor([[False, False, True], [False, False, True]]),
        "mask_positions": torch.zeros(2, 3, dtype=torch.bool),
    }

    metrics = adapter.compute_metrics({"atom_logits": logits}, batch)
    metrics["loss"].backward()

    assert metrics["loss"].item() == 0.0
    assert metrics["acc"].item() == 0.0
    assert torch.count_nonzero(logits.grad).item() == 0


def test_scalar_tasks_do_not_require_fukui_labels(tmp_path):
    import h5py
    import numpy as np

    from uni_react.tasks.vea import VeaAdapter
    from uni_react.tasks.vip import VipAdapter

    path = tmp_path / "scalar_only.h5"
    with h5py.File(path, "w") as h5:
        frames = h5.create_group("frames")
        atoms = h5.create_group("atoms")
        frames.create_dataset("offsets", data=np.array([0, 2], dtype=np.int64))
        frames.create_dataset("vip", data=np.array([8.0], dtype=np.float32))
        frames.create_dataset("vea", data=np.array([1.0], dtype=np.float32))
        atoms.create_dataset("Z", data=np.array([1, 6], dtype=np.int64))
        atoms.create_dataset("R", data=np.zeros((2, 3), dtype=np.float32))

    run_cfg = {"seed": 1}
    model_cfg = {"atom_vocab_size": 16}
    task_cfg = {"batch_size": 1, "params": {"mask_token_id": 15}}

    for name, cls in (("vip", VipAdapter), ("vea", VeaAdapter)):
        adapter = cls(name, task_cfg, run_cfg, model_cfg, {})
        result = adapter.build_dataset([str(path)], split="train")
        sample = result.dataset[0]
        assert "reactivity_global" in sample
        assert sample["reactivity_global"].shape == (1,)
        assert sample["reactivity_atom"].shape == (2, 0)
        assert all("f_plus" not in key and "f_minus" not in key and "f_zero" not in key for key in result.required_keys)


def test_fukui_task_does_not_require_scalar_labels(tmp_path):
    import h5py
    import numpy as np

    from uni_react.tasks.fukui import FukuiAdapter

    path = tmp_path / "fukui_only.h5"
    with h5py.File(path, "w") as h5:
        frames = h5.create_group("frames")
        atoms = h5.create_group("atoms")
        frames.create_dataset("offsets", data=np.array([0, 2], dtype=np.int64))
        atoms.create_dataset("Z", data=np.array([1, 6], dtype=np.int64))
        atoms.create_dataset("R", data=np.zeros((2, 3), dtype=np.float32))
        atoms.create_dataset("f_plus", data=np.array([0.1, 0.2], dtype=np.float32))
        atoms.create_dataset("f_minus", data=np.array([0.0, 0.1], dtype=np.float32))
        atoms.create_dataset("f_zero", data=np.array([0.05, 0.15], dtype=np.float32))

    adapter = FukuiAdapter(
        "fukui",
        {"batch_size": 1, "params": {"mask_token_id": 15}},
        {"seed": 1},
        {"atom_vocab_size": 16},
        {},
    )
    result = adapter.build_dataset([str(path)], split="train")
    sample = result.dataset[0]
    assert sample["reactivity_global"].shape == (0,)
    assert sample["reactivity_atom"].shape == (2, 3)


def test_joint_entry_accepts_legacy_out_dir_override(monkeypatch, tmp_path):
    from uni_react.tasks.joint import entry

    config_path = tmp_path / "joint.yaml"
    config_path.write_text(
        """
run:
  device: cpu
  out_dir: old
model:
  name: single_mol
  atom_vocab_size: 16
tasks:
  atom_mask:
    enabled: true
    train_h5: [dummy.h5]
    batch_size: 1
    params:
      mask_token_id: 15
schedule:
  sample_prob:
    atom_mask: 1.0
learning_rates:
  descriptor:
    atom_mask: 1.0e-5
  head:
    atom_mask: 1.0e-4
optimization:
  train_unit: steps
  max_steps: 1
""",
        encoding="utf-8",
    )
    seen = {}

    class DummyTrainer:
        def fit(self):
            seen["fit"] = True

    class DummyLogger:
        def log_metrics(self, stage, metrics):
            seen["metrics"] = (stage, metrics)

        def log_config(self, config):
            seen["config"] = config

    monkeypatch.setattr(
        "sys.argv",
        ["uni-react-train-joint", "--config", str(config_path), "--out_dir", str(tmp_path / "new")],
    )
    monkeypatch.setattr(entry, "init_distributed", lambda device: (False, 0, 1, 0, "cpu"))
    monkeypatch.setattr(entry, "set_seed", lambda seed: None)
    monkeypatch.setattr(entry, "dump_runtime_config", lambda cfg, out_dir, runtime: seen.update(cfg=cfg, out_dir=out_dir))
    monkeypatch.setattr(entry, "build_console_logger", lambda out_dir, log_file, rank: DummyLogger())
    monkeypatch.setattr(entry, "build_joint_trainer", lambda cfg, **kwargs: DummyTrainer())
    monkeypatch.setattr(entry, "cleanup_distributed", lambda distributed: None)

    entry.run_joint_entry()

    assert seen["cfg"].run["out_dir"] == str(tmp_path / "new")
    assert seen["fit"] is True


def test_joint_next_batch_advances_distributed_sampler_epoch():
    from uni_react.tasks.joint.trainer import JointTrainer

    class DummySampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    class DummyLoader:
        def __init__(self):
            self.sampler = DummySampler()

        def __iter__(self):
            return iter(["batch"])

    trainer = object.__new__(JointTrainer)
    loader = DummyLoader()
    trainer.train_loaders = {"atom_mask": loader}
    trainer._train_iters = {"atom_mask": iter(())}
    trainer._train_sampler_epochs = {"atom_mask": 0}

    assert trainer._next_batch("atom_mask") == "batch"
    assert trainer._train_sampler_epochs["atom_mask"] == 1
    assert loader.sampler.epochs == [1]


def test_joint_eval_reduces_metrics_across_ranks(monkeypatch):
    import torch
    from types import SimpleNamespace

    from uni_react.tasks.joint.trainer import JointTrainer

    class DummyCfg:
        loss_weights = {}

        def evaluation_value(self, key, default=None):
            return default

        def advanced_value(self, *keys, default=None):
            return default

    class DummyModel:
        module = None

        def __init__(self):
            self.module = self

        def eval(self):
            pass

        def forward_task(self, task_name, batch):
            return {}

    class DummyAdapter:
        def metric_names(self):
            return ("loss", "mae")

        def compute_metrics(self, outputs, batch):
            return {"loss": torch.tensor(1.0), "mae": torch.tensor(3.0)}

    def all_reduce(tensor, op=None):
        tensor += torch.tensor([6.0, 2.0, 2.0], dtype=tensor.dtype, device=tensor.device)

    trainer = object.__new__(JointTrainer)
    trainer.model = DummyModel()
    trainer.distributed = True
    trainer.device = torch.device("cpu")
    trainer.global_step = 0
    trainer.max_steps = 1
    trainer.estimated_joint_epoch_steps = None
    trainer.cfg = DummyCfg()
    trainer.val_loaders = {"atom_mask": [{"atomic_numbers": torch.zeros(2, 1, dtype=torch.long)}]}
    trainer.data_plan = SimpleNamespace(
        task_data={"atom_mask": SimpleNamespace(adapter=DummyAdapter())}
    )

    monkeypatch.setattr("uni_react.tasks.joint.trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("uni_react.tasks.joint.trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr("uni_react.tasks.joint.trainer.dist.all_reduce", all_reduce)

    metrics = trainer.eval_all()

    assert metrics["atom_mask_loss"] == 2.0
    assert metrics["atom_mask_mae"] == 2.0
    assert metrics["weighted_val_loss"] == 2.0
