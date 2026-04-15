"""Runtime semantics regressions for entrypoints and datasets."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


def test_density_entrypoint_honors_no_center_coords(monkeypatch, tmp_path):
    import uni_react.training.density_runner as density_main
    from uni_react.configs import DensityPretrainConfig

    observed = {"train": None, "val": None}
    cfg = DensityPretrainConfig(
        train_h5=["train.h5"],
        val_h5=["val.h5"],
        batch_size=1,
        num_workers=0,
        center_coords=True,
        no_center_coords=True,
        out_dir=str(tmp_path),
    )

    class StopRun(Exception):
        pass

    class DummyDataset:
        def __init__(self, h5_files, num_query_points, center_coords, deterministic, seed, return_ids):
            key = "val" if deterministic else "train"
            observed[key] = center_coords

        def __len__(self):
            return 2

    class DummyTrainer:
        def __init__(self, **kwargs):
            raise StopRun

    monkeypatch.setattr(
        density_main, "load_dataclass_config",
        lambda args, cls: cfg,
    )
    monkeypatch.setattr(density_main, "init_distributed", lambda device: (False, 0, 1, 0, __import__("torch").device("cpu")))
    monkeypatch.setattr(density_main, "set_seed", lambda seed: None)
    monkeypatch.setattr(density_main, "dump_runtime_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(density_main, "expand_h5_files", lambda files: list(files))
    monkeypatch.setattr(density_main, "H5DensityPretrainDataset", DummyDataset)
    monkeypatch.setattr(density_main, "DensityPretrainNet", lambda **kwargs: type("M", (), {"parameters": lambda self: [], "descriptor": type("D", (), {"parameters": lambda self: []})(), "named_parameters": lambda self: [], "to": lambda self, device: self})())
    monkeypatch.setattr(density_main, "DensityPretrainTrainer", DummyTrainer)
    monkeypatch.setattr(density_main, "build_split_lr_optimizer", lambda **kwargs: object())
    monkeypatch.setattr(density_main, "build_console_logger", lambda *args, **kwargs: type("L", (), {"set_rank": lambda self, rank: None, "log_config": lambda self, cfg: None, "log": lambda self, metrics, **kwargs: None})())
    monkeypatch.setattr(density_main.SCHEDULER_REGISTRY, "build", lambda cfg: None)
    monkeypatch.setattr(density_main, "load_init_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(
        density_main,
        "build_dataclass_arg_parser",
        lambda *args, **kwargs: type("P", (), {"parse_args": lambda self: type("Args", (), {"config": "cfg.json"})()})(),
    )

    with pytest.raises(StopRun):
        density_main.run_density_entry()

    assert observed["train"] is False
    assert observed["val"] is False


def test_reaction_dataset_singleton_negative_falls_back_to_positive(tmp_path):
    from uni_react.utils.reaction_dataset import ReactionTripletH5Dataset

    h5_path = Path(tmp_path) / "singleton.h5"
    with h5py.File(h5_path, "w") as h5:
        g = h5.create_group("triplets")
        g.create_dataset("offsets", data=np.asarray([0, 2], dtype=np.int64))
        g.create_dataset("n_atoms", data=np.asarray([2], dtype=np.int32))
        g.create_dataset("comp_hash", data=np.asarray([7], dtype=np.int64))
        z = np.asarray([1, 8], dtype=np.int64)
        r = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        for prefix in ("r", "ts", "p"):
            g.create_dataset(f"{prefix}_Z", data=z)
            g.create_dataset(f"{prefix}_R", data=r)

    ds = ReactionTripletH5Dataset(str(h5_path), neg_ratio=1.0, hard_negative=True, seed=0)
    sample = ds[0]
    assert sample["cons_label"] == 1.0
    assert np.array_equal(sample["R"][0], sample["R_cons"][0])
    assert np.allclose(sample["R"][1], sample["R_cons"][1])


def test_entrypoint_parser_accepts_multi_value_h5_lists():
    from uni_react.configs import PretrainConfig
    from uni_react.training.entrypoint_utils import build_dataclass_arg_parser

    parser = build_dataclass_arg_parser(PretrainConfig, "pretrain")
    args = parser.parse_args(["--train_h5", "a.h5", "b.h5", "--val_h5", "v1.h5", "v2.h5"])

    assert args.train_h5 == ["a.h5", "b.h5"]
    assert args.val_h5 == ["v1.h5", "v2.h5"]


def test_entrypoint_parser_can_disable_true_boolean_fields():
    from uni_react.configs import DensityPretrainConfig, ReactionPretrainConfig
    from uni_react.training.entrypoint_utils import build_dataclass_arg_parser

    density_parser = build_dataclass_arg_parser(DensityPretrainConfig, "density")
    density_args = density_parser.parse_args(["--center_coords", "false"])
    assert density_args.center_coords is False

    reaction_parser = build_dataclass_arg_parser(ReactionPretrainConfig, "reaction")
    reaction_args = reaction_parser.parse_args(["--hard_negative", "false"])
    assert reaction_args.hard_negative is False


def test_entrypoint_parser_preserves_optional_numeric_types():
    from uni_react.configs import DensityPretrainConfig, PretrainConfig
    from uni_react.training.entrypoint_utils import build_dataclass_arg_parser, load_dataclass_config

    pretrain_parser = build_dataclass_arg_parser(PretrainConfig, "pretrain")
    pretrain_args = pretrain_parser.parse_args(["--descriptor_lr", "1e-5", "--task_lr", "2e-4"])
    pretrain_cfg = load_dataclass_config(pretrain_args, PretrainConfig)

    assert isinstance(pretrain_args.descriptor_lr, float)
    assert isinstance(pretrain_args.task_lr, float)
    assert pretrain_cfg.descriptor_lr == pytest.approx(1e-5)
    assert pretrain_cfg.task_lr == pytest.approx(2e-4)

    density_parser = build_dataclass_arg_parser(DensityPretrainConfig, "density")
    density_args = density_parser.parse_args(["--lr", "1e-4"])
    density_cfg = load_dataclass_config(density_args, DensityPretrainConfig)

    assert isinstance(density_args.lr, float)
    assert density_cfg.lr == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("train_mode", "encoder_type", "expected_out_dir"),
    [
        ("geometric_structure", "single_mol", "runs/single_mol_geometric"),
        ("cdft", "reacformer_so2", "runs/reacformer_so2_cdft"),
    ],
)
def test_pretrain_runner_derives_stage_specific_default_out_dir(
    monkeypatch, train_mode, encoder_type, expected_out_dir
):
    import uni_react.training.pretrain_runner as pretrain_runner
    from uni_react.configs import PretrainConfig

    observed = {"out_dir": None}

    class StopRun(Exception):
        pass

    cfg = PretrainConfig(
        train_h5=["train.h5"],
        val_h5=["val.h5"],
        batch_size=1,
        num_workers=0,
        atom_vocab_size=128,
        mask_token_id=94,
        encoder_type=encoder_type,
        out_dir="",
    )

    monkeypatch.setattr(
        pretrain_runner,
        "build_dataclass_arg_parser",
        lambda *args, **kwargs: type("P", (), {"parse_args": lambda self: type("Args", (), {"config": ""})()})(),
    )
    monkeypatch.setattr(pretrain_runner, "load_dataclass_config", lambda args, cls: cfg)
    monkeypatch.setattr(
        pretrain_runner,
        "init_distributed",
        lambda device: (False, 0, 1, 0, __import__("torch").device("cpu")),
    )
    monkeypatch.setattr(pretrain_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        pretrain_runner,
        "build_console_logger",
        lambda out_dir, log_file, rank: observed.update({"out_dir": out_dir}) or type(
            "L",
            (),
            {"log_config": lambda self, cfg_dict: None, "log": lambda self, metrics, **kwargs: None},
        )(),
    )
    monkeypatch.setattr(pretrain_runner, "dump_runtime_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_runner, "build_pretrain_model", lambda *args, **kwargs: (_ for _ in ()).throw(StopRun()))

    with pytest.raises(StopRun):
        pretrain_runner.run_pretrain_entry(train_mode=train_mode, description="test")

    assert observed["out_dir"] == expected_out_dir


@pytest.mark.parametrize(
    ("targets", "encoder_type", "pretrained_ckpt", "split", "expected_out_dir"),
    [
        (["gap"], "single_mol", None, "egnn", "runs/qm9_scratch_single_mol_egnn_gap"),
        (
            ["gap", "homo"],
            "reacformer_se3",
            "runs/reacformer_se3_reaction/best.pt",
            "dimenet",
            "runs/qm9_pretrain_reaction_reacformer_se3_dimenet_multi",
        ),
        (
            ["gap"],
            "single_mol",
            "runs/single_mol_density/best.pt",
            "egnn",
            "runs/qm9_pretrain_density_single_mol_egnn_gap",
        ),
    ],
)
def test_qm9_runner_derives_backbone_aware_default_out_dir(
    monkeypatch, targets, encoder_type, pretrained_ckpt, split, expected_out_dir
):
    import uni_react.training.qm9_runner as qm9_runner
    from uni_react.configs import FinetuneQM9Config

    observed = {"out_dir": None}

    class StopRun(Exception):
        pass

    cfg = FinetuneQM9Config(
        data_root="qm9_pyg",
        target=targets[0],
        targets=targets,
        split=split,
        batch_size=1,
        num_workers=0,
        encoder_type=encoder_type,
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_kernel=16,
        pretrained_ckpt=pretrained_ckpt,
        out_dir="",
    )

    monkeypatch.setattr(
        qm9_runner,
        "build_dataclass_arg_parser",
        lambda *args, **kwargs: type("P", (), {"parse_args": lambda self: type("Args", (), {"config": ""})()})(),
    )
    monkeypatch.setattr(qm9_runner, "load_dataclass_config", lambda args, cls: cfg)
    monkeypatch.setattr(
        qm9_runner,
        "init_distributed",
        lambda device: (False, 0, 1, 0, __import__("torch").device("cpu")),
    )
    monkeypatch.setattr(qm9_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        qm9_runner,
        "dump_runtime_config",
        lambda runtime_cfg, out_dir: observed.update({"out_dir": str(out_dir)}) or (_ for _ in ()).throw(StopRun()),
    )

    with pytest.raises(StopRun):
        qm9_runner.run_qm9_entry()

    assert observed["out_dir"] == expected_out_dir


def test_qm9_runner_derives_custom_pretrain_family_name():
    from uni_react.configs import FinetuneQM9Config
    from uni_react.training.qm9_runner import infer_qm9_run_family

    cfg = FinetuneQM9Config(pretrained_ckpt="/tmp/my_custom_source/epoch_0020.pt")
    assert infer_qm9_run_family(cfg) == "pretrain_my_custom_source"


def test_qm9_runner_resume_reuses_checkpoint_parent_dir():
    from uni_react.configs import FinetuneQM9Config
    from uni_react.training.qm9_runner import _derive_qm9_out_dir

    restart_path = "/tmp/runs/qm9_pretrain_cdft_single_mol_egnn_gap/latest.pt"
    cfg = FinetuneQM9Config(
        restart=restart_path,
        pretrained_ckpt=None,
        out_dir="",
    )
    assert _derive_qm9_out_dir(cfg, ["gap"]) == str(Path(restart_path).resolve().parent)


def test_qm9_runner_writes_structured_metrics_artifacts(tmp_path):
    from uni_react.training.qm9_runner import _write_qm9_structured_outputs

    trainer = type(
        "Trainer",
        (),
        {
            "epoch_history": [
                {
                    "epoch": 1,
                    "train": {"loss": 0.9, "mae": 0.8},
                    "val": {"loss": 0.7, "mae": 0.6},
                    "time_sec": 1.0,
                    "is_best": False,
                },
                {
                    "epoch": 2,
                    "train": {"loss": 0.5, "mae": 0.4},
                    "val": {"loss": 0.3, "mae": 0.2},
                    "time_sec": 1.0,
                    "is_best": True,
                },
            ]
        },
    )()

    _write_qm9_structured_outputs(
        tmp_path,
        trainer,
        {
            "train": {"loss": 0.45, "mae": 0.35},
            "val": {"loss": 0.25, "mae": 0.15},
            "test": {"loss": 0.2, "mae": 0.1},
        },
    )

    train_log = tmp_path / "train_log.jsonl"
    test_metrics = tmp_path / "test_metrics.json"
    assert train_log.exists()
    assert test_metrics.exists()

    lines = [json.loads(line) for line in train_log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    payload = json.loads(test_metrics.read_text(encoding="utf-8"))
    assert payload["best_epoch"] == 2
    assert payload["train"]["mae"] == pytest.approx(0.35)
    assert payload["val"]["mae"] == pytest.approx(0.15)
    assert payload["test"]["mae"] == pytest.approx(0.1)


def test_qm9_runner_loads_best_checkpoint_before_test(tmp_path):
    import torch
    from uni_react.training.qm9_runner import _load_best_model_for_qm9_test

    model = torch.nn.Linear(2, 1)
    trainer = type(
        "Trainer",
        (),
        {
            "out_dir": tmp_path,
            "device": torch.device("cpu"),
            "raw_model": model,
        },
    )()

    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(4.0)

    best_state = torch.nn.Linear(2, 1).state_dict()
    best_path = tmp_path / "best.pt"
    torch.save({"model": best_state}, best_path)

    _load_best_model_for_qm9_test(trainer)

    loaded = model.state_dict()
    assert torch.equal(loaded["weight"], best_state["weight"])
    assert torch.equal(loaded["bias"], best_state["bias"])


def test_qm9_runner_runs_best_eval_on_all_ranks(monkeypatch, tmp_path):
    import uni_react.training.qm9_runner as qm9_runner
    from uni_react.configs import FinetuneQM9Config

    observed = {"best_eval": 0, "writes": 0, "logs": 0}

    cfg = FinetuneQM9Config(
        target="gap",
        targets=["gap"],
        device="cpu",
        batch_size=1,
        num_workers=0,
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_kernel=16,
        head_hidden_dim=32,
        out_dir=str(tmp_path / "qm9_rank1"),
    )

    class StopRun(Exception):
        pass

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.out_dir = Path(cfg.out_dir)
            self.device = __import__("torch").device("cpu")
            self.raw_model = type("M", (), {"load_state_dict": lambda *a, **k: None})()
            self.epoch_history = []

        def fit(self, start_epoch=1):
            return None

    monkeypatch.setattr(
        qm9_runner,
        "build_dataclass_arg_parser",
        lambda *args, **kwargs: type("P", (), {"parse_args": lambda self: type("Args", (), {"config": ""})()})(),
    )
    monkeypatch.setattr(qm9_runner, "load_dataclass_config", lambda args, cls: cfg)
    monkeypatch.setattr(
        qm9_runner,
        "init_distributed",
        lambda device: (True, 1, 8, 1, __import__("torch").device("cpu")),
    )
    monkeypatch.setattr(qm9_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(qm9_runner, "dump_runtime_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(qm9_runner, "build_pretrain_encoder", lambda cfg: object())
    monkeypatch.setattr(
        qm9_runner,
        "build_split_lr_optimizer",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        qm9_runner,
        "_load_best_model_for_qm9_test",
        lambda trainer: None,
    )
    monkeypatch.setattr(
        qm9_runner,
        "_evaluate_best_qm9_splits",
        lambda trainer: observed.__setitem__("best_eval", observed["best_eval"] + 1) or (_ for _ in ()).throw(StopRun()),
    )
    monkeypatch.setattr(
        qm9_runner,
        "_write_qm9_structured_outputs",
        lambda *args, **kwargs: observed.__setitem__("writes", observed["writes"] + 1),
    )
    monkeypatch.setattr(
        qm9_runner,
        "build_console_logger",
        lambda *args, **kwargs: type("L", (), {"log": lambda self, *a, **k: observed.__setitem__("logs", observed["logs"] + 1)})(),
    )
    monkeypatch.setattr(
        qm9_runner,
        "cleanup_distributed",
        lambda distributed: None,
    )
    monkeypatch.setattr(
        qm9_runner,
        "is_main_process",
        lambda rank: False,
    )
    monkeypatch.setattr(
        qm9_runner,
        "FinetuneQM9Trainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        qm9_runner,
        "load_init_checkpoint",
        lambda **kwargs: None,
    )

    class DummyModel:
        def __init__(self, **kwargs):
            self.descriptor = object()

        def to(self, device):
            return self

    monkeypatch.setitem(__import__("sys").modules, "uni_react.encoders", type("E", (), {"QM9FineTuneNet": DummyModel})())

    with pytest.raises(StopRun):
        qm9_runner.run_qm9_entry()

    assert observed["best_eval"] == 1
    assert observed["writes"] == 0
    assert observed["logs"] == 0


def test_compare_qm9_tool_expands_default_prefixes_for_all_backbones():
    from uni_react.tools.compare_qm9_mae import _expand_backbone_patterns

    assert _expand_backbone_patterns("runs/qm9_scratch_") == [
        "runs/qm9_scratch_",
        "runs/qm9_scratch_reacformer_se3_",
        "runs/qm9_scratch_reacformer_so2_",
    ]
    assert _expand_backbone_patterns("runs/qm9_pretrain_cdft_*") == [
        "runs/qm9_pretrain_cdft_*",
        "runs/qm9_pretrain_reacformer_se3_cdft_*",
        "runs/qm9_pretrain_reacformer_so2_cdft_*",
    ]


def test_compare_qm9_tool_resolves_duplicate_targets_by_newer_run(tmp_path, monkeypatch):
    from uni_react.tools.compare_qm9_mae import _collect_runs

    older = tmp_path / "qm9_pretrain_cdft_gap"
    newer = tmp_path / "qm9_pretrain_cdft_gap_dup"
    older.mkdir()
    newer.mkdir()

    for run_dir, epoch in ((older, 3), (newer, 5)):
        (run_dir / "config.json").write_text(json.dumps({"target": "gap"}), encoding="utf-8")
        (run_dir / "test_metrics.json").write_text(
            json.dumps({"best_epoch": epoch, "train": {"mae": 0.1}, "val": {"mae": 0.2}, "test": {"mae": 0.3}}),
            encoding="utf-8",
        )

    older.touch()
    newer.touch()
    monkeypatch.chdir(tmp_path)
    results = _collect_runs("qm9_pretrain*")

    assert results["gap"]["run_dir"] == newer.name
    assert results["gap"]["epoch"] == 5


def test_compare_qm9_tool_rejects_ambiguous_multi_backbone_selection(tmp_path, monkeypatch):
    from uni_react.tools.compare_qm9_mae import _collect_runs

    single = tmp_path / "qm9_pretrain_cdft_gap"
    se3 = tmp_path / "qm9_pretrain_reacformer_se3_cdft_gap"
    single.mkdir()
    se3.mkdir()

    for run_dir in (single, se3):
        (run_dir / "config.json").write_text(json.dumps({"target": "gap"}), encoding="utf-8")
        (run_dir / "test_metrics.json").write_text(
            json.dumps({"best_epoch": 5, "train": {"mae": 0.1}, "val": {"mae": 0.2}, "test": {"mae": 0.3}}),
            encoding="utf-8",
        )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Ambiguous QM9 compare selection"):
        _collect_runs("qm9_pretrain*")


def test_compare_qm9_tool_rejects_same_family_duplicate_with_different_split(tmp_path, monkeypatch):
    from uni_react.tools.compare_qm9_mae import _collect_runs

    egnn = tmp_path / "qm9_pretrain_cdft_gap"
    dimenet = tmp_path / "qm9_pretrain_cdft_gap_alt"
    egnn.mkdir()
    dimenet.mkdir()

    (egnn / "config.json").write_text(
        json.dumps({"target": "gap", "split": "egnn", "pretrained_ckpt": "runs/single_mol_cdft/best.pt"}),
        encoding="utf-8",
    )
    (dimenet / "config.json").write_text(
        json.dumps({"target": "gap", "split": "dimenet", "pretrained_ckpt": "runs/single_mol_cdft/best.pt"}),
        encoding="utf-8",
    )
    for run_dir in (egnn, dimenet):
        (run_dir / "test_metrics.json").write_text(
            json.dumps({"best_epoch": 5, "train": {"mae": 0.1}, "val": {"mae": 0.2}, "test": {"mae": 0.3}}),
            encoding="utf-8",
        )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="incompatible run metadata"):
        _collect_runs("qm9_pretrain_cdft*")


def test_compare_qm9_tool_reads_yaml_and_plain_train_log(tmp_path, monkeypatch):
    from uni_react.tools.compare_qm9_mae import _collect_runs, _load_epoch_metric_series

    run_dir = tmp_path / "qm9_scratch_gap"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "target: gap",
                "targets: null",
                f"out_dir: {run_dir}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(
        "\n".join(
            [
                "[train] step=1 loss=0.9 mae=0.8",
                "[val] step=1 loss=0.7 mae=0.6",
                "[epoch] step=1 epoch=1 epochs=2 train_loss=0.9 val_loss=0.7 time_sec=1.0 is_best=False",
                "[train] step=2 loss=0.5 mae=0.4",
                "[val] step=2 loss=0.3 mae=0.2",
                "[epoch] step=2 epoch=2 epochs=2 train_loss=0.5 val_loss=0.3 time_sec=1.0 is_best=True",
                "[test] loss=0.25 mae=0.15",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    results = _collect_runs("qm9_scratch*")
    assert results["gap"]["epoch"] == 2
    assert results["gap"]["train_mae"] == pytest.approx(0.4)
    assert results["gap"]["val_mae"] == pytest.approx(0.2)
    assert results["gap"]["test_mae"] == pytest.approx(0.15)

    series = _load_epoch_metric_series(run_dir, split="train", metric="mae")
    assert series[0][0] == 1
    assert series[0][1] == pytest.approx(0.8)
    assert series[1][0] == 2
    assert series[1][1] == pytest.approx(0.4)


def test_compare_qm9_tool_rejects_multi_target_run(tmp_path, monkeypatch):
    from uni_react.tools.compare_qm9_mae import _collect_runs

    run_dir = tmp_path / "qm9_pretrain_cdft_multi"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "target": "gap",
                "targets": ["gap", "homo"],
                "split": "egnn",
                "pretrained_ckpt": "runs/single_mol_cdft/best.pt",
                "out_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"best_epoch": 3, "train": {"mae": 0.1}, "val": {"mae": 0.2}, "test": {"mae": 0.3}}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="single-target runs"):
        _collect_runs("qm9_pretrain*")


def test_compare_qm9_prefix_mode_matches_modern_family_naming(tmp_path, monkeypatch):
    from uni_react.tools.compare_qm9_mae import _collect_runs_by_prefix

    run_dir = tmp_path / "qm9_pretrain_cdft_single_mol_egnn_gap"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "target": "gap",
                "targets": None,
                "split": "egnn",
                "pretrained_ckpt": "runs/single_mol_cdft/best.pt",
                "out_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"best_epoch": 3, "train": {"mae": 0.1}, "val": {"mae": 0.2}, "test": {"mae": 0.3}}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    results = _collect_runs_by_prefix("qm9_pretrain_cdft_single_mol_egnn_")
    assert results["gap"]["run_dir"] == run_dir.name


def test_compare_qm9_help_prefers_glob_workflow(capsys):
    from uni_react.tools.compare_qm9_mae import parse_args
    import sys

    original_argv = sys.argv
    sys.argv = ["compare_qm9_mae.py", "--help"]
    try:
        with pytest.raises(SystemExit):
            parse_args()
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "family-aware" in captured.out
    lowered = captured.out.lower()
    assert "prefer explicit" in lowered
    assert "globs" in lowered


def test_readme_stage3_commands_match_current_reaction_converter_cli():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "--roots" in readme
    assert "--output_h5" in readme
    assert "--output_train_jsonl" in readme
    assert "--output_val_jsonl" in readme
    assert "--input data.h5 --out triplets.h5" not in readme
    assert "--input data.xyz --out triplets.h5" not in readme


def test_convert_ckpt_docs_match_current_cli_flags():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_cn = Path("README_CN.md").read_text(encoding="utf-8")

    for text in (readme, readme_cn):
        assert "--in_ckpt old_checkpoint.pt" in text
        assert "--out_ckpt converted.pt" in text
        assert "--input old_checkpoint.pt" not in text
        assert "--output converted.pt" not in text


def test_density_launcher_derives_default_geometric_out_dir_before_checkpoint_lookup():
    script = Path("scripts/train_pretrain_density.sh").read_text(encoding="utf-8")

    geom_assign = script.index('GEOMETRIC_OUT_DIR="runs/${ENCODER_TYPE}_geometric"')
    init_lookup = script.index('if [[ -z "${INIT_CKPT}" ]]; then')
    best_lookup = script.index('"${GEOMETRIC_OUT_DIR}/best.pt"')

    assert geom_assign < init_lookup
    assert geom_assign < best_lookup


def test_reaction_init_checkpoint_loads_backbone_weights_from_pretrain_checkpoint():
    from uni_react.configs import PretrainConfig, ReactionPretrainConfig
    from uni_react.training.checkpoint import load_init_checkpoint
    from uni_react.training.pretrain_builders import build_pretrain_encoder, build_pretrain_model

    pretrain_cfg = PretrainConfig(
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=128,
        cutoff=3.0,
        num_kernel=16,
    )
    reaction_cfg = ReactionPretrainConfig(
        train_h5="dummy.h5",
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=128,
        cutoff=3.0,
        num_kernel=16,
    )

    pretrain_model = build_pretrain_model(pretrain_cfg, "geometric_structure")
    reaction_encoder = build_pretrain_encoder(reaction_cfg)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "pretrain.pt"
        torch.save({"model": pretrain_model.state_dict()}, ckpt_path)

        before = {k: v.clone() for k, v in reaction_encoder.state_dict().items()}
        load_init_checkpoint(
            model=reaction_encoder,
            ckpt_path=str(ckpt_path),
            device=torch.device("cpu"),
            strict=False,
            rank=0,
            logger=None,
        )

    changed = sum(int(not torch.equal(before[k], reaction_encoder.state_dict()[k])) for k in before)
    assert changed > 0
