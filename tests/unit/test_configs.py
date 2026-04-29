"""Unit tests for the config I/O system."""
import json
from pathlib import Path

import pytest

from uni_react.configs import (
    DensityConfig,
    QM9Config,
    GeometricConfig,
    ReactionConfig,
)
from uni_react.configs.io import dump_config, load_config, merge_cli_args
from uni_react.models import build_qm9_model_spec
from uni_react.tasks.qm9 import resolve_qm9_task_spec
from uni_react.configs import dump_runtime_config
from uni_react.tasks.qm9.dataset import get_qm9_target_index_map


class TestLoadDumpConfig:
    def test_roundtrip_json(self, tmp_path):
        cfg = GeometricConfig(emb_dim=128, epochs=5)
        path = str(tmp_path / "cfg.json")
        dump_config(cfg, path)
        loaded = load_config(path, GeometricConfig)
        assert loaded.emb_dim == 128
        assert loaded.epochs == 5

    def test_roundtrip_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        cfg = QM9Config(batch_size=64, epochs=10)
        path = str(tmp_path / "cfg.yaml")
        dump_config(cfg, path)
        loaded = load_config(path, QM9Config)
        assert loaded.batch_size == 64
        assert loaded.epochs == 10

    def test_roundtrip_reaction_restart_fields(self, tmp_path):
        cfg = ReactionConfig(restart="resume.pt", restart_ignore_config=True)
        path = str(tmp_path / "reaction.json")
        dump_config(cfg, path)
        loaded = load_config(path, ReactionConfig)
        assert loaded.restart == "resume.pt"
        assert loaded.restart_ignore_config is True
        assert loaded.save_optimizer is True

    def test_roundtrip_density_restart_fields(self, tmp_path):
        cfg = DensityConfig(restart="density.pt", restart_ignore_config=True)
        path = str(tmp_path / "density.json")
        dump_config(cfg, path)
        loaded = load_config(path, DensityConfig)
        assert loaded.restart == "density.pt"
        assert loaded.restart_ignore_config is True

    def test_roundtrip_qm9_restart_fields(self, tmp_path):
        cfg = QM9Config(restart="qm9.pt", restart_ignore_config=True)
        path = str(tmp_path / "qm9.json")
        dump_config(cfg, path)
        loaded = load_config(path, QM9Config)
        assert loaded.restart == "qm9.pt"
        assert loaded.restart_ignore_config is True
        assert loaded.save_optimizer is True

    def test_roundtrip_reaction_save_optimizer_field(self, tmp_path):
        cfg = ReactionConfig(train_h5="triplets.h5", save_optimizer=False)
        path = str(tmp_path / "reaction_save_opt.json")
        dump_config(cfg, path)
        loaded = load_config(path, ReactionConfig)
        assert loaded.save_optimizer is False

    def test_roundtrip_qm9_save_optimizer_field(self, tmp_path):
        cfg = QM9Config(save_optimizer=False)
        path = str(tmp_path / "qm9_save_opt.json")
        dump_config(cfg, path)
        loaded = load_config(path, QM9Config)
        assert loaded.save_optimizer is False

    def test_unknown_key_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        Path(path).write_text(json.dumps({"nonexistent_field": 99}))
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(path, GeometricConfig)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json", GeometricConfig)


class TestMergeCLIArgs:
    def test_override_applies(self):
        cfg = GeometricConfig(emb_dim=256)
        result = merge_cli_args(cfg, {"emb_dim": 128})
        assert result.emb_dim == 128

    def test_none_values_skipped(self):
        cfg = GeometricConfig(emb_dim=256)
        result = merge_cli_args(cfg, {"emb_dim": None})
        assert result.emb_dim == 256

    def test_unknown_keys_ignored(self):
        cfg = GeometricConfig(emb_dim=256)
        result = merge_cli_args(cfg, {"config": "some_file.yaml", "emb_dim": 64})
        assert result.emb_dim == 64

    def test_original_unchanged(self):
        cfg = GeometricConfig(emb_dim=256)
        merge_cli_args(cfg, {"emb_dim": 128})
        assert cfg.emb_dim == 256  # original not mutated


class TestEntrypointUtils:
    def test_dump_runtime_config_prefers_json_when_yaml_unavailable(self, monkeypatch, tmp_path):
        cfg = GeometricConfig()

        def fake_dump_config(config, path):
            path = Path(path)
            if path.suffix == ".yaml":
                raise ImportError("PyYAML missing")
            path.write_text("{}")

        monkeypatch.setattr("uni_react.configs.io.dump_config", fake_dump_config)
        written = dump_runtime_config(cfg, tmp_path)
        assert written.name == "config.json"
        assert written.exists()


@pytest.mark.parametrize(
    "model_name",
    [
        "gotennet_s",
        "gotennet_b",
        "gotennet_l",
        "gotennet_s_hat",
        "gotennet_b_hat",
        "gotennet_l_hat",
    ],
)
def test_gotennet_qm9_models_are_valid_in_configs(model_name):
    cfg = QM9Config(model_name=model_name)
    assert cfg.model_name == model_name


def test_gotennet_qm9_split_and_target_variant_are_valid_in_configs():
    cfg = QM9Config(
        model_name="gotennet_l",
        split="gotennet",
        qm9_target_variant="gotennet",
        lr_factor=0.8,
        lr_patience=15,
        lr_min=1e-7,
        early_stopping_patience=150,
    )
    assert cfg.split == "gotennet"
    assert cfg.qm9_target_variant == "gotennet"
    assert cfg.lr_factor == pytest.approx(0.8)


def test_gotennet_qm9_target_index_variant_uses_official_energy_columns():
    mapping = get_qm9_target_index_map("gotennet")
    assert mapping["U0"] == 7
    assert mapping["U"] == 8
    assert mapping["H"] == 9
    assert mapping["G"] == 10


def test_qm9_task_variant_defaults_to_default_for_non_gotennet():
    cfg = QM9Config(model_name="single_mol")
    spec = resolve_qm9_task_spec(cfg)
    assert spec.variant == "default"
    assert spec.split == "egnn"
    assert spec.target_index_variant == "default"


@pytest.mark.parametrize("model_name", ["gotennet_s", "gotennet_b", "gotennet_l", "gotennet_s_hat", "gotennet_b_hat", "gotennet_l_hat"])
def test_qm9_task_variant_defaults_to_gotennet_for_gotennet_variants(model_name):
    cfg = QM9Config(model_name=model_name)
    spec = resolve_qm9_task_spec(cfg)
    assert spec.variant == "gotennet"
    assert spec.split == "gotennet"
    assert spec.target_index_variant == "gotennet"
    assert spec.center_coords is False


def test_qm9_model_spec_rejects_unsupported_variant():
    with pytest.raises(ValueError, match="does not support QM9 variant"):
        build_qm9_model_spec("gotennet_l", "default")


@pytest.mark.parametrize("model_name", ["gotennet_s", "gotennet_b", "gotennet_l", "gotennet_s_hat", "gotennet_b_hat", "gotennet_l_hat"])
def test_qm9_model_spec_accepts_gotennet_variants(model_name):
    spec = build_qm9_model_spec(model_name, "gotennet")
    assert spec.name == model_name
    assert callable(spec.build_backbone)


def test_qm9_model_spec_accepts_single_mol_default_variant():
    spec = build_qm9_model_spec("single_mol", "default")
    assert spec.name == "single_mol"
    assert callable(spec.build_backbone)
