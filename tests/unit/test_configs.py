"""Unit tests for the config I/O system."""
import json
from pathlib import Path

import pytest

from uni_react.configs import (
    DensityPretrainConfig,
    FinetuneQM9Config,
    PretrainConfig,
    ReactionPretrainConfig,
)
from uni_react.configs.io import dump_config, load_config, merge_cli_args
from uni_react.training.entrypoint_utils import dump_runtime_config


class TestLoadDumpConfig:
    def test_roundtrip_json(self, tmp_path):
        cfg = PretrainConfig(emb_dim=128, epochs=5)
        path = str(tmp_path / "cfg.json")
        dump_config(cfg, path)
        loaded = load_config(path, PretrainConfig)
        assert loaded.emb_dim == 128
        assert loaded.epochs == 5

    def test_roundtrip_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        cfg = FinetuneQM9Config(batch_size=64, epochs=10)
        path = str(tmp_path / "cfg.yaml")
        dump_config(cfg, path)
        loaded = load_config(path, FinetuneQM9Config)
        assert loaded.batch_size == 64
        assert loaded.epochs == 10

    def test_roundtrip_reaction_restart_fields(self, tmp_path):
        cfg = ReactionPretrainConfig(restart="resume.pt", restart_ignore_config=True)
        path = str(tmp_path / "reaction.json")
        dump_config(cfg, path)
        loaded = load_config(path, ReactionPretrainConfig)
        assert loaded.restart == "resume.pt"
        assert loaded.restart_ignore_config is True
        assert loaded.save_optimizer is True

    def test_roundtrip_density_restart_fields(self, tmp_path):
        cfg = DensityPretrainConfig(restart="density.pt", restart_ignore_config=True)
        path = str(tmp_path / "density.json")
        dump_config(cfg, path)
        loaded = load_config(path, DensityPretrainConfig)
        assert loaded.restart == "density.pt"
        assert loaded.restart_ignore_config is True

    def test_roundtrip_qm9_restart_fields(self, tmp_path):
        cfg = FinetuneQM9Config(restart="qm9.pt", restart_ignore_config=True)
        path = str(tmp_path / "qm9.json")
        dump_config(cfg, path)
        loaded = load_config(path, FinetuneQM9Config)
        assert loaded.restart == "qm9.pt"
        assert loaded.restart_ignore_config is True
        assert loaded.save_optimizer is True

    def test_roundtrip_reaction_save_optimizer_field(self, tmp_path):
        cfg = ReactionPretrainConfig(train_h5="triplets.h5", save_optimizer=False)
        path = str(tmp_path / "reaction_save_opt.json")
        dump_config(cfg, path)
        loaded = load_config(path, ReactionPretrainConfig)
        assert loaded.save_optimizer is False

    def test_roundtrip_qm9_save_optimizer_field(self, tmp_path):
        cfg = FinetuneQM9Config(save_optimizer=False)
        path = str(tmp_path / "qm9_save_opt.json")
        dump_config(cfg, path)
        loaded = load_config(path, FinetuneQM9Config)
        assert loaded.save_optimizer is False

    def test_unknown_key_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        Path(path).write_text(json.dumps({"nonexistent_field": 99}))
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(path, PretrainConfig)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json", PretrainConfig)


class TestMergeCLIArgs:
    def test_override_applies(self):
        cfg = PretrainConfig(emb_dim=256)
        result = merge_cli_args(cfg, {"emb_dim": 128})
        assert result.emb_dim == 128

    def test_none_values_skipped(self):
        cfg = PretrainConfig(emb_dim=256)
        result = merge_cli_args(cfg, {"emb_dim": None})
        assert result.emb_dim == 256

    def test_unknown_keys_ignored(self):
        cfg = PretrainConfig(emb_dim=256)
        result = merge_cli_args(cfg, {"config": "some_file.yaml", "emb_dim": 64})
        assert result.emb_dim == 64

    def test_original_unchanged(self):
        cfg = PretrainConfig(emb_dim=256)
        merge_cli_args(cfg, {"emb_dim": 128})
        assert cfg.emb_dim == 256  # original not mutated


class TestEntrypointUtils:
    def test_dump_runtime_config_prefers_json_when_yaml_unavailable(self, monkeypatch, tmp_path):
        cfg = PretrainConfig()

        def fake_dump_config(config, path):
            path = Path(path)
            if path.suffix == ".yaml":
                raise ImportError("PyYAML missing")
            path.write_text("{}")

        monkeypatch.setattr("uni_react.training.entrypoint_utils.dump_config", fake_dump_config)
        written = dump_runtime_config(cfg, tmp_path)
        assert written.name == "config.json"
        assert written.exists()


def test_hybrid_qm9_encoder_is_valid_in_configs():
    cfg = FinetuneQM9Config(encoder_type="reacformer_hybrid")
    assert cfg.encoder_type == "reacformer_hybrid"
