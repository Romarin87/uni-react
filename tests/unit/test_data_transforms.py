"""Unit tests for data transforms."""
import pytest
import torch
import numpy as np

from uni_react.tasks.geometric.common.transforms import (
    AddGaussianNoise,
    CenterCoords,
    Compose,
    MaskAtoms,
)


class TestCenterCoords:
    def test_centers_mean(self):
        coords = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        sample = {"coords": coords}
        out = CenterCoords()(sample)
        assert abs(float(out["coords"].mean(0)[0])) < 1e-5

    def test_preserves_relative_positions(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        sample = {"coords": coords}
        out = CenterCoords()(sample)
        diff = out["coords"][1] - out["coords"][0]
        assert abs(float(diff[0]) - 2.0) < 1e-5


class TestAddGaussianNoise:
    def test_output_keys(self):
        coords = torch.zeros(5, 3)
        sample = {"coords": coords}
        out = AddGaussianNoise(std=0.1)(sample)
        assert "coords_noisy" in out
        assert "coords_target" in out

    def test_target_equals_original(self):
        coords = torch.ones(4, 3)
        sample = {"coords": coords}
        out = AddGaussianNoise(std=0.1)(sample)
        assert torch.allclose(out["coords_target"], coords)

    def test_noisy_differs_from_original(self):
        torch.manual_seed(0)
        coords = torch.ones(100, 3)
        sample = {"coords": coords}
        out = AddGaussianNoise(std=1.0)(sample)
        assert not torch.allclose(out["coords_noisy"], coords)


class TestMaskAtoms:
    def test_mask_ratio_applied(self):
        atomic = torch.arange(1, 21)  # 20 atoms
        sample = {"atomic_numbers": atomic}
        out = MaskAtoms(ratio=0.2, mask_token_id=0, min_masked=1)(sample)
        n_masked = int(out["mask"].sum().item())
        # expect ~4 masked, allow ±1
        assert 1 <= n_masked <= 20

    def test_mask_token_replaces_atoms(self):
        atomic = torch.arange(1, 11)
        sample = {"atomic_numbers": atomic}
        out = MaskAtoms(ratio=1.0, mask_token_id=99, min_masked=1)(sample)
        masked_idx = out["mask"]
        assert (out["input_atomic_numbers"][masked_idx] == 99).all()

    def test_target_unchanged(self):
        atomic = torch.arange(1, 6)
        sample = {"atomic_numbers": atomic}
        out = MaskAtoms(ratio=0.5, mask_token_id=0)(sample)
        assert torch.equal(out["target_atomic_numbers"], atomic)


class TestCompose:
    def test_applies_in_order(self):
        calls = []

        class RecordTransform:
            def __init__(self, name):
                self.name = name
            def __call__(self, sample):
                calls.append(self.name)
                return sample

        t = Compose([RecordTransform("a"), RecordTransform("b")])
        t({})
        assert calls == ["a", "b"]
