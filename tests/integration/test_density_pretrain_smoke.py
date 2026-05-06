"""Smoke tests for the independent electron_density task components.

Uses the 10-sample HDF5 generated from examples/EDdata.
Run after converting the data::

    python -m uni_react.data.converters.ed \
        --tar_glob 'examples/EDdata/*.tar.gz' \
        --out_dir examples/output/ed_h5 --prefix ed_example --limit 10
"""
from pathlib import Path

import numpy as np
import pytest
import torch

# Path to the converted ED HDF5 (relative to repo root)
_ED_H5 = Path(__file__).parent.parent.parent / "examples" / "output" / "ed_h5" / "ed_example_shard_000000.h5"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ed_h5_path():
    if not _ED_H5.exists():
        pytest.skip(f"ED HDF5 not found at {_ED_H5}. Run the ed.py converter first.")
    return str(_ED_H5)


@pytest.fixture(scope="module")
def tiny_dataset(ed_h5_path):
    from uni_react.tasks.electron_density.dataset import H5ElectronDensityDataset
    return H5ElectronDensityDataset(
        h5_files=[ed_h5_path],
        num_query_points=64,
        center_coords=True,
        deterministic=True,
        seed=0,
    )


@pytest.fixture(scope="module")
def tiny_batch(tiny_dataset):
    from uni_react.tasks.electron_density.dataset import collate_fn_density
    samples = [tiny_dataset[i] for i in range(min(4, len(tiny_dataset)))]
    return collate_fn_density(samples)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestH5DensityDataset:
    def test_len_matches_limit(self, tiny_dataset):
        """Dataset should expose exactly 10 frames (--limit 10 used in conversion)."""
        assert len(tiny_dataset) == 10

    def test_getitem_keys(self, tiny_dataset):
        sample = tiny_dataset[0]
        for key in ("atomic_numbers", "coords", "query_points", "density_target",
                    "total_charge", "spin_multiplicity"):
            assert key in sample, f"Missing key: {key}"

    def test_getitem_shapes(self, tiny_dataset):
        sample = tiny_dataset[0]
        n = sample["atomic_numbers"].shape[0]
        assert n > 0
        assert sample["coords"].shape == (n, 3)
        assert sample["query_points"].shape == (64, 3)
        assert sample["density_target"].shape == (64,)
        assert sample["total_charge"].ndim == 0
        assert sample["spin_multiplicity"].ndim == 0

    def test_dtypes(self, tiny_dataset):
        sample = tiny_dataset[0]
        assert sample["atomic_numbers"].dtype == torch.long
        assert sample["coords"].dtype         == torch.float32
        assert sample["query_points"].dtype   == torch.float32
        assert sample["density_target"].dtype == torch.float32

    def test_deterministic_reproducibility(self, ed_h5_path):
        from uni_react.tasks.electron_density.dataset import H5ElectronDensityDataset
        ds1 = H5ElectronDensityDataset([ed_h5_path], num_query_points=32,
                                        deterministic=True, seed=99)
        ds2 = H5ElectronDensityDataset([ed_h5_path], num_query_points=32,
                                        deterministic=True, seed=99)
        s1 = ds1[0]
        s2 = ds2[0]
        assert torch.equal(s1["query_points"], s2["query_points"])
        assert torch.equal(s1["density_target"], s2["density_target"])

    def test_different_seeds_give_different_samples(self, ed_h5_path):
        from uni_react.tasks.electron_density.dataset import H5ElectronDensityDataset
        ds1 = H5ElectronDensityDataset([ed_h5_path], num_query_points=256,
                                        deterministic=True, seed=0)
        ds2 = H5ElectronDensityDataset([ed_h5_path], num_query_points=256,
                                        deterministic=True, seed=1)
        assert not torch.equal(ds1[0]["query_points"], ds2[0]["query_points"])


# ---------------------------------------------------------------------------
# Collate tests
# ---------------------------------------------------------------------------

class TestCollateDensity:
    def test_batch_keys(self, tiny_batch):
        for key in ("atomic_numbers", "coords", "atom_padding",
                    "query_points", "density_target",
                    "total_charge", "spin_multiplicity"):
            assert key in tiny_batch, f"Missing batch key: {key}"

    def test_batch_shapes(self, tiny_batch):
        B = tiny_batch["atomic_numbers"].shape[0]
        assert B == 4
        N = tiny_batch["atomic_numbers"].shape[1]
        P = tiny_batch["query_points"].shape[1]
        assert P == 64
        assert tiny_batch["coords"].shape         == (B, N, 3)
        assert tiny_batch["atom_padding"].shape   == (B, N)
        assert tiny_batch["density_target"].shape == (B, P)
        assert tiny_batch["total_charge"].shape   == (B,)

    def test_padding_mask_valid(self, tiny_batch):
        """At least one atom per sample should be non-padded."""
        valid = ~tiny_batch["atom_padding"]
        assert valid.any(dim=1).all()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestQueryPointDensityHead:
    def test_output_shape(self):
        from uni_react.tasks.electron_density.head import QueryPointDensityHead
        B, N, D, P = 2, 6, 32, 64
        head = QueryPointDensityHead(emb_dim=D, point_hidden_dim=32,
                                      cond_hidden_dim=16, head_hidden_dim=64)
        node_feats  = torch.randn(B, N, D)
        graph_feats = torch.randn(B, D)
        coords      = torch.randn(B, N, 3)
        padding     = torch.zeros(B, N, dtype=torch.bool)
        query       = torch.randn(B, P, 3)
        charge      = torch.zeros(B)
        spin        = torch.ones(B)
        with torch.no_grad():
            out = head(node_feats, graph_feats, coords, padding, query, charge, spin)
        assert out.shape == (B, P)
        assert torch.isfinite(out).all()

    def test_padding_affects_output(self):
        """Masking out atoms should change the prediction."""
        from uni_react.tasks.electron_density.head import QueryPointDensityHead
        B, N, D, P = 1, 8, 32, 16
        torch.manual_seed(0)
        head = QueryPointDensityHead(emb_dim=D, point_hidden_dim=16,
                                      cond_hidden_dim=8, head_hidden_dim=32)
        node_feats  = torch.randn(B, N, D)
        graph_feats = torch.randn(B, D)
        coords      = torch.randn(B, N, 3)
        query       = torch.randn(B, P, 3)
        charge, spin = torch.zeros(B), torch.ones(B)

        pad_none = torch.zeros(B, N, dtype=torch.bool)
        pad_half = torch.zeros(B, N, dtype=torch.bool)
        pad_half[0, N//2:] = True

        with torch.no_grad():
            out_none = head(node_feats, graph_feats, coords, pad_none, query, charge, spin)
            out_half = head(node_feats, graph_feats, coords, pad_half, query, charge, spin)
        assert not torch.allclose(out_none, out_half)
