"""Smoke tests for ReacFormer-SE3 and ReacFormer-SO2 encoders.

Tests cover:
1. Forward pass shape correctness
2. Output key presence
3. SE(3) equivariance (rotation test)
4. Invariance of scalar/graph features under rotation
5. Basic gradient flow
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CFG = dict(
    emb_dim=32, num_layers=2, heads=4,
    atom_vocab_size=16, cutoff=3.0, num_rbf=16,
    path_dropout=0.0, attn_dropout=0.0, activation_dropout=0.0,
    legendre_order=2,
)


@pytest.fixture(scope="module")
def se3_encoder():
    from uni_react.encoders.reacformer_se3 import ReacFormerSE3Encoder
    return ReacFormerSE3Encoder(**TINY_CFG).eval()


@pytest.fixture(scope="module")
def so2_encoder():
    from uni_react.encoders.reacformer_so2 import ReacFormerSO2Encoder
    return ReacFormerSO2Encoder(**TINY_CFG, l_max=2).eval()


@pytest.fixture(scope="module")
def tiny_batch():
    torch.manual_seed(0)
    B, N = 2, 6
    return {
        "z":   torch.randint(1, 10, (B, N)),
        "r":   torch.randn(B, N, 3),
        "pad": torch.zeros(B, N, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Helper: random SO(3) rotation matrix
# ---------------------------------------------------------------------------

def random_rotation(seed: int = 42) -> Tensor:
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(4), dim=0)
    w, x, y, z = q
    R = torch.tensor([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return R  # (3,3) orthogonal


# ---------------------------------------------------------------------------
# Tests for both encoders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enc_fixture", ["se3_encoder", "so2_encoder"])
class TestReacFormerForward:

    def test_output_keys(self, request, enc_fixture, tiny_batch):
        enc = request.getfixturevalue(enc_fixture)
        with torch.no_grad():
            out = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=tiny_batch["r"],
                atom_padding=tiny_batch["pad"],
            )
        for key in ("node_feats", "node_vec", "node_tensor",
                    "graph_feats", "coords_input", "atom_padding"):
            assert key in out, f"Missing key: {key}"

    def test_output_shapes(self, request, enc_fixture, tiny_batch):
        enc = request.getfixturevalue(enc_fixture)
        B, N, D = 2, 6, TINY_CFG["emb_dim"]
        with torch.no_grad():
            out = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=tiny_batch["r"],
                atom_padding=tiny_batch["pad"],
            )
        assert out["node_feats"].shape  == (B, N, D)
        assert out["node_vec"].shape    == (B, N, 3, D)
        assert out["node_tensor"].shape == (B, N, 5, D)
        assert out["graph_feats"].shape == (B, D)

    def test_all_finite(self, request, enc_fixture, tiny_batch):
        enc = request.getfixturevalue(enc_fixture)
        with torch.no_grad():
            out = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=tiny_batch["r"],
                atom_padding=tiny_batch["pad"],
            )
        for key in ("node_feats", "node_vec", "node_tensor", "graph_feats"):
            assert torch.isfinite(out[key]).all(), f"{key} contains NaN/Inf"

    def test_scalar_rotation_invariance(self, request, enc_fixture, tiny_batch):
        """Graph-level scalar features should be invariant under global rotation."""
        enc = request.getfixturevalue(enc_fixture)
        R = random_rotation(seed=7)
        r_rot = tiny_batch["r"] @ R.T  # apply rotation

        with torch.no_grad():
            out_orig = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=tiny_batch["r"],
                atom_padding=tiny_batch["pad"],
            )
            out_rot = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=r_rot,
                atom_padding=tiny_batch["pad"],
            )

        # Graph-level scalars must be invariant (float32 tolerance)
        assert torch.allclose(
            out_orig["graph_feats"], out_rot["graph_feats"], atol=5e-3
        ), "graph_feats not rotation-invariant"

        # Node-level scalars must also be invariant (per-atom, higher variance than graph)
        assert torch.allclose(
            out_orig["node_feats"], out_rot["node_feats"], atol=2e-2
        ), "node_feats not rotation-invariant"

    def test_vector_rotation_equivariance(self, request, enc_fixture, tiny_batch):
        """Node vectors should rotate with the input coordinates."""
        enc = request.getfixturevalue(enc_fixture)
        R = random_rotation(seed=13)
        r_rot = tiny_batch["r"] @ R.T

        with torch.no_grad():
            out_orig = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=tiny_batch["r"],
                atom_padding=tiny_batch["pad"],
            )
            out_rot = enc(
                input_atomic_numbers=tiny_batch["z"],
                coords_noisy=r_rot,
                atom_padding=tiny_batch["pad"],
            )

        # v_rot should equal R @ v_orig (equivariance)
        # node_vec: (B,N,3,D); apply R to spatial dim
        v_orig = out_orig["node_vec"]   # (B,N,3,D)
        v_rot  = out_rot["node_vec"]
        # R @ v: einsum over spatial axis
        # v_orig: (B,N,3,D); apply R to spatial (3) dim only
        # v_expected[b,n,p,d] = Σ_q R[p,q] * v_orig[b,n,q,d]
        v_expected = torch.einsum("pq,bnqd->bnpd", R, v_orig)
        assert torch.allclose(v_expected, v_rot, atol=5e-3), \
            "node_vec not SE(3)-equivariant (float32 tolerance)"

    def test_gradient_flow(self, request, enc_fixture, tiny_batch):
        """Loss backward should propagate gradients to embedding weights."""
        enc = request.getfixturevalue(enc_fixture)
        enc.train()
        out = enc(
            input_atomic_numbers=tiny_batch["z"],
            coords_noisy=tiny_batch["r"],
            atom_padding=tiny_batch["pad"],
        )
        loss = out["graph_feats"].sum()
        loss.backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        assert torch.isfinite(torch.stack([g.norm() for g in grads])).all()
        enc.eval()


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------

def test_registry_build_se3():
    from uni_react.registry import ENCODER_REGISTRY
    enc = ENCODER_REGISTRY.build({
        "type": "reacformer_se3",
        "emb_dim": 16, "num_layers": 1, "heads": 4,
        "atom_vocab_size": 16, "cutoff": 3.0, "num_rbf": 16,
    })
    assert enc is not None
    assert "reacformer_se3" in ENCODER_REGISTRY


def test_registry_build_so2():
    from uni_react.registry import ENCODER_REGISTRY
    enc = ENCODER_REGISTRY.build({
        "type": "reacformer_so2",
        "emb_dim": 16, "num_layers": 1, "heads": 4,
        "atom_vocab_size": 16, "cutoff": 3.0, "num_rbf": 16, "l_max": 2,
    })
    assert enc is not None
    assert "reacformer_so2" in ENCODER_REGISTRY
