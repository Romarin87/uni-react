"""Smoke test: SingleMolEncoder forward pass."""
import torch
import pytest


@pytest.fixture
def encoder():
    from uni_react.encoders import SingleMolEncoder
    return SingleMolEncoder(
        emb_dim=32, inv_layer=1, se3_layer=1, heads=4,
        atom_vocab_size=16, cutoff=3.0, num_kernel=16,
    )


def test_output_keys(encoder):
    B, N = 2, 5
    z = torch.randint(1, 8, (B, N))
    r = torch.randn(B, N, 3)
    with torch.no_grad():
        out = encoder(z, r)
    for key in ("node_feats", "node_vec", "graph_feats", "atom_padding"):
        assert key in out, f"missing key: {key}"


def test_output_shapes(encoder):
    B, N, D = 2, 5, 32
    z = torch.randint(1, 8, (B, N))
    r = torch.randn(B, N, 3)
    with torch.no_grad():
        out = encoder(z, r)
    assert out["node_feats"].shape  == (B, N, D)
    assert out["graph_feats"].shape == (B, D)
    assert out["node_vec"].shape[:3] == (B, N, 3)


def test_padding_mask_respected(encoder):
    B, N = 1, 6
    z   = torch.randint(1, 8, (B, N))
    r   = torch.randn(B, N, 3)
    pad = torch.tensor([[False, False, False, True, True, True]])
    with torch.no_grad():
        out = encoder(z, r, atom_padding=pad)
    # Padded positions should be zeroed out
    assert out["node_feats"][0, 3:].abs().sum() == 0.0
