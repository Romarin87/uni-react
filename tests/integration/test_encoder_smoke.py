"""Smoke test: SingleMolEncoder forward pass."""
import importlib.util
import torch
import pytest


@pytest.fixture
def encoder():
    from uni_react.models.single_mol import SingleMolEncoder
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


def test_gotennet_l_encoder_forward():
    if importlib.util.find_spec("torch_cluster") is None:
        pytest.skip("gotennet_l requires torch_cluster")
    from uni_react.models.gotennet_l import GotenNetLEncoder

    encoder = GotenNetLEncoder(
        emb_dim=32,
        num_layers=2,
        heads=4,
        atom_vocab_size=16,
        cutoff=3.0,
        num_rbf=16,
    )
    z = torch.randint(1, 8, (2, 5))
    r = torch.randn(2, 5, 3)
    with torch.no_grad():
        out = encoder(z, r)
    for key in ("node_feats", "node_vec", "node_tensor", "graph_feats", "coords_input", "atom_padding"):
        assert key in out, f"missing key: {key}"
    assert out["node_feats"].shape == (2, 5, 32)
    assert out["node_vec"].shape == (2, 5, 3, 32)
    assert out["node_tensor"].shape == (2, 5, 5, 32)
    assert out["graph_feats"].shape == (2, 32)


def test_gotennet_l_qm9_official_head_forward():
    if importlib.util.find_spec("torch_cluster") is None:
        pytest.skip("gotennet_l requires torch_cluster")
    from uni_react.tasks.qm9.gotennet_l import GotenNetQM9Net, build_gotennet_qm9_metadata
    from uni_react.models.gotennet_l import GotenNetLEncoder

    model = GotenNetQM9Net(
        descriptor=GotenNetLEncoder(
            emb_dim=32,
            num_layers=2,
            heads=4,
            atom_vocab_size=16,
            cutoff=3.0,
            num_rbf=16,
        ),
        target="gap",
        metadata=build_gotennet_qm9_metadata(target="gap"),
    )
    z = torch.randint(1, 8, (2, 5))
    r = torch.randn(2, 5, 3)
    with torch.no_grad():
        out = model(z, r)
    assert "pred" in out
    assert out["pred"].shape == (2,)
    assert out["pred_is_normalized"] is False
