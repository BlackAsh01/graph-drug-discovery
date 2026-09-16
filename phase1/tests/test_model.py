import pytest
import torch

from dti_gt.data.dataset import collate_pairs
from dti_gt.data.featurize import smiles_to_graph
from dti_gt.models.dti_model import build_model, count_parameters
from dti_gt.utils.config import deep_update

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "[Na+]", "C1CCNCC1"]

BASE_CFG = {
    "model": {"backbone": "transformer", "hidden_dim": 32, "num_layers": 2, "num_heads": 4, "dropout": 0.0,
              "use_edge_features": True, "use_degree_encoding": True, "residual": True, "norm": "batch",
              "fusion": "cross_attention", "fusion_heads": 2, "head_hidden": 16, "head_layers": 2, "head_dropout": 0.0},
    "protein": {"encoder": "pretrained", "dropout": 0.0},
}


def _batch(protein_dim: int, encoder: str):
    graphs = [smiles_to_graph(s) for s in SMILES]
    if encoder == "cnn":
        prots = [torch.randint(0, 26, (50,)) for _ in SMILES]
    else:
        prots = [torch.randn(protein_dim) for _ in SMILES]
    ys = [torch.tensor(11.5) for _ in SMILES]
    return collate_pairs(list(zip(graphs, prots, ys)))


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"model": {"use_edge_features": False}},
        {"model": {"use_degree_encoding": False}},
        {"model": {"residual": False, "norm": "none"}},
        {"model": {"num_layers": 1, "num_heads": 1}},
        {"model": {"fusion": "concat"}},
        {"model": {"fusion": "add"}},
        {"model": {"backbone": "gcn"}},
        {"model": {"backbone": "gat"}},
        {"model": {"backbone": "gine"}},
        {"protein": {"encoder": "cnn"}},
        {"protein": {"encoder": "composition"}},
    ],
)
def test_forward_backward_all_variants(override):
    cfg = deep_update(BASE_CFG, override)
    enc = cfg["protein"]["encoder"]
    protein_dim = {"pretrained": 24, "composition": 26, "cnn": 50}[enc]
    model = build_model(cfg, protein_dim)
    graphs, prots, y = _batch(protein_dim, enc)
    model.train()
    out = model(graphs, prots)
    assert out.shape == (len(SMILES),)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert count_parameters(model) > 0
    model.eval()
    with torch.no_grad():
        assert model(graphs, prots).shape == (len(SMILES),)


def test_edge_features_change_output():
    torch.manual_seed(0)
    cfg = deep_update(BASE_CFG, {"model": {"norm": "none", "dropout": 0.0}})
    model = build_model(cfg, 24).eval()
    graphs, prots, _ = _batch(24, "pretrained")
    with torch.no_grad():
        a = model(graphs, prots)
        graphs.edge_attr = graphs.edge_attr.clone()
        graphs.edge_attr[:, 0] = 4  # relabel every bond as aromatic
        b = model(graphs, prots)
    assert not torch.allclose(a, b)
