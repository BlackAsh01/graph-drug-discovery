"""Model forward passes on the synthetic graph (all ablation flags), losses, baselines."""

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from hgdr.data.dataset import VisitDataset, make_collate
from hgdr.models import build_model, recommendation_loss
from hgdr.models.losses import ddi_loss
from tests.conftest import N_DRUG


def _loader(records, n_drug=N_DRUG, bs=8):
    return DataLoader(VisitDataset(records, max_history=3), batch_size=bs, collate_fn=make_collate(n_drug))


@pytest.mark.parametrize("override", [
    {},
    {"use_mol": False},
    {"gnn_type": "homo"},
    {"use_proc": False},
    {"use_history": False},
    {"use_ddi_edges": False},
    {"gnn_layers": 0},
    {"conv": "gat", "gnn_layers": 1},
])
def test_hgdr_forward_backward(records, graph, base_cfg, override):
    cfg = copy.deepcopy(base_cfg)
    cfg["model"].update(override)
    model = build_model(cfg, graph)
    batch = next(iter(_loader(records)))
    logits = model(batch)
    assert logits.shape == (batch["y"].shape[0], N_DRUG)
    assert torch.isfinite(logits).all()
    losses = recommendation_loss(logits, batch["y"], graph.ddi_adj, ddi_weight=0.1, margin_weight=0.1)
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed"


def test_encode_graph_shapes(graph, base_cfg):
    model = build_model(base_cfg, graph)
    nodes = model.encode_graph()
    for t, n in graph.num_nodes.items():
        assert nodes[t].shape == (n, base_cfg["model"]["hidden_dim"])


@pytest.mark.parametrize("name", ["mlp", "lr", "nearest"])
def test_baselines_forward(records, graph, base_cfg, name):
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["name"] = name
    model = build_model(cfg, graph, drug_prior=torch.full((N_DRUG,), 0.3))
    batch = next(iter(_loader(records)))
    logits = model(batch)
    assert logits.shape == (batch["y"].shape[0], N_DRUG)
    assert torch.isfinite(logits).all()


def test_ddi_loss_prefers_safe_sets(graph):
    adj = graph.ddi_adj
    i, j = torch.nonzero(adj, as_tuple=False)[0]
    unsafe = torch.full((1, N_DRUG), -6.0)
    unsafe[0, i] = unsafe[0, j] = 6.0
    safe = torch.full((1, N_DRUG), -6.0)
    safe[0, i] = 6.0
    assert ddi_loss(unsafe, adj) > ddi_loss(safe, adj)
