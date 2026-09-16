"""End-to-end: two training epochs on the synthetic graph via the public training engine."""

import copy

import torch

from hgdr.data.dataset import VisitDataset, make_collate
from hgdr.data.splits import split_records
from hgdr.models import build_model
from hgdr.train import evaluate, train_model
from hgdr.utils.logging import get_logger
from hgdr.utils.seed import set_seed
from tests.conftest import N_DRUG
from torch.utils.data import DataLoader


def test_train_two_epochs_and_evaluate(records, graph, base_cfg, tmp_path):
    set_seed(0)
    cfg = copy.deepcopy(base_cfg)
    train, val, test = split_records(records, cfg["data"]["split_fractions"], cfg["data"]["split_seed"])
    collate = make_collate(N_DRUG)
    mk = lambda r, s: DataLoader(VisitDataset(r, 3), batch_size=8, shuffle=s, collate_fn=collate)  # noqa: E731
    loaders = (mk(train, True), mk(val, False), mk(test, False))
    model = build_model(cfg, graph)
    log = get_logger("test", tmp_path / "train.log")
    model, history = train_model(cfg, model, loaders, graph.ddi_adj, torch.device("cpu"), tmp_path, log)
    assert len(history) == 2 and (tmp_path / "best.pt").exists()
    m = evaluate(model, loaders[2], torch.device("cpu"), graph.ddi_adj.numpy())
    for k in ("jaccard", "prauc", "f1", "ddi_rate", "avg_drugs", "precision@10"):
        assert k in m and 0.0 <= m[k] or k == "avg_drugs"


def test_seed_reproducibility(records, graph, base_cfg):
    def run():
        set_seed(123)
        model = build_model(base_cfg, graph)
        batch = next(iter(DataLoader(VisitDataset(records, 3), batch_size=8, collate_fn=make_collate(N_DRUG))))
        return model(batch).detach()
    assert torch.allclose(run(), run())
