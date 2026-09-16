"""Shared fixtures: a tiny synthetic heterogeneous graph + synthetic patient records.

No MIMIC data is required; everything runs on CPU in well under a minute.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hgdr.data.graph import HeteroGraph, build_hetero_graph, smiles_to_graph  # noqa: E402

N_DIAG, N_PROC, N_DRUG = 30, 8, 12
SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
          "CC(=O)Nc1ccc(O)cc1", "CN(C)CCCN1c2ccccc2CCc2ccccc21", "C1CCNCC1"]


def make_records(n_patients: int = 20, seed: int = 0) -> List[List[Dict[str, object]]]:
    rng = random.Random(seed)
    recs = []
    for sid in range(n_patients):
        visits = []
        for _ in range(rng.choice([1, 1, 2, 3])):
            visits.append({
                "subject_id": 1000 + sid,
                "diag": sorted(rng.sample(range(N_DIAG), rng.randint(2, 6))),
                "proc": sorted(rng.sample(range(N_PROC), rng.randint(0, 3))),
                "drug": sorted(rng.sample(range(N_DRUG), rng.randint(2, 6))),
            })
        recs.append(visits)
    return recs


@pytest.fixture(scope="session")
def records() -> List[List[Dict[str, object]]]:
    return make_records()


@pytest.fixture(scope="session")
def graph(records) -> HeteroGraph:
    drug_words = [f"DRUG{i}" for i in range(N_DRUG)]
    mapping = {f"DRUG{i}": {"cid": 1000 + i, "smiles": SMILES[i % len(SMILES)]} for i in range(N_DRUG - 3)}
    rng = np.random.default_rng(0)
    ddi = (rng.random((N_DRUG, N_DRUG)) < 0.2).astype(np.float32)
    ddi = np.triu(ddi, 1)
    ddi = ddi + ddi.T
    return build_hetero_graph(records, N_DIAG, N_PROC, drug_words, mapping, ddi,
                              min_cooccur=1, top_k_per_node=10, drug_cooccur_min=1, verbose=False)


@pytest.fixture(scope="session")
def base_cfg() -> Dict[str, object]:
    return {
        "name": "test", "seed": 0, "device": "cpu",
        "data": {"processed_dir": "unused", "split_fractions": [0.6, 0.2, 0.2], "split_seed": 0, "max_history": 3},
        "model": {"name": "hgdr", "hidden_dim": 16, "gnn_type": "hetero", "gnn_layers": 2, "conv": "sage",
                  "heads": 2, "use_mol": True, "mol_layers": 1, "mol_conv": "gat", "use_proc": True,
                  "use_history": True, "use_ddi_edges": True, "use_coprescription_edges": True, "dropout": 0.1},
        "train": {"epochs": 2, "patience": 2, "batch_size": 8, "lr": 1e-2, "weight_decay": 0.0,
                  "grad_clip": 1.0, "ddi_weight": 0.05, "margin_weight": 0.0, "threshold": 0.5, "min_epochs": 1},
    }


@pytest.fixture(autouse=True)
def _cpu_threads():
    torch.set_num_threads(2)
