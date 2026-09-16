"""End-to-end smoke test on the 200-row sample shipped in data/sample (CPU, < 1 min)."""

import json
from pathlib import Path

import numpy as np
import torch

from dti_gt.data.dataset import KIBAPairDataset, build_dataloaders, collate_pairs, load_or_build_drug_graphs
from dti_gt.data.kiba import load_kiba_table, make_split, save_split, split_name, write_compact_kiba
from dti_gt.data.proteins import build_protein_features, load_protein_embeddings, save_protein_embeddings
from dti_gt.train import train_from_config
from dti_gt.utils.config import load_config


def test_sample_table_and_split(sample_table: Path):
    df = load_kiba_table(sample_table)
    assert len(df) == 200 and list(df.columns) == ["drug_id", "smiles", "target_id", "sequence", "y"]
    split = make_split(df, seed=1, subset_fraction=1.0)
    idx = np.concatenate(list(split.values()))
    assert len(idx) == 200 and len(np.unique(idx)) == 200
    sub = make_split(df, seed=1, subset_fraction=0.5)
    assert 90 <= sum(len(v) for v in sub.values()) <= 110


def test_protein_embedding_roundtrip(tmp_path: Path):
    ids = ["P1", "P2"]
    emb = np.random.default_rng(0).normal(size=(2, 8)).astype(np.float32)
    f = tmp_path / "e.npz"
    save_protein_embeddings(f, ids, emb, "dummy")
    index, mat = load_protein_embeddings(f)
    assert index == {"P1": 0, "P2": 1} and torch.allclose(mat, torch.from_numpy(emb))


def test_dataset_and_training_smoke(sample_table: Path, tmp_path: Path, root: Path):
    df = load_kiba_table(sample_table)
    data_dir = tmp_path / "data"
    write_compact_kiba(df, data_dir / "kiba")
    split = make_split(df, seed=7, subset_fraction=1.0)
    save_split(split, data_dir / "splits" / split_name(7, 1.0))

    # dummy pretrained embeddings so the test does not depend on ProtBERT
    targets = df["target_id"].unique().tolist()
    emb = np.random.default_rng(0).normal(size=(len(targets), 16)).astype(np.float32)
    save_protein_embeddings(tmp_path / "emb.npz", targets, emb, "dummy")

    cfg = load_config(root / "configs" / "default.yaml")
    cfg["data"].update({"data_dir": str(data_dir), "split_seed": 7, "subset_fraction": 1.0})
    cfg["protein"].update({"embedding_file": str(tmp_path / "emb.npz")})
    cfg["model"].update({"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "head_hidden": 16, "head_layers": 1})
    cfg["training"].update({"epochs": 2, "batch_size": 32, "device": "cpu", "early_stopping_patience": 5})
    cfg["results_dir"] = str(tmp_path / "results")

    loaders, info = build_dataloaders(cfg, root)
    assert set(loaders) == {"train", "val", "test"} and info["protein_input_dim"] == 16
    graphs, prots, y = next(iter(loaders["train"]))
    assert prots.shape[1] == 16 and y.ndim == 1

    result = train_from_config(cfg, root, run_name="smoke")
    run_dir = tmp_path / "results" / "runs" / "smoke"
    assert (run_dir / "metrics.json").is_file() and (run_dir / "history.csv").is_file()
    m = json.loads((run_dir / "metrics.json").read_text())
    assert np.isfinite(m["test"]["mse"]) and m["epochs_run"] == 2 and result["test"]["mse"] == m["test"]["mse"]
