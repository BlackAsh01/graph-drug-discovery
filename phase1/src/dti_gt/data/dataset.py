"""PyTorch datasets / loaders for drug-target pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from dti_gt.data.featurize import featurize_drugs
from dti_gt.data.kiba import load_split, resolve_kiba, split_name
from dti_gt.data.proteins import build_protein_features


class KIBAPairDataset(Dataset):
    """Drug-target pairs with pre-featurised drug graphs and protein tensors.

    Args:
        pairs: DataFrame with ``drug_id, target_id, y`` (only the rows in ``indices`` are used).
        drug_graphs: mapping ``drug_id -> Data``.
        protein_feats: mapping ``target_id -> Tensor`` (encoder specific).
        indices: row indices of ``pairs`` to expose.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        drug_graphs: Dict[str, Data],
        protein_feats: Dict[str, torch.Tensor],
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if indices is None:
            indices = np.arange(len(pairs))
        indices = np.asarray(indices, dtype=np.int64)
        sub = pairs.iloc[indices]
        keep = sub["drug_id"].isin(drug_graphs.keys()) & sub["target_id"].isin(protein_feats.keys())
        dropped = int((~keep).sum())
        if dropped:
            print(f"[dataset] dropping {dropped} pairs whose drug/target could not be featurised")
        sub = sub[keep]
        self.drug_ids: List[str] = sub["drug_id"].tolist()
        self.target_ids: List[str] = sub["target_id"].tolist()
        self.y = torch.tensor(sub["y"].to_numpy(), dtype=torch.float)
        self.row_index = torch.tensor(indices[keep.to_numpy()], dtype=torch.long)
        self.drug_graphs = drug_graphs
        self.protein_feats = protein_feats

    def __len__(self) -> int:
        return len(self.drug_ids)

    def __getitem__(self, i: int) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        return self.drug_graphs[self.drug_ids[i]], self.protein_feats[self.target_ids[i]], self.y[i]


def collate_pairs(batch: Sequence[Tuple[Data, torch.Tensor, torch.Tensor]]) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
    """Collate ``(graph, protein, y)`` triples into ``(Batch, [B, ...], [B])``."""
    graphs, prots, ys = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(prots), torch.stack(ys)


def load_or_build_drug_graphs(df: pd.DataFrame, cache_file: Optional[Path]) -> Dict[str, Data]:
    """Featurise all unique drugs, caching to ``cache_file`` (``torch.save``)."""
    if cache_file is not None and cache_file.is_file():
        return torch.load(cache_file, weights_only=False)
    smiles_by_id = df.drop_duplicates("drug_id").set_index("drug_id")["smiles"].to_dict()
    graphs = featurize_drugs(smiles_by_id)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, cache_file)
    return graphs


def build_dataloaders(cfg: dict, root: Path) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    """Create train/val/test loaders from the ``data`` and ``protein`` sections of a config.

    Returns:
        ``(loaders, info)`` where ``info`` contains dataset statistics and the resolved split.
    """
    dcfg, pcfg = cfg["data"], cfg["protein"]
    data_dir = root / dcfg.get("data_dir", "data")
    df = resolve_kiba(data_dir, dcfg.get("pairs_file"))

    split_dir = data_dir / "splits" / (dcfg.get("split_name") or split_name(int(dcfg["split_seed"]), float(dcfg["subset_fraction"])))
    split = load_split(split_dir)

    cache = data_dir / "processed" / "drug_graphs.pt"
    drug_graphs = load_or_build_drug_graphs(df, cache)
    sequences = df.drop_duplicates("target_id").set_index("target_id")["sequence"].to_dict()
    embedding_file = pcfg.get("embedding_file")
    protein_feats = build_protein_features(
        pcfg["encoder"], sequences, pcfg.get("max_len", 1000), root / embedding_file if embedding_file else None
    )

    loaders: Dict[str, DataLoader] = {}
    sizes = {}
    g = torch.Generator()
    g.manual_seed(int(cfg["training"]["seed"]))
    for name, idx in split.items():
        ds = KIBAPairDataset(df, drug_graphs, protein_feats, idx)
        sizes[name] = len(ds)
        loaders[name] = DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=(name == "train"),
            collate_fn=collate_pairs,
            num_workers=int(dcfg.get("num_workers", 0)),
            generator=g if name == "train" else None,
            drop_last=False,
        )
    info = {
        "split_dir": str(split_dir),
        "sizes": sizes,
        "n_drugs": len(drug_graphs),
        "n_targets": len(protein_feats),
        "protein_input_dim": int(next(iter(protein_feats.values())).shape[-1]),
    }
    return loaders, info
