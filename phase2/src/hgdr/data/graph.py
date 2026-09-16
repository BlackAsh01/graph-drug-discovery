"""Heterogeneous medical-entity graph construction.

Node types
    ``diag`` (ICD-9 diagnosis codes), ``proc`` (procedure codes) and
    ``drug`` (normalised drug keys).

Edge types (all stored directed; reverse relations are added explicitly)
    * ``('diag', 'treated_by', 'drug')`` – diagnosis/drug co-occurrence in
      **training** admissions (count >= ``min_cooccur``, top-``k`` per
      diagnosis by count), a cleaned-up version of the original notebook's
      ``('diagnosis', 'related_to', 'drug')`` relation.
    * ``('proc', 'followed_by', 'drug')`` – the same for procedures.
    * ``('drug', 'coprescribed', 'drug')`` – EHR co-prescription graph
      (GAMENet-style), from training admissions.
    * ``('drug', 'interacts', 'drug')`` – TWOSIDES DDI graph.

Only training visits are used to build co-occurrence edges so that no
label information from validation/test admissions leaks into the graph.
Molecular graphs are built from SMILES with RDKit and stored per drug so
the model can compute molecular features on the fly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

EdgeKey = Tuple[str, str, str]


# --------------------------------------------------------------------------- #
# Molecular graphs
# --------------------------------------------------------------------------- #
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 11, 12, 14, 19, 20, 26, 30, 1]  # C N O F P S Cl Br I B Na Mg Si K Ca Fe Zn H
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "S", "UNSPECIFIED"]
ATOM_FEAT_DIM = len(ATOM_LIST) + 1 + 6 + 5 + len(HYBRIDIZATIONS) + 1 + 1  # 40


def atom_features(atom) -> List[float]:
    """One-hot atom features: element, degree, formal charge, hybridisation, aromatic, H count.

    Mirrors the five properties used in the original ``smiles_to_graph``
    (atomic number, degree, formal charge, hybridisation, aromaticity),
    but one-hot encoded so that magnitudes are comparable.
    """
    feats = [0.0] * ATOM_FEAT_DIM
    z = atom.GetAtomicNum()
    idx = ATOM_LIST.index(z) if z in ATOM_LIST else len(ATOM_LIST)
    feats[idx] = 1.0
    off = len(ATOM_LIST) + 1
    deg = min(atom.GetDegree(), 5)
    feats[off + deg] = 1.0
    off += 6
    charge = int(np.clip(atom.GetFormalCharge(), -2, 2)) + 2
    feats[off + charge] = 1.0
    off += 5
    hyb = str(atom.GetHybridization())
    feats[off + (HYBRIDIZATIONS.index(hyb) if hyb in HYBRIDIZATIONS else len(HYBRIDIZATIONS) - 1)] = 1.0
    off += len(HYBRIDIZATIONS)
    feats[off] = 1.0 if atom.GetIsAromatic() else 0.0
    feats[off + 1] = min(atom.GetTotalNumHs(), 4) / 4.0
    return feats


def smiles_to_graph(smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert SMILES to ``(x [n_atoms, ATOM_FEAT_DIM], edge_index [2, n_bonds*2])``.

    Returns ``None`` if RDKit cannot parse the string.  Single-atom
    molecules get an empty edge index (they are still pooled correctly).
    """
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("rdkit is required for molecular features") from e
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index


# --------------------------------------------------------------------------- #
# Graph container
# --------------------------------------------------------------------------- #
@dataclass
class HeteroGraph:
    """Serializable heterogeneous graph + molecular side information."""

    num_nodes: Dict[str, int]
    edge_index: Dict[EdgeKey, torch.Tensor]
    ddi_adj: torch.Tensor  # [n_drug, n_drug] float {0,1}
    mol_x: List[Optional[torch.Tensor]] = field(default_factory=list)  # per drug
    mol_edge_index: List[Optional[torch.Tensor]] = field(default_factory=list)
    drug_cid: List[Optional[int]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def node_types(self) -> List[str]:
        return list(self.num_nodes.keys())

    @property
    def edge_types(self) -> List[EdgeKey]:
        return list(self.edge_index.keys())

    def has_mol(self) -> torch.Tensor:
        return torch.tensor([x is not None for x in self.mol_x], dtype=torch.bool)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "num_nodes": self.num_nodes,
            "edge_index": {"|".join(k): v for k, v in self.edge_index.items()},
            "ddi_adj": self.ddi_adj,
            "mol_x": self.mol_x,
            "mol_edge_index": self.mol_edge_index,
            "drug_cid": self.drug_cid,
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "HeteroGraph":
        d = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            num_nodes=d["num_nodes"],
            edge_index={tuple(k.split("|")): v for k, v in d["edge_index"].items()},
            ddi_adj=d["ddi_adj"],
            mol_x=d["mol_x"],
            mol_edge_index=d["mol_edge_index"],
            drug_cid=d["drug_cid"],
            meta=d.get("meta", {}),
        )

    def summary(self) -> str:
        lines = [f"nodes: {self.num_nodes}"]
        for k, v in self.edge_index.items():
            lines.append(f"  {k[0]}-{k[1]}->{k[2]}: {v.shape[1]:,} edges")
        n_mol = int(self.has_mol().sum())
        lines.append(f"drugs with molecular graph: {n_mol}/{len(self.mol_x)}; "
                     f"DDI edges: {int(self.ddi_adj.sum().item() // 2):,}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def _cooccurrence_edges(
    visits: Sequence[Dict[str, object]],
    src_field: str,
    dst_field: str,
    min_count: int,
    top_k: int,
) -> torch.Tensor:
    """Count ``(src, dst)`` co-occurrence within visits; keep frequent, top-k per src."""
    counter: Counter = Counter()
    for v in visits:
        s_list, d_list = v[src_field], v[dst_field]
        if not s_list or not d_list:
            continue
        for s in s_list:
            for d in d_list:
                if src_field == dst_field and s == d:
                    continue
                counter[(s, d)] += 1
    per_src: Dict[int, List[Tuple[int, int]]] = {}
    for (s, d), c in counter.items():
        if c >= min_count:
            per_src.setdefault(s, []).append((c, d))
    src, dst = [], []
    for s, lst in per_src.items():
        lst.sort(reverse=True)
        for _, d in lst[:top_k] if top_k > 0 else lst:
            src.append(s)
            dst.append(d)
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _dense_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    r, c = np.nonzero(adj)
    return torch.tensor(np.stack([r, c]), dtype=torch.long) if len(r) else torch.zeros((2, 0), dtype=torch.long)


def build_hetero_graph(
    train_records: Sequence[Sequence[Dict[str, object]]],
    n_diag: int,
    n_proc: int,
    drug_idx2word: Sequence[str],
    drug_mapping: Dict[str, Dict[str, object]],
    ddi_adj: np.ndarray,
    min_cooccur: int = 5,
    top_k_per_node: int = 20,
    drug_cooccur_min: int = 5,
    drug_cooccur_top_k: int = 20,
    verbose: bool = True,
) -> HeteroGraph:
    """Build the entity graph from **training** visits, DDI matrix and SMILES.

    Args:
        train_records: Patient records of the training split only.
        n_diag / n_proc: Vocabulary sizes.
        drug_idx2word: Drug vocabulary.
        drug_mapping: ``{drug_key: {cid, smiles}}`` dictionary.
        ddi_adj: Dense ``[n_drug, n_drug]`` DDI matrix.
        min_cooccur: Minimum admission co-occurrence for diag/proc->drug edges.
        top_k_per_node: Keep at most this many drug neighbours per diag/proc.
        drug_cooccur_min / drug_cooccur_top_k: Same for the drug-drug
            co-prescription relation.
    """
    visits = [v for p in train_records for v in p]
    n_drug = len(drug_idx2word)

    ei: Dict[EdgeKey, torch.Tensor] = {}
    e = _cooccurrence_edges(visits, "diag", "drug", min_cooccur, top_k_per_node)
    ei[("diag", "treated_by", "drug")] = e
    ei[("drug", "treats", "diag")] = e.flip(0)
    if n_proc > 0:
        e = _cooccurrence_edges(visits, "proc", "drug", min_cooccur, top_k_per_node)
        ei[("proc", "followed_by", "drug")] = e
        ei[("drug", "follows", "proc")] = e.flip(0)
    e = _cooccurrence_edges(visits, "drug", "drug", drug_cooccur_min, drug_cooccur_top_k)
    ei[("drug", "coprescribed", "drug")] = e
    ei[("drug", "interacts", "drug")] = _dense_to_edge_index(np.asarray(ddi_adj))

    mol_x: List[Optional[torch.Tensor]] = []
    mol_ei: List[Optional[torch.Tensor]] = []
    cids: List[Optional[int]] = []
    for w in drug_idx2word:
        m = drug_mapping.get(w)
        g = smiles_to_graph(m["smiles"]) if m and m.get("smiles") else None
        if g is None:
            mol_x.append(None)
            mol_ei.append(None)
        else:
            mol_x.append(g[0])
            mol_ei.append(g[1])
        cids.append(int(m["cid"]) if m and m.get("cid") is not None else None)

    graph = HeteroGraph(
        num_nodes={"diag": int(n_diag), "proc": int(n_proc), "drug": int(n_drug)},
        edge_index=ei,
        ddi_adj=torch.tensor(np.asarray(ddi_adj), dtype=torch.float32),
        mol_x=mol_x,
        mol_edge_index=mol_ei,
        drug_cid=cids,
        meta={
            "min_cooccur": min_cooccur,
            "top_k_per_node": top_k_per_node,
            "drug_cooccur_min": drug_cooccur_min,
            "drug_cooccur_top_k": drug_cooccur_top_k,
            "n_train_visits": len(visits),
            "n_drugs_with_smiles": int(sum(x is not None for x in mol_x)),
            "n_drugs_with_cid": int(sum(c is not None for c in cids)),
        },
    )
    if verbose:
        print("[graph]\n" + graph.summary())
    return graph
