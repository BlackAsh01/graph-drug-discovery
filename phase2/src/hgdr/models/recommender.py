"""HGDR: heterogeneous-graph, DDI-aware medication recommender.

This is the **released EHR+DDI subset** of the official GRAND figure
(diagnose / medication channels, no genome / M–G–M, no Dual-attention
Graph Transformer + InfoNCE).  ``AttentionPool`` reuses the thesis
semantic-fusion MLP shape (Linear → Tanh → Linear) as *admission-code*
pooling, not HAN attention over meta-path embeddings Z^(m), Z^(p).

Forward pass (per mini-batch of admissions):

1. **Node initialisation** – learned ID embeddings for diagnoses,
   procedures and drugs; drug embeddings are optionally enriched with
   molecular features from :class:`MolEncoder` (SMILES graphs).
2. **Graph encoder** – :class:`HeteroGNN` (relation-aware) or
   :class:`HomoGNN` (type-agnostic ablation) or none (``num_layers=0``)
   refines node embeddings over the entity graph.
3. **Visit encoder** – attention pooling over the admission's diagnosis
   and procedure node embeddings.
4. **History encoder** (optional) – GRU over the patient's previous
   admissions (their visit vectors + pooled prescribed-drug vectors).
5. **Scoring** – bilinear score between the visit vector and every
   (graph-refined) drug embedding plus a per-drug bias -> logits.

Every component can be switched off through the config to run the
ablation study.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.graph import ATOM_FEAT_DIM, HeteroGraph
from .hetero_gnn import HeteroGNN, HomoGNN
from .mol_encoder import MolEncoder, pack_molecules

EdgeKey = Tuple[str, str, str]


class AttentionPool(nn.Module):
    """Masked additive attention pooling over a set of code embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x: [..., L, d]``, ``mask: [..., L]`` -> ``[..., d]`` (zeros where the set is empty)."""
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)  # rows with no valid element
        return (w.unsqueeze(-1) * x).sum(-2)


class HGDRecommender(nn.Module):
    """See module docstring.

    Args:
        graph: The :class:`HeteroGraph` (used for sizes and molecule packing).
        dim: Embedding / hidden size.
        gnn_layers: Number of graph layers (0 disables the graph encoder).
        gnn_type: ``'hetero'`` or ``'homo'``.
        conv: ``'sage'`` or ``'gat'`` message function.
        heads: Attention heads for GAT variants.
        use_mol: Use molecular (SMILES) features for drug nodes.
        mol_layers: Layers of the molecular encoder.
        use_proc: Use the procedure node type.
        use_history: Encode previous admissions with a GRU.
        use_ddi_edges: Keep the ``('drug','interacts','drug')`` relation.
        use_coprescription_edges: Keep the EHR co-prescription relation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        dim: int = 64,
        gnn_layers: int = 2,
        gnn_type: str = "hetero",
        conv: str = "sage",
        heads: int = 4,
        use_mol: bool = True,
        mol_layers: int = 2,
        mol_conv: str = "gat",
        use_proc: bool = True,
        use_history: bool = True,
        use_ddi_edges: bool = True,
        use_coprescription_edges: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.n_diag = graph.num_nodes["diag"]
        self.n_proc = graph.num_nodes.get("proc", 0)
        self.n_drug = graph.num_nodes["drug"]
        self.use_proc = use_proc and self.n_proc > 0
        self.use_history = use_history
        self.use_mol = use_mol
        self.dropout = dropout

        # --- node embeddings --------------------------------------------------
        self.diag_emb = nn.Embedding(self.n_diag, dim)
        self.proc_emb = nn.Embedding(max(1, self.n_proc), dim)
        self.drug_emb = nn.Embedding(self.n_drug, dim)
        for e in (self.diag_emb, self.proc_emb, self.drug_emb):
            nn.init.normal_(e.weight, std=0.1)

        # --- molecular features -----------------------------------------------
        if use_mol:
            x, ei, batch, has = pack_molecules(graph.mol_x, graph.mol_edge_index, ATOM_FEAT_DIM)
            self.register_buffer("mol_x", x)
            self.register_buffer("mol_edge_index", ei)
            self.register_buffer("mol_batch", batch)
            self.register_buffer("mol_has", has)
            self.mol_encoder = MolEncoder(ATOM_FEAT_DIM, dim, num_layers=mol_layers, conv=mol_conv,
                                          heads=heads, dropout=dropout)
            self.mol_gate = nn.Linear(2 * dim, dim)

        # --- graph -------------------------------------------------------------
        keep: Dict[EdgeKey, torch.Tensor] = {}
        for et, ei in graph.edge_index.items():
            s, rel, d = et
            if not self.use_proc and ("proc" in (s, d)):
                continue
            if not use_ddi_edges and rel == "interacts":
                continue
            if not use_coprescription_edges and rel == "coprescribed":
                continue
            keep[et] = ei
        self.edge_types = list(keep.keys())
        for i, (et, ei) in enumerate(keep.items()):
            self.register_buffer(f"edge_index_{i}", ei)
        node_types = ["diag", "drug"] + (["proc"] if self.use_proc else [])
        self.node_types = node_types
        self.gnn: Optional[nn.Module] = None
        if gnn_layers > 0:
            if gnn_type == "hetero":
                self.gnn = HeteroGNN(node_types, self.edge_types, dim, gnn_layers, conv, heads, dropout)
            elif gnn_type == "homo":
                self.gnn = HomoGNN(node_types, dim, gnn_layers, conv, heads, dropout)
            else:
                raise ValueError(f"unknown gnn_type '{gnn_type}'")

        # --- visit / history encoders ------------------------------------------
        self.pool_diag = AttentionPool(dim)
        self.pool_proc = AttentionPool(dim)
        self.visit_proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(dim))
        if use_history:
            self.pool_drug = AttentionPool(dim)
            self.hist_gru = nn.GRU(2 * dim, dim, batch_first=True)
            self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(dim))

        # --- scoring -----------------------------------------------------------
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.drug_bias = nn.Parameter(torch.zeros(self.n_drug))

    # ------------------------------------------------------------------------ #
    def _edge_index_dict(self) -> Dict[EdgeKey, torch.Tensor]:
        return {et: getattr(self, f"edge_index_{i}") for i, et in enumerate(self.edge_types)}

    def encode_graph(self) -> Dict[str, torch.Tensor]:
        """Return refined node embeddings ``{'diag','proc','drug'}``."""
        drug = self.drug_emb.weight
        if self.use_mol:
            mol = self.mol_encoder(self.mol_x, self.mol_edge_index, self.mol_batch, self.n_drug, self.mol_has)
            gate = torch.sigmoid(self.mol_gate(torch.cat([drug, mol], -1)))
            drug = drug + gate * mol
        x = {"diag": self.diag_emb.weight, "drug": drug}
        if self.use_proc:
            x["proc"] = self.proc_emb.weight
        if self.gnn is not None:
            x = self.gnn(x, self._edge_index_dict())
        return x

    def _visit_vector(self, nodes: Dict[str, torch.Tensor], diag: torch.Tensor, diag_mask: torch.Tensor,
                      proc: torch.Tensor, proc_mask: torch.Tensor) -> torch.Tensor:
        # F.embedding instead of tensor[idx]: identical forward, but its backward uses the
        # fused embedding kernel instead of index_put(accumulate=True), which was ~10x slower.
        hd = self.pool_diag(F.embedding(diag, nodes["diag"]), diag_mask)
        if self.use_proc:
            hp = self.pool_proc(F.embedding(proc, nodes["proc"]), proc_mask)
        else:
            hp = torch.zeros_like(hd)
        return self.visit_proj(torch.cat([hd, hp], -1))

    def forward(self, batch: Dict[str, torch.Tensor],
                nodes: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Return drug logits ``[B, n_drug]`` for a collated batch."""
        if nodes is None:
            nodes = self.encode_graph()
        h = self._visit_vector(nodes, batch["diag"], batch["diag_mask"], batch["proc"], batch["proc_mask"])

        if self.use_history:
            vmask = batch["hist_visit_mask"]  # [B, T]
            B, T = vmask.shape
            hv = self._visit_vector(
                nodes,
                batch["hist_diag"].reshape(B * T, -1), batch["hist_diag_mask"].reshape(B * T, -1),
                batch["hist_proc"].reshape(B * T, -1), batch["hist_proc_mask"].reshape(B * T, -1),
            ).reshape(B, T, -1)
            hdrug = self.pool_drug(F.embedding(batch["hist_drug"], nodes["drug"]), batch["hist_drug_mask"])  # [B, T, d]
            seq = torch.cat([hv, hdrug], -1) * vmask.unsqueeze(-1)
            out, _ = self.hist_gru(seq)
            lengths = vmask.sum(1)  # [B]
            last_idx = (lengths - 1).clamp(min=0)
            h_hist = out[torch.arange(B, device=out.device), last_idx]
            h_hist = h_hist * (lengths > 0).unsqueeze(-1)
            h = self.fuse(torch.cat([h, h_hist], -1))

        q = self.query(h)  # [B, d]
        k = self.key(nodes["drug"])  # [n_drug, d]
        logits = q @ k.t() / (self.dim ** 0.5) + self.drug_bias
        return logits
