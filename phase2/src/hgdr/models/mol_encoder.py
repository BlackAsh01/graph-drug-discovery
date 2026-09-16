"""Molecular graph encoder for drug nodes (SMILES -> vector).

All drug molecules are packed into one disjoint batch graph once; a
GAT/GIN message-passing stack followed by mean pooling yields one vector
per drug.  Drugs without a parsable SMILES receive a learned
``no_molecule`` vector.  This replaces the official-figure Dual-attention
Graph Transformer + InfoNCE DDI branch (DrugDAGT in the research
``dagt/`` tree) with a lighter encoder that fits a 4 GB GPU together
with the EHR graph.  Pairwise InfoNCE is **not** implemented.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv
from torch_geometric.utils import scatter


class MolEncoder(nn.Module):
    """Batched molecular encoder.

    Args:
        in_dim: Atom feature dimension.
        hidden_dim: Hidden/output dimension.
        num_layers: Number of message-passing layers.
        conv: ``'gat'`` or ``'gin'``.
        heads: Attention heads for GAT.
        dropout: Dropout probability.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, conv: str = "gat",
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "gat":
                self.layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            elif conv == "gin":
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(GINConv(mlp, train_eps=True))
            else:
                raise ValueError(f"unknown conv '{conv}'")
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
        self.no_molecule = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.no_molecule, std=0.02)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                n_drugs: int, has_mol: torch.Tensor) -> torch.Tensor:
        """Return ``[n_drugs, hidden_dim]`` molecular features.

        Args:
            x: ``[n_atoms_total, in_dim]`` atom features of all molecules.
            edge_index: ``[2, n_bonds_total]`` bond indices (already offset).
            batch: ``[n_atoms_total]`` drug index of every atom.
            n_drugs: Vocabulary size.
            has_mol: ``[n_drugs]`` bool mask of drugs with a molecule.
        """
        h = F.relu(self.proj(x))
        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, edge_index)
            h = norm(h + F.dropout(F.relu(h_new), self.dropout, self.training))
        pooled = scatter(h, batch, dim=0, dim_size=n_drugs, reduce="mean")
        pooled = self.out(pooled)
        fallback = self.no_molecule.unsqueeze(0).expand(n_drugs, -1)
        return torch.where(has_mol.unsqueeze(1), pooled, fallback)


def pack_molecules(mol_x: Sequence[Optional[torch.Tensor]], mol_edge_index: Sequence[Optional[torch.Tensor]],
                   atom_dim: int):
    """Concatenate per-drug molecule graphs into ``(x, edge_index, batch, has_mol)``."""
    xs: List[torch.Tensor] = []
    eis: List[torch.Tensor] = []
    batch: List[torch.Tensor] = []
    has = []
    offset = 0
    for i, (x, ei) in enumerate(zip(mol_x, mol_edge_index)):
        if x is None:
            has.append(False)
            continue
        has.append(True)
        xs.append(x)
        eis.append(ei + offset)
        batch.append(torch.full((x.shape[0],), i, dtype=torch.long))
        offset += x.shape[0]
    if not xs:  # no molecules at all -> dummy single atom
        xs = [torch.zeros((1, atom_dim))]
        eis = [torch.zeros((2, 0), dtype=torch.long)]
        batch = [torch.zeros((1,), dtype=torch.long)]
    return (torch.cat(xs, 0), torch.cat(eis, 1), torch.cat(batch, 0),
            torch.tensor(has, dtype=torch.bool))
