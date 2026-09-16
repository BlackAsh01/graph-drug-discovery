"""HGDR: Heterogeneous-Graph Drug Recommendation with DDI awareness.

Phase 2 of *GRAND: Graph-based Recommendation with Attention Network for
Personalized Drug Therapy* (Ashwin Prabhu M, 2023176029, Anna University,
May 2025).  The official architecture also draws a genome channel, HAN
meta-paths, semantic fusion over Z^(m) / Z^(p), and a Dual-attention
Graph Transformer + InfoNCE DDI head.  **This package implements the
EHR+DDI subset that was actually trained:** MIMIC-III diagnoses /
procedures / drugs, PubChem SMILES (GAT/GIN), TWOSIDES DDI edges, and
BCE + soft-DDI-rate (L1 in the release).  See the Phase 2 README gap
table and ``docs/thesis_notes.md``.
"""

__version__ = "1.0.2"
