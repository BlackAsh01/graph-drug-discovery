"""Protein encoders. All of them map a per-target input tensor to a ``[B, dim]`` vector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dti_gt.data.proteins import AA_VOCAB


class PretrainedEmbeddingEncoder(nn.Module):
    """MLP on top of frozen, pre-computed protein-language-model embeddings (e.g. ProtBERT-BFD)."""

    def __init__(self, input_dim: int, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNSequenceEncoder(nn.Module):
    """Learned residue embedding + 3 x Conv1d (kernels 4/8/12, as in DeepDTA / GraphormerDTI) + max-pool."""

    def __init__(self, dim: int, embed_dim: int = 128, channels: int = 128, kernels=(4, 8, 12), dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(AA_VOCAB) + 1, embed_dim, padding_idx=0)
        convs, c_in = [], embed_dim
        for k in kernels:
            convs += [nn.Conv1d(c_in, channels, kernel_size=k), nn.ReLU()]
            c_in = channels
        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embedding(tokens).transpose(1, 2)  # [B, C, L]
        h = self.convs(h)
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return self.proj(self.dropout(h))


class CompositionEncoder(nn.Module):
    """MLP on amino-acid composition (bag-of-residues) - a deliberately weak baseline."""

    def __init__(self, input_dim: int, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_protein_encoder(cfg: dict, input_dim: int, dim: int) -> nn.Module:
    """Factory keyed on ``cfg['encoder']`` (``pretrained`` / ``cnn`` / ``composition``)."""
    kind = cfg["encoder"]
    dropout = float(cfg.get("dropout", 0.1))
    if kind == "pretrained":
        return PretrainedEmbeddingEncoder(input_dim, dim, dropout)
    if kind == "cnn":
        return CNNSequenceEncoder(dim, int(cfg.get("embed_dim", 128)), int(cfg.get("channels", 128)), tuple(cfg.get("kernels", (4, 8, 12))), dropout)
    if kind == "composition":
        return CompositionEncoder(input_dim, dim, dropout)
    raise ValueError(f"unknown protein encoder '{kind}'")
