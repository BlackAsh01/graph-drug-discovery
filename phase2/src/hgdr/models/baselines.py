"""Cheap baselines: multi-hot logistic regression / MLP and nearest-visit copy."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


def _multi_hot(idx: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
    """``idx: [B, L]``, ``mask: [B, L]`` -> dense ``[B, n]`` {0,1}."""
    out = torch.zeros((idx.shape[0], n), device=idx.device)
    src = mask.float()
    out.scatter_add_(1, idx, src)
    return out.clamp_(max=1.0)


class MultiHotBaseline(nn.Module):
    """Logistic regression (``hidden_dim=0``) or MLP on multi-hot diag+proc(+history drugs)."""

    def __init__(self, n_diag: int, n_proc: int, n_drug: int, hidden_dim: int = 256, dropout: float = 0.3,
                 use_proc: bool = True, use_history: bool = True):
        super().__init__()
        self.n_diag, self.n_proc, self.n_drug = n_diag, n_proc, n_drug
        self.use_proc = use_proc and n_proc > 0
        self.use_history = use_history
        in_dim = n_diag + (n_proc if self.use_proc else 0) + (n_drug if use_history else 0)
        if hidden_dim > 0:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, n_drug))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_drug))

    def encode_graph(self):  # API compatibility with HGDRecommender
        return None

    def forward(self, batch: Dict[str, torch.Tensor], nodes=None) -> torch.Tensor:
        feats = [_multi_hot(batch["diag"], batch["diag_mask"], self.n_diag)]
        if self.use_proc:
            feats.append(_multi_hot(batch["proc"], batch["proc_mask"], self.n_proc))
        if self.use_history:
            B, T, L = batch["hist_drug"].shape
            hd = _multi_hot(batch["hist_drug"].reshape(B, T * L), batch["hist_drug_mask"].reshape(B, T * L),
                            self.n_drug)
            feats.append(hd)
        return self.net(torch.cat(feats, -1))


class NearestVisitBaseline(nn.Module):
    """Parameter-free: copy the previous admission's drugs; else the most frequent drugs.

    ``drug_prior`` (``[n_drug]`` frequencies from the training split) is used
    for first admissions and to break ties.
    """

    def __init__(self, n_drug: int, drug_prior: torch.Tensor):
        super().__init__()
        self.n_drug = n_drug
        self.register_buffer("drug_prior", drug_prior.float())
        self._dummy = nn.Parameter(torch.zeros(1))  # so optimizers do not choke

    def encode_graph(self):
        return None

    def forward(self, batch: Dict[str, torch.Tensor], nodes=None) -> torch.Tensor:
        vmask = batch["hist_visit_mask"]
        B, T = vmask.shape
        lengths = vmask.sum(1)
        last = (lengths - 1).clamp(min=0)
        idx = batch["hist_drug"][torch.arange(B, device=vmask.device), last]  # [B, L]
        m = batch["hist_drug_mask"][torch.arange(B, device=vmask.device), last]
        copy = _multi_hot(idx, m, self.n_drug) * (lengths > 0).float().unsqueeze(-1)
        prior = self.drug_prior.unsqueeze(0).expand(B, -1)
        # prior logit: sigmoid(.) >= 0.5  <=>  drug prescribed in >= 50% of training visits
        prior_logit = torch.log(prior + 1e-6) - math.log(0.5)
        # copied drugs -> +5, others -> prior-based
        logits = copy * 5.0 + (1 - copy) * prior_logit
        no_hist = (lengths == 0).unsqueeze(-1)
        return torch.where(no_hist, prior_logit, logits)
