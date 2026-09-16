"""Recommendation losses: BCE (+ multi-label margin) and the DDI penalty.

The DDI loss is a differentiable surrogate of the *DDI rate* metric.  With
predicted probabilities ``p`` and the symmetric DDI adjacency ``A``

    L_ddi = mean_b  (p_b^T A p_b) / (sum_{i != j} p_bi p_bj)

i.e. the expected fraction of recommended drug pairs that interact (the
"soft DDI rate").  GAMENet / SafeDrug use the un-normalised numerator
divided by ``n_drug^2``; with a 600-drug vocabulary that constant makes the
term ~20x smaller than in their 131-drug setting, so we normalise by the
expected number of pairs instead - the coefficient ``ddi_weight`` is then
comparable across vocabularies.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def ddi_loss(logits: torch.Tensor, ddi_adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft DDI rate: expected interacting pair mass / expected total pair mass."""
    p = torch.sigmoid(logits)  # [B, n]
    inter = ((p @ ddi_adj) * p).sum(-1)  # sum_ij A_ij p_i p_j (both orderings)
    total = p.sum(-1) ** 2 - (p * p).sum(-1)  # sum_{i != j} p_i p_j
    return (inter / (total + eps)).mean()


def recommendation_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    ddi_adj: Optional[torch.Tensor] = None,
    ddi_weight: float = 0.0,
    margin_weight: float = 0.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return ``{'loss', 'bce', 'ddi', 'margin'}``.

    Args:
        logits: ``[B, n_drug]`` raw scores.
        y: ``[B, n_drug]`` binary targets.
        ddi_adj: DDI adjacency for the penalty (ignored if ``ddi_weight==0``).
        ddi_weight: Coefficient of the DDI penalty.
        margin_weight: Coefficient of ``multilabel_margin_loss`` (GAMENet
            uses 0.05; default 0 = pure BCE).
        pos_weight: Optional per-drug positive weight for BCE.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    out = {"bce": bce}
    total = bce
    if margin_weight > 0:
        # multilabel_margin_loss wants target index lists padded with -1
        tgt = torch.full_like(y, -1, dtype=torch.long)
        for i in range(y.shape[0]):
            idx = torch.nonzero(y[i]).flatten()
            tgt[i, :len(idx)] = idx
        m = F.multilabel_margin_loss(torch.sigmoid(logits), tgt)
        out["margin"] = m
        total = (1.0 - margin_weight) * total + margin_weight * m
    else:
        out["margin"] = torch.zeros((), device=logits.device)
    if ddi_weight > 0 and ddi_adj is not None:
        d = ddi_loss(logits, ddi_adj)
        out["ddi"] = d
        total = total + ddi_weight * d
    else:
        out["ddi"] = torch.zeros((), device=logits.device)
    out["loss"] = total
    return out
