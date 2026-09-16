"""Visit-level dataset and collate function.

Each sample is one admission (the *target visit*) together with the
patient's previous admissions (the *history*).  Codes are returned as
padded index tensors with masks so the model can pool node embeddings
from the heterogeneous graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class VisitSample:
    diag: List[int]
    proc: List[int]
    drug: List[int]
    hist_diag: List[List[int]]
    hist_proc: List[List[int]]
    hist_drug: List[List[int]]
    subject_id: int
    hadm_id: int


class VisitDataset(Dataset):
    """Flatten patient records into per-visit samples with bounded history."""

    def __init__(self, records: Sequence[Sequence[Dict[str, object]]], max_history: int = 10,
                 min_visit_index: int = 0):
        self.samples: List[VisitSample] = []
        for patient in records:
            for t, visit in enumerate(patient):
                if t < min_visit_index:
                    continue
                hist = patient[max(0, t - max_history):t]
                self.samples.append(VisitSample(
                    diag=list(visit["diag"]),
                    proc=list(visit.get("proc", [])),
                    drug=list(visit["drug"]),
                    hist_diag=[list(h["diag"]) for h in hist],
                    hist_proc=[list(h.get("proc", [])) for h in hist],
                    hist_drug=[list(h["drug"]) for h in hist],
                    subject_id=int(visit.get("subject_id", -1)),
                    hadm_id=int(visit.get("hadm_id", -1)),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> VisitSample:
        return self.samples[i]


def _pad_lists(lists: Sequence[Sequence[int]], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of index lists -> ``(idx [B, L], mask [B, L])``; empty lists get a length-1 masked row."""
    L = max(1, max((len(x) for x in lists), default=1))
    idx = np.full((len(lists), L), pad_value, dtype=np.int64)
    mask = np.zeros((len(lists), L), dtype=bool)
    for i, x in enumerate(lists):
        if x:
            idx[i, :len(x)] = x
            mask[i, :len(x)] = True
    return torch.from_numpy(idx), torch.from_numpy(mask)


def _pad_history(hists: Sequence[Sequence[Sequence[int]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad histories -> ``(idx [B, T, L], code_mask [B, T, L], visit_mask [B, T])``."""
    B = len(hists)
    T = max(1, max((len(h) for h in hists), default=1))
    L = max(1, max((len(v) for h in hists for v in h), default=1))
    idx = np.zeros((B, T, L), dtype=np.int64)
    cmask = np.zeros((B, T, L), dtype=bool)
    vmask = np.zeros((B, T), dtype=bool)
    for i, h in enumerate(hists):
        if h:
            vmask[i, :len(h)] = True
        for t, v in enumerate(h):
            if v:
                idx[i, t, :len(v)] = v
                cmask[i, t, :len(v)] = True
    return torch.from_numpy(idx), torch.from_numpy(cmask), torch.from_numpy(vmask)


def make_collate(n_drug: int):
    """Return a collate function producing a dict of padded tensors."""

    def collate(batch: Sequence[VisitSample]) -> Dict[str, torch.Tensor]:
        diag, diag_mask = _pad_lists([s.diag for s in batch])
        proc, proc_mask = _pad_lists([s.proc for s in batch])
        y = np.zeros((len(batch), n_drug), dtype=np.float32)
        for i, s in enumerate(batch):
            if s.drug:
                y[i, s.drug] = 1.0
        y = torch.from_numpy(y)
        h_diag, h_diag_mask, h_vmask = _pad_history([s.hist_diag for s in batch])
        h_proc, h_proc_mask, _ = _pad_history([s.hist_proc for s in batch])
        h_drug, h_drug_mask, _ = _pad_history([s.hist_drug for s in batch])
        return {
            "diag": diag, "diag_mask": diag_mask,
            "proc": proc, "proc_mask": proc_mask,
            "y": y,
            "hist_diag": h_diag, "hist_diag_mask": h_diag_mask,
            "hist_proc": h_proc, "hist_proc_mask": h_proc_mask,
            "hist_drug": h_drug, "hist_drug_mask": h_drug_mask,
            "hist_visit_mask": h_vmask,
        }

    return collate


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
