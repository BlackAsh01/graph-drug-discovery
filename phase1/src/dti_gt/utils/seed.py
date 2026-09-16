"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Args:
        seed: the seed.
        deterministic: if ``True`` also disable cuDNN autotuning so that runs with the
            same seed produce (near-)identical results. Scatter-based message passing on
            CUDA is still not bit-exact, so expect tiny run-to-run differences on GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # pragma: no cover - only used with num_workers > 0
    """DataLoader worker init function that derives a per-worker seed from torch's seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
