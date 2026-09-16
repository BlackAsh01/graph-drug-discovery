"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Args:
        seed: Random seed.
        deterministic: If True, also switch cuDNN to deterministic mode
            (slightly slower, fully reproducible on the same hardware).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
