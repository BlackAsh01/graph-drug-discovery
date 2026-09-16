"""Minimal logging setup writing to console and an optional file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "dti_gt", log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints to stdout and (optionally) appends to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # reset handlers so repeated calls (e.g. ablation loop) don't duplicate output
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
