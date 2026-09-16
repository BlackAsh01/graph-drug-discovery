"""Logging and run-directory helpers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_logger(name: str = "hgdr", log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger that prints to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and detach all handlers (call at the end of a run so files are closed)."""
    for h in list(logger.handlers):
        h.flush()
        if isinstance(h, logging.FileHandler):
            h.close()
        logger.removeHandler(h)


def setup_run_dir(results_dir: Path, run_name: str) -> Path:
    """Create ``results/runs/<run_name>/`` and return it."""
    run_dir = Path(results_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Write a JSON file (creating parent directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)
