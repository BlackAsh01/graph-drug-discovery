"""YAML configuration loading with inheritance and CLI overrides.

A config file may contain a ``base:`` key pointing to another YAML file
(relative to the config's directory); the child is deep-merged on top of
the base.  Command-line overrides use dotted paths, e.g.
``--set model.hidden_dim=128 train.epochs=5``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    """Parse a CLI scalar with YAML semantics (``true``, ``1e-3``, ``null`` ...)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``key.sub=value`` overrides in place and return the config."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(value)
    return cfg


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config, resolving ``base:`` inheritance and overrides."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base", None)
    if base_ref:
        base_cfg = load_config(path.parent / base_ref)
        cfg = _deep_merge(base_cfg, cfg)
    cfg.setdefault("config_path", str(path))
    return apply_overrides(cfg, overrides)


def config_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``cfg['a']['b']`` from ``'a.b'`` with a default."""
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
