"""YAML configuration handling with inheritance (``base:`` key) and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML scalars)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config, resolving an optional ``base: <relative path>`` chain."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        base_cfg = load_config((path.parent / base).resolve())
        cfg = deep_update(base_cfg, cfg)
    cfg.setdefault("config_path", str(path))
    return apply_overrides(cfg, overrides)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Dump a resolved config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
