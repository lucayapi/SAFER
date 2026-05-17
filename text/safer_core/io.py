"""I/O YAML/JSON/CSV pour les runs SAFER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML racine doit être un mapping : {path}")
    return data


def flatten_method_config(data: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section in ("data", "model", "training", "method"):
        block = data.get(section)
        if isinstance(block, dict):
            flat.update(block)
    for key, value in data.items():
        if key not in ("data", "model", "training", "method") and not isinstance(value, dict):
            flat[key] = value
    return flat


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def save_config_resolved(config: Dict[str, Any], method_dir: Path) -> Path:
    out = method_dir / "configs" / "config_resolved.yaml"
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return out


def save_metrics_geometry(row: Dict[str, Any], metrics_dir: Path, stem: str = "metrics_geometry") -> None:
    ensure_dir(metrics_dir)
    save_json(row, metrics_dir / f"{stem}.json")
    pd.DataFrame([row]).to_csv(metrics_dir / f"{stem}.csv", index=False)
