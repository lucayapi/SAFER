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
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        hint_lines: list[str] = []
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            mark = exc.problem_mark
            lines = text.splitlines()
            idx = mark.line
            for j in range(max(0, idx - 2), min(len(lines), idx + 3)):
                hint_lines.append(f"  L{j + 1}: {lines[j]!r}")
        hint = "\n".join(hint_lines) if hint_lines else ""
        raise ValueError(
            f"YAML invalide : {p}\n{exc}"
            + (f"\nContexte :\n{hint}" if hint else "")
            + "\nVérifier les ':' manquants (ex. `encode_batch_size: 8`)."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML racine doit être un mapping : {p}")
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
