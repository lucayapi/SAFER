"""Cache JSONL et chemins de sortie pour reprise d'annotation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from annotation.export_io import ANNOTATION_TABLE_SUFFIX


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def make_cache_key(row: pd.Series, row_idx: Optional[int] = None) -> str:
    accident_id = str(row.get("accident_id", "")).strip()
    fact_id = str(row.get("fact_id", "")).strip()
    if fact_id:
        return f"{accident_id}||{fact_id}"
    if row_idx is not None:
        return f"{accident_id}||ROWIDX_{row_idx}"
    return f"{accident_id}||ROWIDX_FALLBACK"


def batch_attempt_count(cached: Optional[Dict[str, Any]]) -> int:
    """Nombre de soumissions batch déjà enregistrées pour cette ligne."""
    if not cached:
        return 0
    if cached.get("pred_ok"):
        return 0
    if "batch_attempts" in cached:
        return int(cached.get("batch_attempts") or 0)
    # Entrées legacy : échec déjà stocké = une tentative consommée.
    if cached.get("pred_error") or cached.get("pred_raw"):
        return 1
    return 0


def should_skip_row_for_batch(
    cached: Optional[Dict[str, Any]],
    *,
    max_batch_retries: int = 1,
) -> bool:
    """True si la ligne ne doit pas être renvoyée au batch."""
    if not cached:
        return False
    if cached.get("pred_ok"):
        return True
    return batch_attempt_count(cached) >= max(1, int(max_batch_retries))


def get_batch_paths(
    output_dir: Path,
    *,
    model_id: str,
    prompt_version: str,
    artifact_slug: Optional[str] = None,
) -> Tuple[Path, Path, Path, Path]:
    model_slug = sanitize_filename(model_id)
    slug = artifact_slug or prompt_version
    base_name = f"{model_slug}__{slug}"
    return (
        output_dir / f"{base_name}__batch_input.jsonl",
        output_dir / "batch_output.jsonl",
        output_dir / "batch_errors.jsonl",
        output_dir / "batch_state.json",
    )


def get_output_paths(
    output_dir: Path,
    *,
    model_id: str,
    prompt_version: str,
    artifact_slug: Optional[str] = None,
) -> Tuple[Path, Path, Path, Path, Path]:
    model_slug = sanitize_filename(model_id)
    slug = artifact_slug or prompt_version
    base_name = f"{model_slug}__{slug}"
    return (
        output_dir / f"{base_name}.jsonl",
        output_dir / f"{base_name}__snapshot{ANNOTATION_TABLE_SUFFIX}",
        output_dir / f"{base_name}__annotated{ANNOTATION_TABLE_SUFFIX}",
        output_dir / f"{base_name}__summary{ANNOTATION_TABLE_SUFFIX}",
        output_dir / f"{base_name}__accident_outcomes{ANNOTATION_TABLE_SUFFIX}",
    )


def load_cache(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.is_file():
        return cache
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["cache_key"]] = obj
            except Exception:
                continue
    return cache


def append_to_cache(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_run_config(output_dir: Path, config: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path
