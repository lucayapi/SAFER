"""Statistiques d'annotation multi-corpus pour article / exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from annotation.export_io import ANNOTATION_TABLE_SUFFIX
from annotation.prompts.v13_two_pass_ambiguity_context import LABELS

_DEFAULT_CORPUS_NAMES: dict[str, str] = {
    "run_all_btp": "BTP",
    "run_all_metallurgie": "Métallurgie",
    "run_all_caou_chimie_plas": "Caoutchouc-chimie-plastiques",
    "run_all_bois_textille": "Bois-textile",
    "run_all_service_temporaire": "Service temporaire",
}


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def corpus_display_name(run_id: str, run_config: Optional[dict[str, Any]] = None) -> str:
    """Nom lisible du corpus pour tableaux / graphiques."""
    if run_id in _DEFAULT_CORPUS_NAMES:
        return _DEFAULT_CORPUS_NAMES[run_id]

    cfg = run_config or {}
    input_csv = str(cfg.get("input_csv") or "").strip()
    if input_csv:
        stem = Path(input_csv).stem
        return stem.replace("_sentence_accidents", "").replace("_", " ").strip().title()

    basename = str(cfg.get("output_basename") or "").strip()
    if basename:
        return basename.replace("_v13_pass1", "").replace("_", " ").strip().title()

    return run_id


def find_annotated_xlsx(run_dir: Path) -> Optional[Path]:
    """Cherche le fichier *__annotated.xlsx dans un dossier de run."""
    matches = sorted(run_dir.glob(f"*__annotated{ANNOTATION_TABLE_SUFFIX}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    # Préfère le fichier le plus récent si plusieurs.
    return max(matches, key=lambda path: path.stat().st_mtime)


def discover_annotation_runs(outputs_dir: Path) -> list[dict[str, Any]]:
    """Liste les runs avec un export annoté disponible."""
    if not outputs_dir.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        annotated_path = find_annotated_xlsx(run_dir)
        if annotated_path is None:
            continue
        run_config = _load_run_config(run_dir)
        run_id = str(run_config.get("run_id") or run_dir.name)
        runs.append(
            {
                "run_id": run_id,
                "corpus": corpus_display_name(run_id, run_config),
                "run_dir": run_dir,
                "annotated_path": annotated_path,
                "run_config": run_config,
            }
        )
    return runs


def _valid_annotation_mask(df: pd.DataFrame) -> pd.Series:
    if "pred_ok" in df.columns:
        return df["pred_ok"].astype("boolean").fillna(False)
    if "pred_label" in df.columns:
        return df["pred_label"].isin(LABELS)
    return pd.Series(False, index=df.index)


def corpus_annotation_summary(df: pd.DataFrame, *, corpus: str) -> dict[str, Any]:
    """Résumé d'un corpus annoté."""
    n_units = int(len(df))
    n_recits = int(df["accident_id"].nunique()) if "accident_id" in df.columns else 0

    valid = df.loc[_valid_annotation_mask(df)].copy()
    n_annotated = int(len(valid))

    label_counts = (
        valid["pred_label"].value_counts()
        if "pred_label" in valid.columns
        else pd.Series(dtype=int)
    )
    denom = int(label_counts.sum()) if len(label_counts) else 0

    pct: dict[str, float] = {}
    for label in LABELS:
        count = int(label_counts.get(label, 0))
        pct[f"pct_{label}"] = round(100.0 * count / denom, 2) if denom else 0.0
        pct[f"n_{label}"] = count

    return {
        "corpus": corpus,
        "n_recits": n_recits,
        "n_units": n_units,
        "n_annotated": n_annotated,
        "n_missing_or_failed": n_units - n_annotated,
        **pct,
    }


def build_summary_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Tableau article : récits, unités, % par classe."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = pd.read_excel(run["annotated_path"], engine="openpyxl")
        rows.append(corpus_annotation_summary(df, corpus=str(run["corpus"])))
    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    ordered_cols = [
        "corpus",
        "n_recits",
        "n_units",
        "n_annotated",
        "n_missing_or_failed",
        "pct_A0",
        "pct_A1",
        "pct_B",
        "pct_C",
        "n_A0",
        "n_A1",
        "n_B",
        "n_C",
    ]
    existing = [col for col in ordered_cols if col in table.columns]
    return table[existing]


def build_label_distribution_long(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Format long pour graphiques (corpus × label × effectif / %)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = pd.read_excel(run["annotated_path"], engine="openpyxl")
        valid = df.loc[_valid_annotation_mask(df)]
        counts = valid["pred_label"].value_counts() if "pred_label" in valid.columns else pd.Series(dtype=int)
        total = int(counts.sum())
        for label in LABELS:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "corpus": str(run["corpus"]),
                    "run_id": str(run["run_id"]),
                    "label": label,
                    "n_units": count,
                    "pct": round(100.0 * count / total, 2) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)
