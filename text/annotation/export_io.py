"""Export tabulaire des sorties d'annotation (format uniforme XLSX)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

ANNOTATION_TABLE_SUFFIX = ".xlsx"
ANNOTATION_TABLE_ENGINE = "openpyxl"
SUMMARY_OUTPUT_COL = "accident_summary"
SUMMARY_FALLBACK_COLS = ("accident_summary", "summary_accident")

# Caractères de contrôle interdits par openpyxl / XML (tab, LF, CR conservés).
_ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


def sanitize_excel_cell_value(value: Any) -> Any:
    """Supprime les caractères de contrôle illégaux pour l'export XLSX."""
    if not isinstance(value, str):
        return value
    return _ILLEGAL_EXCEL_CHARS_RE.sub("", value)


def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les colonnes texte avant écriture Excel."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(sanitize_excel_cell_value)
    return out


def attach_accident_summary_column(
    df: pd.DataFrame,
    *,
    summary_col: str = SUMMARY_OUTPUT_COL,
) -> pd.DataFrame:
    """Garantit une colonne ``accident_summary`` dans les exports."""
    out = df.copy()
    if SUMMARY_OUTPUT_COL in out.columns:
        return out
    if summary_col in out.columns:
        out[SUMMARY_OUTPUT_COL] = out[summary_col]
        return out
    for col in SUMMARY_FALLBACK_COLS:
        if col in out.columns:
            out[SUMMARY_OUTPUT_COL] = out[col]
            return out
    return out


def reorder_annotation_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place les colonnes clés en tête du fichier annoté."""
    front = ["accident_id", "fact_id", "sentence", SUMMARY_OUTPUT_COL]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    ordered = [c for c in front if c in df.columns]
    ordered.extend(c for c in pred_cols if c not in ordered)
    ordered.extend(c for c in df.columns if c not in ordered)
    return df[ordered]


def save_annotation_table(df: pd.DataFrame, path: Path | str) -> Path:
    """Écrit un DataFrame annotation en XLSX."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sanitize_dataframe_for_excel(df).to_excel(out, index=False, engine=ANNOTATION_TABLE_ENGINE)
    return out
