"""Agrégation des prédictions au niveau accident."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from annotation.export_io import SUMMARY_OUTPUT_COL, SUMMARY_FALLBACK_COLS
from annotation.prompts.v10_macro_labels_independent_outcomes import MENTION_VALUES


def aggregate_mention(values: pd.Series) -> Tuple[str, bool]:
    cleaned = {
        str(value).strip()
        for value in values.dropna()
        if str(value).strip() in MENTION_VALUES
    }
    has_yes = "YES" in cleaned
    has_no = "NO" in cleaned
    conflict = has_yes and has_no
    if has_yes:
        return "YES", conflict
    if has_no:
        return "NO", conflict
    return "NOT_MENTIONED", conflict


def _count_true(values: pd.Series) -> int:
    """Compte les valeurs truthy sans downcast implicite sur object dtype."""
    return int(values.astype("boolean").fillna(False).sum())


def _first_non_empty_summary(group: pd.DataFrame, summary_col: str) -> str:
    candidates = [summary_col, *SUMMARY_FALLBACK_COLS]
    seen: set[str] = set()
    for col in candidates:
        if col in seen or col not in group.columns:
            continue
        seen.add(col)
        for value in group[col].dropna():
            text = str(value).strip()
            if text:
                return text
    return ""


def aggregate_outcomes_by_accident(
    df_pred: pd.DataFrame,
    *,
    summary_col: str = SUMMARY_OUTPUT_COL,
) -> pd.DataFrame:
    if "accident_id" not in df_pred.columns:
        raise ValueError("La colonne accident_id est requise pour l'agrégation.")

    rows = []
    for accident_id, group in df_pred.groupby("accident_id", dropna=False):
        injury, injury_conflict = aggregate_mention(group["pred_injury_mentioned"])
        hospitalized, hospitalized_conflict = aggregate_mention(group["pred_hospitalized"])
        fatal, fatal_conflict = aggregate_mention(group["pred_fatal"])
        context_used = group.get("pred_context_used")
        n_context_used = _count_true(context_used) if context_used is not None else 0

        ambiguous = group.get("pred_ambiguous")
        n_ambiguous = _count_true(ambiguous) if ambiguous is not None else 0

        context_needed = group.get("pred_context_needed")
        n_context_needed = _count_true(context_needed) if context_needed is not None else 0

        rows.append(
            {
                "accident_id": accident_id,
                SUMMARY_OUTPUT_COL: _first_non_empty_summary(group, summary_col),
                "n_factual_units": int(len(group)),
                "n_valid_predictions": _count_true(group["pred_ok"]),
                "n_context_used_units": n_context_used,
                "accident_any_context_used": n_context_used > 0,
                "n_ambiguous_units": n_ambiguous,
                "accident_any_ambiguous": n_ambiguous > 0,
                "n_context_needed_units": n_context_needed,
                "accident_any_context_needed": n_context_needed > 0,
                "accident_injury_mentioned": injury,
                "accident_hospitalized": hospitalized,
                "accident_fatal": fatal,
                "injury_annotation_conflict": injury_conflict,
                "hospitalization_annotation_conflict": hospitalized_conflict,
                "fatal_annotation_conflict": fatal_conflict,
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = {
        "label": "pred_label",
        "injury_mentioned": "pred_injury_mentioned",
        "hospitalized": "pred_hospitalized",
        "fatal": "pred_fatal",
        "context_used": "pred_context_used",
        "ambiguous": "pred_ambiguous",
        "context_needed": "pred_context_needed",
        "alternative_label": "pred_alternative_label",
        "ambiguity_type": "pred_ambiguity_type",
        "prediction_ok": "pred_ok",
    }
    for level, column in columns.items():
        if column not in df_pred.columns:
            continue
        for value, count in df_pred[column].value_counts(dropna=False).items():
            rows.append({"level": level, "value": value, "count": int(count)})
    return pd.DataFrame(rows)
