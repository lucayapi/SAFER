"""Tests agrégation annotation."""

from __future__ import annotations

import pandas as pd

from annotation.aggregate import aggregate_outcomes_by_accident
from annotation.export_io import attach_accident_summary_column, reorder_annotation_output_columns


def test_attach_accident_summary_column_from_summary_accident():
    df = pd.DataFrame(
        {
            "accident_id": ["A1"],
            "sentence": ["chute"],
            "summary_accident": ["Résumé accident A1"],
        }
    )
    out = attach_accident_summary_column(df, summary_col="summary_accident")
    assert out["accident_summary"].iloc[0] == "Résumé accident A1"


def test_aggregate_outcomes_includes_accident_summary():
    df = pd.DataFrame(
        {
            "accident_id": ["A1", "A1"],
            "fact_id": [1, 2],
            "accident_summary": ["Résumé A1", "Résumé A1"],
            "pred_injury_mentioned": ["YES", "NOT_MENTIONED"],
            "pred_hospitalized": ["NO", "NO"],
            "pred_fatal": ["NOT_MENTIONED", "NOT_MENTIONED"],
            "pred_ok": [True, True],
            "pred_context_used": [True, False],
        }
    )
    out = aggregate_outcomes_by_accident(df)
    assert out["accident_summary"].iloc[0] == "Résumé A1"
    assert out["n_context_used_units"].iloc[0] == 1


def test_aggregate_outcomes_includes_ambiguity_counts():
    df = pd.DataFrame(
        {
            "accident_id": ["A1", "A1", "A2"],
            "pred_ok": [True, True, True],
            "pred_ambiguous": [True, False, True],
            "pred_context_needed": [True, False, False],
            "pred_injury_mentioned": ["NOT_MENTIONED"] * 3,
            "pred_hospitalized": ["NOT_MENTIONED"] * 3,
            "pred_fatal": ["NOT_MENTIONED"] * 3,
        }
    )
    out = aggregate_outcomes_by_accident(df)
    row_a1 = out[out["accident_id"] == "A1"].iloc[0]
    assert row_a1["n_ambiguous_units"] == 1
    assert bool(row_a1["accident_any_ambiguous"]) is True
    assert row_a1["n_context_needed_units"] == 1


def test_reorder_annotation_output_columns_puts_summary_near_front():
    df = pd.DataFrame(
        {
            "division": ["BTP"],
            "accident_id": ["A1"],
            "fact_id": [1],
            "sentence": ["chute"],
            "accident_summary": ["Résumé"],
            "pred_label": ["B"],
        }
    )
    out = reorder_annotation_output_columns(df)
    assert list(out.columns[:4]) == ["accident_id", "fact_id", "sentence", "accident_summary"]
