"""Tests module two_pass (filtrage et fusion)."""

from __future__ import annotations

import pandas as pd

from annotation.two_pass import (
    build_first_pass_annotation,
    filter_pass2_candidates,
    merge_pass1_pass2,
    pass2_ambiguity_overview,
)


def test_filter_pass2_candidates():
    df = pd.DataFrame(
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "pred_label": "A0",
                "pred_ambiguous": True,
                "pred_context_needed": True,
                "pred_alternative_label": "B",
            },
            {
                "accident_id": "A1",
                "fact_id": 2,
                "pred_label": "B",
                "pred_ambiguous": False,
                "pred_context_needed": False,
                "pred_alternative_label": "NONE",
            },
        ]
    )
    out = filter_pass2_candidates(df)
    assert len(out) == 1
    assert out.iloc[0]["fact_id"] == 1


def test_build_first_pass_annotation():
    row = pd.Series(
        {
            "pred_label": "A0",
            "pred_confidence": 0.6,
            "pred_ambiguous": True,
            "pred_context_needed": True,
            "pred_alternative_label": "B",
            "pred_ambiguity_type": "ACTION_INTENT",
            "pred_ambiguity_reason": "doute",
        }
    )
    ann = build_first_pass_annotation(row)
    assert ann["label"] == "A0"
    assert ann["alternative_label"] == "B"


def test_pass2_ambiguity_overview_counts_candidates():
    df = pd.DataFrame(
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "pred_label": "A0",
                "pred_ok": True,
                "pred_ambiguous": True,
                "pred_context_needed": True,
                "pred_alternative_label": "B",
                "pred_ambiguity_type": "ACTION_INTENT",
            },
            {
                "accident_id": "A1",
                "fact_id": 2,
                "pred_label": "B",
                "pred_ok": True,
                "pred_ambiguous": False,
                "pred_context_needed": False,
                "pred_alternative_label": "NONE",
            },
            {
                "accident_id": "A2",
                "fact_id": 3,
                "pred_label": "A1",
                "pred_ok": False,
                "pred_ambiguous": False,
                "pred_context_needed": False,
                "pred_alternative_label": "NONE",
            },
        ]
    )
    overview = pass2_ambiguity_overview(df)
    summary = overview["summary"]
    assert summary["n_pass1_units"] == 3
    assert summary["n_pass2_candidates"] == 1
    assert summary["n_pred_ok"] == 2
    assert summary["n_pred_not_ok"] == 1
    assert summary["n_ambiguous"] == 1
    assert summary["n_accidents_with_candidate"] == 1
    assert len(overview["by_label"]) == 1


def test_merge_pass1_pass2_keeps_pass1_outcomes():
    pass1 = pd.DataFrame(
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "sentence": "s1",
                "pred_label": "A0",
                "pred_injury_mentioned": "NOT_MENTIONED",
                "pred_hospitalized": "NOT_MENTIONED",
                "pred_fatal": "NOT_MENTIONED",
            },
            {
                "accident_id": "A1",
                "fact_id": 2,
                "sentence": "s2",
                "pred_label": "B",
                "pred_injury_mentioned": "YES",
                "pred_hospitalized": "NO",
                "pred_fatal": "NOT_MENTIONED",
            },
        ]
    )
    pass2 = pd.DataFrame(
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "pred_label": "B",
                "pred_injury_mentioned": "YES",
                "pred_hospitalized": "YES",
                "pred_fatal": "YES",
                "pred_ok": True,
                "pred_ambiguous": False,
                "pred_context_used": True,
            }
        ]
    )
    merged = merge_pass1_pass2(pass1, pass2)
    row = merged[merged["fact_id"] == 1].iloc[0]
    assert row["pred_label"] == "B"
    assert row["pred_injury_mentioned"] == "NOT_MENTIONED"
    assert bool(row["pred_reannotated"]) is True
    assert merged[merged["fact_id"] == 2].iloc[0]["pred_label"] == "B"
