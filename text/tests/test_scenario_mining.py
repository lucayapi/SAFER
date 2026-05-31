"""Tests extraction scénarios BN (chemins macro complets)."""

from __future__ import annotations

import pandas as pd

from bn_pipeline.scenario_mining import (
    extract_typical_scenarios,
    macro_path_from_topics,
    sort_topics_by_macro_order,
)


def test_sort_topics_by_macro_order():
    cfg = (
        "macro_topic_C_01",
        "macro_topic_A0_03",
        "macro_topic_B_02",
        "macro_topic_A1_01",
    )
    ordered = sort_topics_by_macro_order(cfg)
    assert [c.split("_")[2] for c in ordered] == ["A0", "A1", "B", "C"]


def test_macro_path_from_topics():
    cfg = (
        "macro_topic_B_01",
        "macro_topic_A0_03",
        "macro_topic_C_04",
        "macro_topic_A1_02",
    )
    assert macro_path_from_topics(cfg) == "A0 -> A1 -> B -> C"


def test_extract_typical_scenarios_requires_full_macro_path():
    df = pd.DataFrame(
        [
            {
                "accident_id": "a1",
                "macro_topic_A0_01": 1,
                "macro_topic_A1_01": 1,
                "macro_topic_B_01": 1,
                "macro_topic_C_01": 1,
            },
            {
                "accident_id": "a2",
                "macro_topic_B_01": 1,
                "macro_topic_C_01": 1,
            },
            {
                "accident_id": "a3",
                "macro_topic_A0_01": 0,
                "macro_topic_A1_01": 0,
                "macro_topic_B_01": 0,
                "macro_topic_C_01": 0,
            },
        ]
    )
    topic_cols = [
        "macro_topic_A0_01",
        "macro_topic_A1_01",
        "macro_topic_B_01",
        "macro_topic_C_01",
    ]
    freq, _ = extract_typical_scenarios(
        df,
        None,
        topic_cols,
        accident_id_col="accident_id",
        min_support=1,
        top_n=10,
        exclude_empty=True,
        require_full_macro_path=True,
    )
    assert len(freq) == 1
    assert freq.iloc[0]["macro_path"] == "A0 -> A1 -> B -> C"
    assert freq.iloc[0]["support"] == 1
