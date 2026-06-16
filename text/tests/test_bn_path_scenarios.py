"""Tests chemins BN et scénarios par support accident."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bn_pipeline.bn_paths import enumerate_macro_paths, extract_bn_path_scenarios


@dataclass
class _MockBnModel:
    edges_list: list[tuple[str, str]]

    def edges(self) -> list[tuple[str, str]]:
        return list(self.edges_list)


def _synthetic_model() -> _MockBnModel:
    return _MockBnModel(
        [
            ("macro_topic_A0_01", "macro_topic_A1_01"),
            ("macro_topic_A1_01", "macro_topic_B_01"),
            ("macro_topic_B_01", "macro_topic_C_01"),
            ("macro_topic_A0_01", "macro_topic_B_02"),
            ("macro_topic_B_02", "macro_topic_C_02"),
        ]
    )


def _macro_map() -> dict[str, str]:
    return {
        "macro_topic_A0_01": "A0",
        "macro_topic_A1_01": "A1",
        "macro_topic_B_01": "B",
        "macro_topic_B_02": "B",
        "macro_topic_C_01": "C",
        "macro_topic_C_02": "C",
    }


def test_enumerate_macro_paths_min_two_macros():
    model = _synthetic_model()
    paths = enumerate_macro_paths(model, _macro_map(), min_macros=2, max_path_len=6)

    assert ("macro_topic_A0_01", "macro_topic_A1_01", "macro_topic_B_01", "macro_topic_C_01") in paths
    assert ("macro_topic_A0_01", "macro_topic_B_02", "macro_topic_C_02") in paths
    assert ("macro_topic_A1_01", "macro_topic_B_01") in paths
    assert ("macro_topic_B_01", "macro_topic_C_01") in paths


def test_enumerate_macro_paths_rejects_single_macro():
    model = _MockBnModel([("macro_topic_B_01", "macro_topic_B_02")])
    macro_map = {
        "macro_topic_B_01": "B",
        "macro_topic_B_02": "B",
    }
    paths = enumerate_macro_paths(model, macro_map, min_macros=2, max_path_len=4)
    assert paths == []


def test_enumerate_macro_paths_min_three_macros():
    model = _synthetic_model()
    paths = enumerate_macro_paths(model, _macro_map(), min_macros=3, max_path_len=6)
    assert ("macro_topic_A0_01", "macro_topic_A1_01", "macro_topic_B_01", "macro_topic_C_01") in paths
    assert ("macro_topic_A0_01", "macro_topic_B_02", "macro_topic_C_02") in paths
    assert ("macro_topic_A1_01", "macro_topic_B_01") not in paths
    assert ("macro_topic_B_01", "macro_topic_C_01") not in paths


def test_extract_bn_path_scenarios_support_and_fallback():
    model = _synthetic_model()
    topic_cols = [
        "macro_topic_A0_01",
        "macro_topic_A1_01",
        "macro_topic_B_01",
        "macro_topic_B_02",
        "macro_topic_C_01",
        "macro_topic_C_02",
    ]
    df = pd.DataFrame(
        [
            {
                "accident_id": "a1",
                "macro_topic_A0_01": 1,
                "macro_topic_A1_01": 1,
                "macro_topic_B_01": 1,
                "macro_topic_B_02": 0,
                "macro_topic_C_01": 1,
                "macro_topic_C_02": 0,
            },
            {
                "accident_id": "a2",
                "macro_topic_A0_01": 1,
                "macro_topic_A1_01": 1,
                "macro_topic_B_01": 1,
                "macro_topic_B_02": 0,
                "macro_topic_C_01": 1,
                "macro_topic_C_02": 0,
            },
            {
                "accident_id": "a3",
                "macro_topic_B_01": 1,
                "macro_topic_C_01": 1,
            },
        ]
    )
    for col in topic_cols:
        if col not in df.columns:
            df[col] = 0

    freq, diag = extract_bn_path_scenarios(
        df,
        model,
        topic_cols,
        _macro_map(),
        min_support=5,
        top_n=10,
    )
    assert len(freq) >= 1
    assert int(freq.iloc[0]["support"]) >= 2
    assert "macro_path" in freq.columns
    assert diag["n_paths_dag"] >= 1
    assert diag["support_fallback"] is True
