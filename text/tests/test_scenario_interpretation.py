"""Tests bn_pipeline.scenario_interpretation (sans appel API)."""

from __future__ import annotations

import pandas as pd
import pytest

from bn_pipeline.scenario_interpretation import (
    build_configuration_probable,
    build_topic_variable_label_map,
    enrich_scenarios_table,
)


def test_build_configuration_probable():
    themes = pd.DataFrame(
        [
            {
                "macro": "A0",
                "topic_id": 3,
                "theme_label": "Absence de protection collective",
            }
        ]
    )
    label_map = build_topic_variable_label_map(themes)
    row = pd.Series(
        {
            "topics_present": "macro_topic_A0_03 + macro_topic_B_01",
            "macro_path": "A0 -> B",
            "support": 10,
        }
    )
    cfg = build_configuration_probable(row, label_map)
    assert "Absence" in cfg or "A0" in cfg
    assert "→" in cfg


def test_enrich_scenarios_table_no_openai():
    freq = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "macro_path": "A0 -> B",
                "topics_present": "macro_topic_A0_03",
                "support": 5,
                "representative_sentences": "exemple accident",
            }
        ]
    )
    out = enrich_scenarios_table(
        freq,
        n_accidents=20,
        themes_df=pd.DataFrame(),
        enable_openai=False,
        max_rows=5,
    )
    assert "prob" in out.columns
    assert out.iloc[0]["prob"] == pytest.approx(0.25)
