"""Tests macro_transfer.report_tables et topics_export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macro_transfer.bertopic_exports import compute_topic_stats
from macro_transfer.report_tables import load_macro_topic_stats
from macro_transfer.topics_export import build_macro_topic_test_table


def test_compute_topic_stats_largest_topic_id():
    import numpy as np

    ids = np.array([0, 0, 0, 1, 1, -1])
    stats = compute_topic_stats(ids, n_units=len(ids))
    assert stats["largest_topic_id"] == 0
    assert stats["largest_topic_size"] == 3


def test_build_macro_topic_test_table():
    counts = {
        "A0": {
            "n_units": 100,
            "n_topics": 3,
            "noise_rate": 0.1,
            "largest_topic_id": 2,
            "largest_topic_share": 0.4,
        }
    }
    themes = pd.DataFrame(
        [{"macro": "A0", "topic_id": 2, "theme_label": "Chute depuis hauteur"}]
    )
    assignments = pd.DataFrame(
        {"macro": ["A0", "A0"], "topic_id": [2, 2], "doc_idx": [0, 1]}
    )
    df = build_macro_topic_test_table(counts, assignments, themes)
    assert len(df) == 4
    row = df.loc[df["macro"] == "A0"].iloc[0]
    assert row["n_units"] == 100
    assert row["bruit_pct"] == pytest.approx(10.0)
    assert "Chute" in str(row["plus_gros_topic"])


def test_load_macro_topic_stats_from_csv(tmp_path: Path):
    summary = tmp_path / "summary"
    summary.mkdir()
    pd.DataFrame(
        [
            {
                "macro": "B",
                "n_units": 10,
                "n_topics": 2,
                "bruit_pct": 5.0,
                "plus_gros_topic": "Coincement",
                "plus_gros_topic_pct": 20.0,
            }
        ]
    ).to_csv(summary / "macro_topic_stats.csv", index=False)
    df = load_macro_topic_stats(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["macro"] == "B"
