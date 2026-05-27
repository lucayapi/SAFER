"""Tests métriques tuning λ_pres."""

from __future__ import annotations

import pandas as pd
import pytest

from macro_transfer.tune_preserve import (
    aggregate_topic_metrics_from_stats,
    lambda_pres_run_dir,
    lambda_pres_tag,
    metrics_rows_to_dataframe,
    parse_lambda_pres_list,
)


def test_lambda_pres_tag():
    assert lambda_pres_tag(0.0) == "0"
    assert lambda_pres_tag(0.25) == "0p25"
    assert lambda_pres_run_dir("/tmp/tune", 0.1).name == "lambda_0p1"


def test_parse_lambda_pres_list():
    assert parse_lambda_pres_list("0, 0.05, 0.25") == [0.0, 0.05, 0.25]


def test_aggregate_topic_metrics_weighted():
    stats = pd.DataFrame(
        [
            {"macro": "A0", "n_units": 100, "n_topics": 4, "bruit_pct": 10.0, "plus_gros_topic_pct": 40.0},
            {"macro": "B", "n_units": 300, "n_topics": 2, "bruit_pct": 20.0, "plus_gros_topic_pct": 60.0},
        ]
    )
    agg = aggregate_topic_metrics_from_stats(stats)
    # R_m weighted: (0.4*100 + 0.6*300) / 400 = 0.55
    assert agg["R_m"] == pytest.approx(0.55)
    assert agg["K_m"] == pytest.approx((4 * 100 + 2 * 300) / 400)
    assert agg["r_noise"] == pytest.approx((0.1 * 100 + 0.2 * 300) / 400)
    assert agg["R_m_A0"] == pytest.approx(0.4)
    assert agg["K_m_B"] == pytest.approx(2.0)


def test_metrics_rows_to_dataframe_sort():
    rows = [
        {"base_method": "scgm_text", "lambda_pres": 0.25},
        {"base_method": "scgm_text", "lambda_pres": 0.0},
    ]
    df = metrics_rows_to_dataframe(rows)
    assert list(df["lambda_pres"]) == [0.0, 0.25]
