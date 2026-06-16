"""Tests agrégation kfold_summary → tableau comparaison BTP μ±σ."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from metrics.compare_display import (
    collect_kfold_btp_comparison,
    format_mean_std,
    kfold_geometry_display_table,
    kfold_ipr_display_table,
)


def test_format_mean_std():
    assert format_mean_std(10.0, 0.5) == "10.00 ± 0.50"
    assert format_mean_std(10.0, 0.0) == "10.00"
    assert format_mean_std(None, 1.0) == "—"


def test_collect_kfold_btp_comparison(tmp_path):
    root = tmp_path / "output"
    triplet = root / "softtriple" / "metrics"
    triplet.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "n_folds": 5,
                "mean_eta2_macro_balanced": 0.12,
                "std_eta2_macro_balanced": 0.01,
                "mean_eta2_macro_balanced_perc": 12.0,
                "std_eta2_macro_balanced_perc": 1.0,
            }
        ]
    ).to_csv(triplet / "kfold_summary.csv", index=False)

    df = collect_kfold_btp_comparison(root)
    assert len(df) == 1
    assert df.iloc[0]["method"] == "SoftTriple"
    assert df.iloc[0]["mean_eta2_macro_balanced_perc"] == 12.0

    disp = kfold_geometry_display_table(df)
    assert "12.00 ± 1.00" in disp.iloc[0]["eta2_macro_balanced_perc"]


def test_collect_kfold_btp_comparison_ipr(tmp_path):
    root = tmp_path / "output"
    triplet = root / "softtriple" / "metrics"
    triplet.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "n_folds": 3,
                "mean_IPR_mean": 1.0,
                "std_IPR_mean": 0.05,
                "mean_IPR_A0": 1.2,
                "std_IPR_A0": 0.1,
            }
        ]
    ).to_csv(triplet / "kfold_summary.csv", index=False)

    df = collect_kfold_btp_comparison(root)
    assert df.iloc[0]["mean_IPR_mean"] == 1.0
    disp = kfold_ipr_display_table(df)
    assert "1.000 ± 0.050" in disp.iloc[0]["IPR_mean"]
