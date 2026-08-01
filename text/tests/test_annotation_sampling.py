"""Tests sous-échantillonnage annotation."""

from __future__ import annotations

import pandas as pd

from annotation.sampling import sample_accidents_and_units, sampling_stats


def _df() -> pd.DataFrame:
    rows = []
    for acc in ("acc1", "acc2", "acc3"):
        for fid in range(1, 5):
            rows.append(
                {
                    "accident_id": acc,
                    "fact_id": fid,
                    "sentence": f"unit {acc}-{fid}",
                }
            )
    return pd.DataFrame(rows)


def test_sample_n_accidents_reproducible():
    df = _df()
    a = sample_accidents_and_units(df, n_accidents=2, units_per_accident="all", seed=7)
    b = sample_accidents_and_units(df, n_accidents=2, units_per_accident="all", seed=7)
    assert a.equals(b)
    assert a["accident_id"].nunique() == 2


def test_sample_units_per_accident():
    df = _df()
    out = sample_accidents_and_units(df, n_accidents="all", units_per_accident=2, seed=0)
    assert len(out) == 3 * 2
    for acc, grp in out.groupby("accident_id"):
        assert len(grp) == 2


def test_sampling_stats():
    df = _df()
    stats = sampling_stats(df)
    assert stats["n_rows"] == 12
    assert stats["n_accidents"] == 3
    assert stats["estimated_api_calls"] == 12


def test_sample_accident_frac_reproducible():
    df = _df()
    a = sample_accidents_and_units(
        df, accident_sample_frac=0.3, units_per_accident="all", seed=42
    )
    b = sample_accidents_and_units(
        df, accident_sample_frac=0.3, units_per_accident="all", seed=42
    )
    assert a.equals(b)
    assert a["accident_id"].nunique() == 1


def test_sample_accident_frac_ten_accidents():
    rows = []
    for i in range(10):
        for fid in range(2):
            rows.append(
                {
                    "accident_id": f"acc{i}",
                    "fact_id": fid,
                    "sentence": f"s{i}-{fid}",
                }
            )
    df = pd.DataFrame(rows)
    out = sample_accidents_and_units(
        df, accident_sample_frac=0.3, units_per_accident="all", seed=0
    )
    assert out["accident_id"].nunique() == 3
    assert len(out) == 6


def test_sample_n_accidents_keeps_all_units():
    df = _df()
    out = sample_accidents_and_units(df, n_accidents=2, units_per_accident="all", seed=0)
    assert out["accident_id"].nunique() == 2
    for _, grp in out.groupby("accident_id"):
        assert len(grp) == 4
