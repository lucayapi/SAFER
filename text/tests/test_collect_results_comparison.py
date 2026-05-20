"""Tests agrégation BTP / test pour notebook 01."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from metrics.compare_display import (
    EMBEDDING_COMPARE_METHODS,
    METHOD_DISPLAY,
    fill_eta2_macro_balanced_perc,
    fill_rankme_over_d,
    normalize_method_display_name,
    order_methods,
    slim_geometry_table,
)
from scripts.collect_results import (
    _load_method_row,
    _load_method_row_for_corpus,
    collect_embedding_comparison,
)


def _write_metrics(metrics_dir: Path, stem: str, method_label: str, eta2: float) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"method": method_label, "eta2_macro_balanced": eta2, "eta2_macro_balanced_perc": eta2 * 100}]
    ).to_csv(metrics_dir / f"{stem}.csv", index=False)


def test_load_method_row_btp_fallback(tmp_path):
    m = tmp_path / "raw_embedding" / "metrics"
    _write_metrics(m, "metrics_geometry", "Embedding brut", 0.1)
    row = _load_method_row(tmp_path / "raw_embedding")
    assert row is not None
    assert row["method"] == "Embedding brut"


def test_load_method_row_for_corpus_test(tmp_path):
    m = tmp_path / "supcon" / "metrics"
    _write_metrics(m, "metrics_geometry_test", "SupCon_test", 0.2)
    row = _load_method_row_for_corpus(tmp_path / "supcon", "test")
    assert row is not None


def test_collect_embedding_comparison_order_includes_raw(tmp_path):
    for key, eta2 in [
        ("raw_embedding", 0.1),
        ("batch_triplet", 0.3),
        ("supcon", 0.5),
        ("scgm_text", 0.4),
    ]:
        _write_metrics(
            tmp_path / key / "metrics",
            "metrics_geometry_btp",
            METHOD_DISPLAY[key],
            eta2,
        )
    df = collect_embedding_comparison(tmp_path, corpus="btp")
    assert list(df["method"]) == [
        METHOD_DISPLAY["raw_embedding"],
        METHOD_DISPLAY["batch_triplet"],
        METHOD_DISPLAY["supcon"],
        METHOD_DISPLAY["scgm_text"],
    ]
    assert len(df) == 4


def test_fill_eta2_perc_from_balanced_when_nan():
    df = pd.DataFrame(
        {
            "method": ["SoftTriple_btp"],
            "eta2_macro_balanced": [0.65124],
            "eta2_macro_balanced_perc": [float("nan")],
        }
    )
    out = fill_eta2_macro_balanced_perc(df)
    assert abs(out.iloc[0]["eta2_macro_balanced_perc"] - 65.124) < 0.01


def test_fill_rankme_over_d_when_embedding_dim_column_is_nan():
    df = pd.DataFrame(
        {
            "method": ["Embedding brut", "SoftTriple", "SCGM", "SupCon"],
            "rankme_global": [649.65, 604.1, 57.9, 520.5],
            "embedding_dim": [float("nan"), float("nan"), float("nan"), 1024.0],
        }
    )
    out = fill_rankme_over_d(df)
    assert out.loc[0, "embedding_dim"] == 1024
    assert out.loc[2, "embedding_dim"] == 128
    assert abs(out.loc[0, "rankme_over_d"] - 649.65 / 1024) < 0.01
    assert abs(out.loc[2, "rankme_over_d"] - 57.9 / 128) < 0.01


def test_normalize_method_display_name_strips_suffix():
    assert normalize_method_display_name("Batch Triplet_btp", "batch_triplet") == "Batch Triplet"


def test_raw_test_uses_raw_embedding_test_dir(tmp_path):
    _write_metrics(
        tmp_path / "metallurgie" / "raw_embedding" / "metrics",
        "metrics_geometry",
        "Embedding brut_test",
        0.08,
    )
    df = collect_embedding_comparison(tmp_path, corpus="test", test_corpus_id="metallurgie")
    assert len(df) == 1
    assert df.iloc[0]["method"] == "Embedding brut"


def test_collect_btp_and_test_separate(tmp_path):
    _write_metrics(tmp_path / "supcon" / "metrics", "metrics_geometry_btp", "SupCon", 0.5)
    _write_metrics(
        tmp_path / "metallurgie" / "supcon" / "metrics",
        "metrics_geometry_test",
        "SupCon_test",
        0.2,
    )
    df_btp = collect_embedding_comparison(tmp_path, corpus="btp")
    df_test = collect_embedding_comparison(tmp_path, corpus="test", test_corpus_id="metallurgie")
    assert len(df_btp) == 1
    assert len(df_test) == 1


def test_slim_geometry_table_columns():
    df = pd.DataFrame(
        {
            "method": ["SCGM"],
            "eta2_macro_balanced_perc": [10.0],
            "rankme_global": [5.0],
            "W_A0": [1.0],
        }
    )
    slim = slim_geometry_table(df)
    assert "W_A0" not in slim.columns
    assert "eta2_macro_balanced_perc" in slim.columns
    assert "rankme_over_d" in slim.columns


def test_generated_notebook_01_references_two_tables():
    nb_path = TEXT_ROOT / "notebooks" / "01_compare_embedding_methods.ipynb"
    if not nb_path.is_file():
        pytest.skip("notebook not generated — run build_analysis_notebooks.py")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    src = "".join("".join(c.get("source", [])) for c in data["cells"])
    assert "embedding_geometry_comparison_btp.csv" in src
    assert "embedding_geometry_comparison_test.csv" in src
    assert "métallurgie" in src
    assert "slim_geometry_table" in src
