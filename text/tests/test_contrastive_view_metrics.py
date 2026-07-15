"""Tests tableau unifié notebooks contrastif."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.view_metrics import (
    build_view_classification_summary_table,
    format_ood_summary_line,
    validate_contrastive_results_dir,
)


def _write_fixture(tmp_path: Path) -> Path:
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "mean_val_balanced_accuracy": 0.75,
                "std_val_balanced_accuracy": 0.02,
                "mean_val_macro_f1": 0.71,
                "std_val_macro_f1": 0.03,
                "mean_val_accuracy": 0.73,
                "std_val_accuracy": 0.02,
            }
        ]
    ).to_csv(metrics / "kfold_summary.csv", index=False)
    pd.DataFrame(
        [{"corpus": "btp", "balanced_accuracy": 0.83, "macro_f1": 0.79, "accuracy": 0.81}]
    ).to_csv(metrics / "metrics_classification_btp.csv", index=False)
    pd.DataFrame(
        [{"corpus": "metallurgie", "balanced_accuracy": 0.55, "macro_f1": 0.48, "accuracy": 0.52}]
    ).to_csv(metrics / "metrics_classification_test_metallurgie.csv", index=False)
    pd.DataFrame(
        [{"corpus": "caou", "balanced_accuracy": 0.41, "macro_f1": 0.35, "accuracy": 0.40}]
    ).to_csv(metrics / "metrics_classification_test_caou.csv", index=False)
    pd.DataFrame(
        [{"ba_ood_avg": 0.48, "ba_ood_worst": 0.41}]
    ).to_csv(metrics / "cross_domain_generalization.csv", index=False)
    (tmp_path / "checkpoints" / "best_model").mkdir(parents=True)
    (tmp_path / "checkpoints" / "best_model" / "config.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_validate_contrastive_results_dir(tmp_path):
    root = _write_fixture(tmp_path)
    assert validate_contrastive_results_dir(root) == root.resolve()


def test_build_view_classification_summary_table(tmp_path):
    root = _write_fixture(tmp_path)
    table = build_view_classification_summary_table(root, test_corpora=["metallurgie", "caou"])
    assert list(table["phase"]) == ["cv_val", "lr_eval", "lr_eval", "lr_eval"]
    assert list(table["corpus"]) == ["btp", "btp", "metallurgie", "caou"]
    assert table.loc[0, "balanced_accuracy"] == "0.750 ± 0.020"
    assert table.loc[1, "balanced_accuracy"] == "0.830"
    assert table.loc[2, "balanced_accuracy"] == "0.550"


def test_format_ood_summary_line(tmp_path):
    root = _write_fixture(tmp_path)
    line = format_ood_summary_line(root)
    assert line is not None
    assert "BA moyenne : 0.480" in line
    assert "BA pire corpus : 0.410" in line


def test_build_macro_ft_classification_summary_table():
    from contrastive_methods.view_metrics import build_macro_ft_classification_summary_table

    cv = pd.DataFrame(
        [{"mean_balanced_accuracy": 0.80, "std_balanced_accuracy": 0.04, "mean_macro_f1": 0.75, "std_macro_f1": 0.03, "mean_accuracy": 0.78, "std_accuracy": 0.03}]
    )
    metrics = {
        "btp": pd.DataFrame([{"balanced_accuracy": 0.85, "macro_f1": 0.80, "accuracy": 0.82}]),
        "metallurgie": pd.DataFrame([{"balanced_accuracy": 0.55, "macro_f1": 0.48, "accuracy": 0.52}]),
    }
    table = build_macro_ft_classification_summary_table(cv, metrics, test_corpora=["metallurgie"])
    assert list(table["phase"]) == ["cv_val", "lr_eval", "lr_eval"]
    assert list(table["corpus"]) == ["btp", "btp", "metallurgie"]
