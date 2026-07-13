"""Tests agrégation OOD supervised_macro_ft."""

from __future__ import annotations

import pandas as pd
import pytest

from supervised_macro_ft.eval_summary import (
    build_all_test_corpora_metrics_table,
    resolve_test_corpora,
    summarize_ood_classification,
)


def test_resolve_test_corpora_list_and_legacy():
    assert resolve_test_corpora({"test_corpora": ["metallurgie", "caou"]}) == [
        "metallurgie",
        "caou",
    ]
    assert resolve_test_corpora({"test_corpus": "metallurgie"}) == ["metallurgie"]
    assert resolve_test_corpora({}) == ["metallurgie"]


def test_summarize_ood_classification():
    cv_summary = pd.DataFrame(
        [{"model": "supervised_macro_ft", "mean_balanced_accuracy": 0.80, "std_balanced_accuracy": 0.04}]
    )
    test_metrics = {
        "metallurgie": {"balanced_accuracy": 0.60, "macro_f1": 0.55, "accuracy": 0.58, "loss": 1.2},
        "caou": {"balanced_accuracy": 0.40, "macro_f1": 0.35, "accuracy": 0.38, "loss": 1.5},
    }
    summary = summarize_ood_classification(test_metrics, cv_summary)
    assert summary.loc[0, "ba_ood_avg"] == pytest.approx(0.50)
    assert summary.loc[0, "ba_ood_worst"] == pytest.approx(0.40)
    assert summary.loc[0, "balanced_accuracy_metallurgie"] == pytest.approx(0.60)
    assert summary.loc[0, "balanced_accuracy_caou"] == pytest.approx(0.40)


def test_build_all_test_corpora_metrics_table():
    table = build_all_test_corpora_metrics_table(
        {
            "metallurgie": {"balanced_accuracy": 0.6, "macro_f1": 0.5, "accuracy": 0.55, "loss": 1.0},
            "caou": {"balanced_accuracy": 0.4, "macro_f1": 0.3, "accuracy": 0.35, "loss": 1.1},
        }
    )
    assert list(table["corpus"]) == ["metallurgie", "caou"]
    assert table.loc[1, "balanced_accuracy"] == pytest.approx(0.4)
