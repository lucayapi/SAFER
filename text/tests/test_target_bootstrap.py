"""Tests de l'IC bootstrap cible utilisé par les notebooks 07 et 07c."""

from __future__ import annotations

import pandas as pd

from macro_transfer.target_bootstrap import bootstrap_target_predictions


def test_bootstrap_resamples_complete_accidents_and_exports_cache(tmp_path):
    truth = ["A0", "A1", "B", "C", "A0", "A1", "B", "C"]
    accidents = ["a", "a", "b", "b", "c", "c", "d", "d"]
    predictions = {
        "correct": pd.DataFrame({
            "accident_id": accidents, "true_macro": truth, "pred_macro": truth,
        }),
        "wrong": pd.DataFrame({
            "accident_id": accidents, "true_macro": truth, "pred_macro": ["A0"] * len(truth),
        }),
    }
    destination = tmp_path / "bootstrap_ci.csv"
    intervals = bootstrap_target_predictions(
        predictions, destination=destination, n_resamples=100, seed=17
    )

    assert destination.is_file()
    assert destination.with_name("bootstrap_ci_manifest.json").is_file()
    assert set(intervals["resampling_unit"]) == {"accident_id"}
    assert set(intervals["n_accidents"]) == {4}
    assert set(intervals["n_units"]) == {8}
    assert set(intervals["n_resamples"]) == {100}
    assert set(intervals["metric"]) == {"balanced_accuracy", "macro_f1"}
    assert intervals.loc[intervals["model"].eq("correct"), "point_estimate"].eq(1.0).all()
    assert intervals.loc[intervals["model"].eq("wrong"), "point_estimate"].lt(1.0).all()

    # Même prédictions et mêmes paramètres : le résultat rechargé est identique.
    cached = bootstrap_target_predictions(
        predictions, destination=destination, n_resamples=100, seed=17
    )
    pd.testing.assert_frame_equal(intervals, cached)

