from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_transfer_error_diagnostics import (
    bootstrap_overlap,
    factorial_ss,
    nearest_source,
    read_predictions,
)


def test_prediction_alignment_rejects_duplicate_keys_and_wrong_labels(tmp_path):
    expected = pd.DataFrame({"pred_label": ["A0", "A1"]},
        index=pd.MultiIndex.from_tuples([("a", "1"), ("a", "2")], names=["accident_id", "fact_id"]))
    path = tmp_path / "predictions.csv"
    rows = pd.DataFrame({"accident_id": ["a", "a"], "fact_id": ["1", "2"],
                         "true_macro": ["A0", "A1"], "pred_macro": ["A0", "B"],
                         "confidence": [0.7, 0.8]})
    rows.to_csv(path, index=False)
    assert list(read_predictions(path, expected).pred_macro) == ["A0", "B"]
    rows.loc[1, "fact_id"] = "1"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_predictions(path, expected)
    rows.loc[1, "fact_id"] = "2"
    rows.loc[1, "true_macro"] = "C"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="True labels"):
        read_predictions(path, expected)


def test_bootstrap_keeps_seed_and_method_errors_on_same_accidents():
    errors = np.zeros((5, 3, 4), dtype=np.uint8)
    errors[:, :, :2] = 1
    results = bootstrap_overlap(errors, np.array(["a", "a", "b", "b"]), n_boot=200, seed=9)
    assert results["common_unit_ci_low"] >= 0
    assert results["common_unit_ci_high"] <= 1
    assert results["common_unit_ci_low"] <= .5 <= results["common_unit_ci_high"]
    assert results["common_among_any_ci_high"] == 1


def test_factorial_decomposition_is_orthogonal_for_balanced_cells():
    rows = []
    for target in ("a", "b"):
        for method in ("x", "y"):
            for seed in range(5):
                rows.append({"target": target, "method": method, "score":
                             (1 if target == "a" else 3) + (0 if method == "x" else .25) + seed*.01})
    result = factorial_ss(pd.DataFrame(rows), "score", ("target", "method"))
    assert np.isclose(result.share_total_variation.sum(), 1)
    assert result.set_index("term").loc["target", "sum_squares"] > result.set_index("term").loc["method", "sum_squares"]


def test_nearest_source_returns_cosine_distance_and_label_entropy():
    source = np.eye(4, dtype=np.float32)
    target = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    distance, entropy = nearest_source(source, target, np.array([0, 0, 1, 1]), k=2, block=1)
    assert np.allclose(distance, [0.5, 0.5])
    assert np.all(entropy >= 0)
