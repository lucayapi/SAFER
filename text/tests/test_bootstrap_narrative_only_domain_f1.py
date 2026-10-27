"""Regression checks for the paired, corpus-level F1 bootstrap."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bootstrap_narrative_only_domain_f1 import paired_bootstrap  # noqa: E402


def test_paired_bootstrap_recomputes_micro_f1_from_whole_narratives():
    # The second narrative has many gold units. Averaging document F1 values
    # would yield a different point estimate from corpus-level span F1.
    counts = np.array([
        [1, 1, 1, 1, 1],
        [9, 9, 0, 9, 9],
    ], dtype=np.int64)
    result = paired_bootstrap(counts, 1000, np.random.default_rng(7))
    assert result["m2_f1"] == 0.1
    assert result["m3_f1"] == 1.0
    assert result["delta_f1"] == 0.9
    assert result["delta_ci_low"] >= 0
    assert result["delta_ci_high"] <= 1


def test_identical_paired_predictions_have_zero_difference_interval():
    counts = np.array([
        [2, 3, 1, 3, 1],
        [5, 4, 3, 4, 3],
        [1, 2, 0, 2, 0],
    ], dtype=np.int64)
    result = paired_bootstrap(counts, 1000, np.random.default_rng(19))
    assert result["delta_f1"] == 0
    assert result["delta_ci_low"] == 0
    assert result["delta_ci_high"] == 0
