"""Tests post-évaluation classification contrastive."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.post_eval import (
    CLASSIFICATION_METRIC_KEYS,
    evaluate_classifier_on_embeddings,
    fit_classifier_on_embeddings,
)


def _synthetic_xy(n_per_class: int = 20, dim: int = 8):
    rng = np.random.RandomState(0)
    parts_x, parts_y = [], []
    for cls in range(4):
        center = np.zeros(dim, dtype=np.float64)
        center[cls % dim] = 3.0
        X = rng.randn(n_per_class, dim) * 0.2 + center
        y = np.full(n_per_class, cls, dtype=np.int64)
        parts_x.append(X)
        parts_y.append(y)
    return np.vstack(parts_x), np.concatenate(parts_y)


def test_fit_and_eval_logistic_on_synthetic_embeddings():
    X, y = _synthetic_xy()
    cfg = ContrastiveConfig(
        method_name="supcon",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        post_eval_enabled=True,
        post_eval_classifier="logistic_regression",
    )
    pipe = fit_classifier_on_embeddings(X, y, cfg, seed=42)
    macros = ["A0", "A1", "B", "C"]
    y_macro = np.array([macros[i] for i in y], dtype=object)
    metrics = evaluate_classifier_on_embeddings(pipe, X, y_macro, macros=macros)
    for key in CLASSIFICATION_METRIC_KEYS:
        assert key in metrics
        assert float(metrics[key]) > 0.9


def test_post_eval_oversampling_vs_class_weight_conflict():
    from contrastive_methods.config_validation import validate_contrastive_config

    cfg = ContrastiveConfig(
        method_name="supcon",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        post_eval_oversampling=True,
        post_eval_class_weight="balanced",
    )
    with pytest.raises(ValueError, match="post_eval"):
        validate_contrastive_config(cfg)
