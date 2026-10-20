"""Checks for HDBSCAN clustering directly on the configured embeddings."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEXT_ROOT / "recurrent_scenarios"))

import scenario_pipeline as pipeline


def raw_leaf_config():
    return {
        "screening": {
            "umap": {"enabled": False},
            "hdbscan": {
                "min_cluster_size": [15, 25, 50],
                "min_samples": [5, 10],
                "cluster_selection_method": ["leaf"],
                "metric": "cosine",
                "algorithm": "generic",
                "prediction_data": False,
            },
        }
    }


def test_raw_embedding_grid_only_varies_hdbscan_parameters():
    plan = pipeline.parameter_plan(raw_leaf_config())
    assert len(plan) == 6
    assert all("umap_n_neighbors" not in params for params in plan)
    assert {params["hdbscan_cluster_selection_method"] for params in plan} == {"leaf"}


def test_fit_skips_umap_and_uses_original_embeddings(monkeypatch):
    seen = {}

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def fit_predict(self, matrix):
            seen["matrix"] = matrix
            self.probabilities_ = np.ones(len(matrix), dtype=np.float32)
            return np.array([0, 0, 1, 1], dtype=int)

    fake_hdbscan = types.SimpleNamespace(HDBSCAN=FakeHDBSCAN)
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)
    embeddings = np.arange(12, dtype=np.float32).reshape(4, 3)
    params = {
        "hdbscan_min_cluster_size": 2,
        "hdbscan_min_samples": 1,
        "hdbscan_cluster_selection_method": "leaf",
    }

    labels, cluster_space, strength = pipeline._fit_cluster_with_embedding(
        [], embeddings, params, 42, raw_leaf_config(), return_membership_strength=True
    )

    np.testing.assert_array_equal(seen["matrix"], embeddings)
    assert seen["matrix"].dtype == np.float64
    np.testing.assert_array_equal(cluster_space, embeddings)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    np.testing.assert_array_equal(strength, np.ones(4, dtype=np.float32))
    assert seen["kwargs"]["metric"] == "cosine"
    assert seen["kwargs"]["algorithm"] == "generic"
    assert seen["kwargs"]["cluster_selection_method"] == "leaf"


def test_hdbscan_cosine_and_dbcv_run_on_direct_embedding_space():
    pytest.importorskip("hdbscan")
    rng = np.random.default_rng(4)
    embeddings = np.vstack([
        rng.normal(1.0, 0.1, size=(20, 4)),
        rng.normal(-1.0, 0.1, size=(20, 4)),
    ]).astype(np.float32)
    config = raw_leaf_config()
    params = {
        "hdbscan_min_cluster_size": 5,
        "hdbscan_min_samples": 3,
        "hdbscan_cluster_selection_method": "leaf",
    }

    labels, cluster_space = pipeline._fit_cluster_with_embedding(
        [], embeddings, params, 42, config
    )
    dbcv = pipeline._dbcv(cluster_space, labels, metric="cosine")

    assert cluster_space is embeddings
    assert len(set(labels) - {-1}) == 2
    assert np.isfinite(dbcv)
