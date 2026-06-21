"""Tests paramètres BERTopic par macro."""

from __future__ import annotations

import json
from pathlib import Path

from macro_transfer.bertopic_utils import (
    build_bertopic_for_macro,
    resolve_macro_bertopic_params,
    save_macro_bertopic_config,
)


def _sample_cfg():
    return {
        "min_topic_size": 10,
        "random_state": 42,
        "hdbscan": {"cluster_selection_method": "eom", "min_samples": None},
        "umap": {"enabled": True, "n_neighbors": 15, "n_components": 5, "min_dist": 0.1},
        "vectorizer": {"min_df": 1, "ngram_range": [1, 2]},
        "macro_params": {
            "A0": {
                "min_topic_size": 25,
                "min_cluster_size": 25,
                "min_samples": 3,
                "cluster_selection_method": "leaf",
                "n_neighbors": 10,
            },
            "B": {
                "min_topic_size": 35,
                "min_cluster_size": 35,
                "cluster_selection_method": "eom",
            },
        },
    }


def test_a0_uses_leaf_cluster_selection():
    p = resolve_macro_bertopic_params("A0", _sample_cfg())
    assert p["cluster_selection_method"] == "leaf"
    assert p["min_cluster_size"] == 25
    assert p["min_samples"] == 3
    assert p["n_neighbors"] == 10


def test_frozen_source_prototypes_macro_params_eom_all_macros():
    from macro_transfer.bertopic_config import load_bertopic_macro_shared

    shared = load_bertopic_macro_shared(
        anchor=Path(__file__).resolve().parents[1]
    )
    macro_params = shared["bertopic"]["macro_params"]
    for macro in ("A0", "A1", "B", "C"):
        assert macro in macro_params
        block = macro_params[macro]
        assert block["cluster_selection_method"] == "eom"
        assert block["min_cluster_size"] == 8
        assert block["min_samples"] == 2
        assert block["min_topic_size"] == 8


def test_b_uses_eom():
    p = resolve_macro_bertopic_params("B", _sample_cfg())
    assert p["cluster_selection_method"] == "eom"
    assert p["min_cluster_size"] == 35


def test_build_bertopic_for_macro_instantiates():
    pytest = __import__("pytest")
    pytest.importorskip("bertopic")
    pytest.importorskip("hdbscan")
    pytest.importorskip("umap")
    cfg = {**_sample_cfg(), "representation": {"enabled": False}}
    model = build_bertopic_for_macro(
        "A0",
        cfg,
        random_state=42,
        disable_representation=True,
    )
    assert model is not None
    assert model.hdbscan_model.cluster_selection_method == "leaf"


def test_save_macro_bertopic_config_json(tmp_path):
    out = tmp_path / "config_used.json"
    save_macro_bertopic_config("A1", _sample_cfg(), out, extra={"embedding_mode": "mixed"})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["macro"] == "A1"
    assert "resolved_params" in data
    assert data["embedding_mode"] == "mixed"
