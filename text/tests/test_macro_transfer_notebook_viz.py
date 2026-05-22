"""Tests légers pour macro_transfer.notebook_viz (sans fit BERTopic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_transfer.notebook_viz import (
    RunArtifacts,
    load_run_artifacts,
    merge_assignments,
    _theme_label_map,
)
from safer_core.test_corpus import macro_transfer_output_dir

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_import_notebook_viz():
    import macro_transfer.notebook_viz as nv

    assert hasattr(nv, "plot_global_embedding_map")
    assert hasattr(nv, "plot_topics_per_macro")


def test_macro_transfer_output_dir_convention():
    p = macro_transfer_output_dir("scgm_text", "metallurgie", anchor=TEXT_ROOT)
    assert p.name == "scgm_text"
    assert p.parent.name == "macro_transfer"
    assert "metallurgie" in str(p)


def test_load_run_artifacts_minimal(tmp_path: Path):
    root = tmp_path / "run"
    emb = root / "embeddings"
    emb.mkdir(parents=True)
    z = np.random.randn(20, 8).astype(np.float32)
    np.save(emb / "target_adapted.npy", z)
    transfer = root / "transfer"
    transfer.mkdir()
    meta = pd.DataFrame(
        {
            "sentence": [f"s{i}" for i in range(20)],
            "pred_label": ["A0"] * 5 + ["A1"] * 5 + ["B"] * 5 + ["C"] * 5,
            "m_hat": ["A0"] * 5 + ["A1"] * 5 + ["B"] * 5 + ["C"] * 5,
            "q_conf": np.linspace(0.4, 0.99, 20),
            "p_A0": 0.25,
            "p_A1": 0.25,
            "p_B": 0.25,
            "p_C": 0.25,
        }
    )
    meta.to_csv(transfer / "metadata_with_tpn_macro_probs.csv", index=False)
    with open(transfer / "transfer_metrics_adapted.json", "w", encoding="utf-8") as f:
        json.dump({"n_eval": 20, "accuracy": 1.0}, f)

    art = load_run_artifacts(root)
    assert isinstance(art, RunArtifacts)
    assert art.z.shape == (20, 8)
    assert len(art.meta) == 20


def test_merge_assignments_and_theme_map():
    meta = pd.DataFrame(
        {"m_hat": ["A0", "A0", "A1"], "q_conf": [0.9, 0.3, 0.8], "doc_idx": [0, 1, 2]}
    )
    assign = pd.DataFrame(
        {"doc_idx": [0, 2], "macro": ["A0", "A1"], "topic_id": [0, 1]}
    )
    merged = merge_assignments(meta, assign, confidence_threshold=0.5)
    assert len(merged) == 2
    assert merged.iloc[0]["topic_id"] == 0

    themes = pd.DataFrame(
        {"macro": ["A0"], "topic_id": [0], "top_words": "acier fusion"}
    )
    labels = _theme_label_map(themes)
    assert labels[("A0", 0)].startswith("A0|T0")
