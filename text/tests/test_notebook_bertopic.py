"""Tests macro_transfer.notebook_bertopic (sans fit BERTopic réel)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from macro_transfer.notebook_bertopic import (
    build_gating_from_true_labels,
    build_notebook_bertopic_summary_table,
    build_transfer_predictions_csv,
    load_notebook_bertopic_config,
    resolve_gating,
)


def test_build_gating_from_true_labels_schema():
    meta = pd.DataFrame({"pred_label": ["A0", "A1", "B", "C"]})
    g = build_gating_from_true_labels(meta)
    assert list(g.columns) == ["m_hat", "ambiguous", "q_conf", "p_A0", "p_A1", "p_B", "p_C"]
    assert g.loc[0, "m_hat"] == "A0"
    assert g.loc[0, "p_A0"] == 1.0
    assert g.loc[2, "p_B"] == 1.0


def test_resolve_gating_predicted_vs_true_label():
    meta = pd.DataFrame({"pred_label": ["A0", "B"]})
    preds = pd.DataFrame(
        {
            "pred_macro": ["A1", "C"],
            "prob_A0": [0.1, 0.1],
            "prob_A1": [0.7, 0.1],
            "prob_B": [0.1, 0.1],
            "prob_C": [0.1, 0.7],
            "confidence": [0.7, 0.7],
        }
    )
    g_pred = resolve_gating(meta, preds, mode="predicted")
    assert "m_hat" in g_pred.columns
    g_true = resolve_gating(meta, preds, mode="true_label")
    assert g_true.loc[0, "m_hat"] == "A0"


def test_load_notebook_bertopic_config_fusion(text_root: Path):
    cfg = load_notebook_bertopic_config(anchor=text_root)
    assert "bertopic" in cfg
    assert cfg.get("notebook", {}).get("segment_mode") == "predicted"
    assert int(cfg["bertopic"].get("min_topic_size", 0)) >= 1


def test_build_notebook_bertopic_summary_table_on_fixtures():
    macro_counts = {"A0": {"n_topics": 2, "n_outliers": 1, "n_units": 10}}
    assignments = pd.DataFrame(
        {
            "doc_idx": [0, 1, 2],
            "macro": ["A0", "A0", "A0"],
            "topic_id": [0, 0, -1],
            "gamma": [0.9, 0.8, 0.1],
        }
    )
    themes = pd.DataFrame(
        {
            "macro": ["A0", "A0"],
            "topic_id": [0, 1],
            "theme_label": ["thème A", "thème B"],
            "n_units": [5, 4],
            "top_words": ["a b", "c d"],
        }
    )
    summary = build_notebook_bertopic_summary_table(macro_counts, assignments, themes)
    assert not summary.empty
    assert "macro" in summary.columns


def test_build_transfer_predictions_csv(tmp_path):
    meta = pd.DataFrame(
        {
            "accident_id": ["a1", "a2"],
            "doc_id": [0, 1],
            "sentence": ["s1", "s2"],
            "pred_label": ["A0", "B"],
        }
    )
    preds = pd.DataFrame(
        {
            "pred_macro": ["A1", "B"],
            "confidence": [0.8, 0.9],
            "prob_A0": [0.1, 0.1],
            "prob_A1": [0.7, 0.1],
            "prob_B": [0.1, 0.7],
            "prob_C": [0.1, 0.1],
        }
    )
    out = build_transfer_predictions_csv(meta, preds, tmp_path / "transfer" / "target_macro_predictions.csv")
    df = pd.read_csv(out)
    assert "m_hat" in df.columns
    assert "p_A0" in df.columns or "prob_A0" in df.columns or "pred_macro" in df.columns


@pytest.fixture
def text_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_notebook_bertopic_mock_fit(tmp_path, text_root):
    from macro_transfer.notebook_bertopic import run_notebook_bertopic

    results_dir = tmp_path / "output" / "batch_triplet"
    emb_dir = results_dir / "embeddings"
    emb_dir.mkdir(parents=True)
    n = 4
    import numpy as np

    np.save(emb_dir / "projected_metallurgie.npy", np.random.randn(n, 8))
    meta = pd.DataFrame(
        {
            "accident_id": ["a1"] * n,
            "doc_id": range(n),
            "sentence": [f"s{i}" for i in range(n)],
            "pred_label": ["A0", "A1", "B", "C"],
        }
    )
    meta.to_csv(emb_dir / "projected_metallurgie_metadata.csv", index=False)
    np.save(emb_dir / "projected_btp.npy", np.random.randn(n, 8))
    btp_meta = meta.copy()
    btp_meta.to_csv(emb_dir / "projected_btp_metadata.csv", index=False)

    themes = pd.DataFrame(
        {
            "macro": ["A0"],
            "topic_id": [0],
            "theme_label": ["test"],
            "n_units": [2],
            "top_words": ["x y"],
        }
    )
    assignments = pd.DataFrame(
        {
            "doc_idx": [0, 1],
            "macro": ["A0", "A1"],
            "topic_id": [0, 0],
            "gamma": [0.9, 0.8],
        }
    )
    partial = {"macro_topic_counts": {"A0": {"n_topics": 1}}, "warnings": []}

    with patch("macro_transfer.notebook_bertopic.fit_bertopic_per_macro", return_value=(themes, assignments, partial)):
        with patch("macro_transfer.notebook_bertopic.export_bertopic_datamaps_from_run"):
            with patch("macro_transfer.notebook_bertopic.load_notebook_bertopic_config") as mock_cfg:
                mock_cfg.return_value = {
                    "notebook": {"output_subdir": "bertopic_notebook", "save_datamap": False, "export_for_bn": True},
                    "bertopic": {"min_topic_size": 2},
                    "topics_export": {},
                }
                out = run_notebook_bertopic(
                    results_dir,
                    "metallurgie",
                    method_name="batch_triplet",
                    view_kind="contrastive",
                    bertopic_cfg={"min_topic_size": 2, "diagnostics": {"save_datamap": False}},
                    topics_export_cfg={},
                    anchor=text_root,
                    method_key="batch_triplet",
                )
    assert (out / "transfer" / "target_macro_predictions.csv").is_file()
    assert (out / "bertopic_manifest.json").is_file()
    manifest = json.loads((out / "bertopic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_id"] == "metallurgie"
