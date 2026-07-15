"""Tests eval_corpus classification (mock)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))


@patch("contrastive_methods.eval_corpus.build_and_save_predictions")
@patch("contrastive_methods.eval_corpus.save_classification_outputs")
@patch("contrastive_methods.eval_corpus.evaluate_classifier_on_embeddings")
@patch("contrastive_methods.eval_corpus.fit_classifier_on_embeddings")
@patch("contrastive_methods.eval_corpus.export_projected_embeddings")
@patch("contrastive_methods.eval_corpus._encode_corpus_df")
@patch("contrastive_methods.eval_corpus.prepare_text_dataset")
def test_run_final_classification_eval_mock(
    mock_dataset,
    mock_encode,
    mock_export,
    mock_fit,
    mock_eval_cls,
    mock_save,
    mock_save_preds,
    tmp_path,
):
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.eval_corpus import run_final_classification_eval

    meta = pd.DataFrame(
        {
            "doc_id": list(range(8)),
            "sentence": ["x"] * 8,
            "pred_label": ["A0", "A1", "B", "C"] * 2,
            "label_id": [0, 1, 2, 3] * 2,
            "accident_id": list(range(8)),
        }
    )
    ds = MagicMock()
    ds.metadata_df = meta
    mock_dataset.return_value = ds
    mock_encode.return_value = np.random.randn(8, 4)
    mock_fit.return_value = MagicMock()
    details = {
        "pred_macro": np.array(["A0", "A1", "B", "C"] * 2, dtype=object),
        "probs": np.eye(4)[np.array([0, 1, 2, 3] * 2)],
        "confidence": np.ones(8),
        "margin": np.ones(8) * 0.5,
        "entropy": np.zeros(8),
        "macros": ["A0", "A1", "B", "C"],
    }
    mock_eval_cls.return_value = (
        {"balanced_accuracy": 0.5, "macro_f1": 0.4, "accuracy": 0.45},
        details,
    )
    mock_save.return_value = {"btp": tmp_path / "btp.csv"}
    mock_save_preds.return_value = (pd.DataFrame(), tmp_path / "predictions" / "predictions_btp.csv")

    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=tmp_path / "data.csv",
        test_corpora=[],
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    run_final_classification_eval(cfg, ckpt, tmp_path / "out")
    assert mock_save.called
    assert mock_save_preds.called
    assert mock_eval_cls.call_args.kwargs.get("return_details") is True


def test_run_final_classification_eval_writes_predictions(tmp_path, monkeypatch):
    """Intégration légère : vrai evaluate + écriture CSV prédictions."""
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods import eval_corpus as ec

    meta = pd.DataFrame(
        {
            "doc_id": list(range(12)),
            "sentence": [f"s{i}" for i in range(12)],
            "pred_label": ["A0", "A1", "B", "C"] * 3,
            "label_id": [0, 1, 2, 3] * 3,
            "accident_id": list(range(12)),
        }
    )
    ds = MagicMock()
    ds.metadata_df = meta

    X = np.random.RandomState(0).randn(12, 8)
    for i, lid in enumerate(meta["label_id"]):
        X[i, lid] += 5.0

    monkeypatch.setattr(ec, "prepare_text_dataset", lambda cfg: ds)
    monkeypatch.setattr(ec, "_encode_corpus_df", lambda *a, **k: X)
    monkeypatch.setattr(ec, "export_projected_embeddings", lambda *a, **k: (tmp_path / "e.npy", tmp_path / "m.csv"))
    monkeypatch.setattr(ec, "get_device", lambda: "cpu")

    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=tmp_path / "data.csv",
        test_corpora=[],
        post_eval_classifier="logistic_regression",
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    out = tmp_path / "out"
    ec.run_final_classification_eval(cfg, ckpt, out)
    pred_path = out / "predictions" / "predictions_btp.csv"
    assert pred_path.is_file()
    preds = pd.read_csv(pred_path)
    assert "pred_macro" in preds.columns
    assert "prob_A0" in preds.columns
    assert len(preds) == 12
    assert "corpus" in preds.columns
