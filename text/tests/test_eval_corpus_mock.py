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
        }
    )
    ds = MagicMock()
    ds.metadata_df = meta
    mock_dataset.return_value = ds
    mock_encode.return_value = np.random.randn(8, 4)
    mock_fit.return_value = MagicMock()
    mock_eval_cls.return_value = {"balanced_accuracy": 0.5, "macro_f1": 0.4, "accuracy": 0.45}
    mock_save.return_value = {"btp": tmp_path / "btp.csv"}

    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=tmp_path / "data.csv",
        test_corpora=[],
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    run_final_classification_eval(cfg, ckpt, tmp_path / "out")
    assert mock_save.called
