"""Tests eval_corpus (mock géométrie)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.eval_geometry import evaluate_embeddings_geometry


def test_evaluate_embeddings_geometry_columns():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((40, 8)).astype(np.float32)
    labels = np.array(["A"] * 20 + ["B"] * 20)
    row = evaluate_embeddings_geometry(emb, labels, method="mock")
    for col in ("eta2_macro_balanced_perc", "eta2_macro_balanced", "eta2_weighted"):
        assert col in row


@patch("contrastive_methods.eval_corpus.encode_contrastive_texts")
@patch("contrastive_methods.eval_corpus.prepare_text_dataset")
def test_evaluate_contrastive_on_csv_mock(mock_dataset, mock_encode, tmp_path):
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.eval_corpus import evaluate_contrastive_on_csv

    meta = pd.DataFrame(
        {
            "doc_id": list(range(10)),
            "sentence": ["x"] * 10,
            "pred_label": ["A0", "A1", "B", "C"] * 2 + ["A0", "A1"],
        }
    )
    ds = MagicMock()
    ds.metadata_df = meta
    ds.text_col = "sentence"
    mock_dataset.return_value = ds

    mock_encode.return_value = np.random.randn(10, 8).astype(np.float32)

    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=tmp_path / "data.csv",
    )
    (tmp_path / "data.csv").write_text("doc_id,sentence,pred_label\n0,x,A\n", encoding="utf-8")

    row = evaluate_contrastive_on_csv(
        cfg,
        tmp_path / "ckpt",
        tmp_path / "data.csv",
        corpus="btp",
    )
    assert "eta2_macro_balanced_perc" in row
