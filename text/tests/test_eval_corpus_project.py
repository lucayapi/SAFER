"""Tests project_embedding_corpus (input_mode text vs precomputed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))


def test_project_embedding_corpus_precomputed_mode():
    from scgm_text.eval_corpus import project_embedding_corpus

    fake_features = torch.ones(2, 4)
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = fake_features

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 2
    mock_ds.__getitem__.side_effect = lambda i: (torch.zeros(8), torch.tensor(i), i)
    mock_ds.get_metadata_df.return_value = __import__("pandas").DataFrame({"pred_label": ["A0", "A1"]})

    with patch("scgm_text.eval_corpus.load_scgm_checkpoint", return_value=(mock_model, {"input_mode": "precomputed_embeddings"}, {})):
        with patch("scgm_text.eval_corpus.TextEmbeddingDataset", return_value=mock_ds):
            with patch("scgm_text.eval_corpus.forward_features", return_value=fake_features):
                proj, labels = project_embedding_corpus("ckpt.pt", "data.csv", "emb.csv", device="cpu")

    assert proj.shape == (2, 4)
    assert len(labels) == 2
    mock_model.assert_not_called()
    mock_model.eval.assert_called_once()


def test_project_embedding_corpus_text_mode_uses_dict_batch():
    from scgm_text.eval_corpus import project_embedding_corpus

    captured_batches = []
    fake_features = torch.ones(1, 4)

    def _capture_forward(model, batch):
        captured_batches.append(batch)
        return fake_features

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.side_effect = _capture_forward

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 1
    mock_ds.__getitem__.return_value = ("text", 0, 0)
    mock_ds.get_metadata_df.return_value = __import__("pandas").DataFrame({"pred_label": ["A0"]})

    mock_batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4), "label_ids": torch.zeros(1, dtype=torch.long), "indices": torch.zeros(1, dtype=torch.long)}

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

    with patch("scgm_text.eval_corpus.load_scgm_checkpoint", return_value=(mock_model, {"input_mode": "text", "backbone_model_name_or_path": "test/model"}, {})):
        with patch("scgm_text.eval_corpus.TextRawDataset", return_value=mock_ds):
            with patch("scgm_text.eval_corpus.make_text_collate_fn", return_value=lambda items: mock_batch):
                with patch.dict("sys.modules", {"transformers": mock_transformers}):
                    with patch("scgm_text.eval_corpus.forward_features", side_effect=_capture_forward):
                        proj, _ = project_embedding_corpus("ckpt.pt", "data.csv", "emb.csv", device="cpu")

    assert proj.shape == (1, 4)
    assert len(captured_batches) >= 1
    assert isinstance(captured_batches[0], dict)
