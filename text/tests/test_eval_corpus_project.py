"""Tests project_embedding_corpus (end2end text)."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from scgm_text.eval_corpus import project_embedding_corpus


def test_project_embedding_corpus_end2end():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.hiddim = 16

    def _forward(batch):
        n = batch["input_ids"].shape[0]
        return torch.randn(n, 16)

    mock_model.side_effect = _forward

    with patch("scgm_text.eval_corpus.load_scgm_checkpoint", return_value=(mock_model, {"backbone_name": "test/model", "hiddim": 16}, {})):
        with patch("scgm_text.eval_corpus.TextRawDataset") as mock_ds:
            mock_ds.return_value.get_metadata_df.return_value = __import__("pandas").DataFrame(
                {"pred_label": ["A0", "B"]}
            )
            with patch("transformers.AutoTokenizer"):
                with patch("scgm_text.eval_corpus.make_text_collate_fn") as mock_collate:
                    mock_collate.return_value = lambda items: {
                        "input_ids": torch.ones(len(items), 4, dtype=torch.long),
                        "attention_mask": torch.ones(len(items), 4, dtype=torch.long),
                        "label_ids": torch.zeros(len(items), dtype=torch.long),
                        "indices": torch.arange(len(items)),
                    }
                    mock_ds.return_value.__len__ = lambda self: 2
                    mock_ds.return_value.__getitem__ = lambda self, i: {
                        "text": "x",
                        "label": 0,
                        "group": "g",
                        "row_id": i,
                        "index": i,
                    }
                    proj, labels = project_embedding_corpus(
                        "ckpt.pt",
                        "data.csv",
                        batch_size=2,
                        device="cpu",
                    )
    assert proj.shape[0] == 2
    assert len(labels) == 2
