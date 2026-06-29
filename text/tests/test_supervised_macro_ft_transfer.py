"""Tests transfert supervised_macro_ft (encode + predict)."""

from __future__ import annotations

import numpy as np
import torch

from supervised_macro_ft.model import SupervisedMacroModel
from supervised_macro_ft.paths import (
    resolve_supervised_macro_output_dir,
    supervised_macro_output_dir,
)
from supervised_macro_ft.transfer import encode_texts, predict_corpus


class _DummyTokenizer:
    pad_token = ""

    def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        n = len(texts)
        return {
            "input_ids": torch.randint(0, 50, (n, 8)),
            "attention_mask": torch.ones(n, 8, dtype=torch.long),
        }


def test_encode_and_predict_shapes():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=32,
    )
    device = torch.device("cpu")
    tok = _DummyTokenizer()
    texts = ["a", "b", "c"]
    h = encode_texts(model, tok, texts, max_length=32, batch_size=2, device=device)
    assert h.shape == (3, 32)
    pred, probs, conf, margin, entropy = predict_corpus(
        model, tok, texts, macros=["A0", "A1", "B", "C"], max_length=32, batch_size=2, device=device
    )
    assert len(pred) == 3
    assert probs.shape == (3, 4)
    assert np.allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)


def test_resolve_supervised_macro_output_dir_legacy_fallback(tmp_path):
    anchor = tmp_path
    corpus = "metallurgie"
    method = "supervised_macro_geo_ft"
    legacy = anchor / "output_test" / corpus / method / "transfer"
    legacy.mkdir(parents=True)
    (legacy / "target_macro_predictions.csv").write_text("x\n", encoding="utf-8")

    resolved = resolve_supervised_macro_output_dir(method, corpus, anchor=anchor)
    assert resolved == (anchor / "output_test" / corpus / method).resolve()

    canonical = supervised_macro_output_dir(method, corpus, anchor=anchor)
    assert resolved != canonical
