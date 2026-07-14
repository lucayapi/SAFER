"""Tests cache embeddings backbone supervised_macro_ft."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scgm_text.utils_io import create_doc_id_if_missing
from supervised_macro_ft.embedding_cache import (
    BackboneHiddenDataset,
    collate_hidden_batch,
    encode_projected_matrix,
    load_backbone_hidden_for_corpus,
    predict_from_hidden_matrix,
    should_cache_backbone_embeddings,
)
from supervised_macro_ft.model import SupervisedMacroModel


def _make_meta_and_emb_csv(tmp_path, n: int = 3, dim: int = 4):
    meta_df = pd.DataFrame(
        {
            "accident_id": [f"g{i}" for i in range(n)],
            "sentence": [f"text {i}" for i in range(n)],
            "pred_label": ["A0", "B", "C"][:n],
        }
    )
    meta_df = create_doc_id_if_missing(meta_df)
    emb_rows = {"doc_id": meta_df["doc_id"].tolist()}
    rng = np.random.default_rng(0)
    for d in range(dim):
        emb_rows[f"dim_{d:04d}"] = rng.random(n).tolist()
    emb_csv = tmp_path / "emb.csv"
    pd.DataFrame(emb_rows).to_csv(emb_csv, index=False)
    return meta_df, emb_csv


def test_should_cache_backbone_embeddings_only_when_frozen():
    assert should_cache_backbone_embeddings({"backbone_trainable": False, "cache_backbone_embeddings": True})
    assert not should_cache_backbone_embeddings({"backbone_trainable": True})
    assert not should_cache_backbone_embeddings(
        {"backbone_trainable": False, "cache_backbone_embeddings": False}
    )


def test_forward_on_hidden_batch_skips_backbone():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=16,
    )
    h = torch.randn(5, model.backbone.hidden_size)
    batch = {"hidden": h, "label_ids": torch.tensor([0, 1, 2, 3, 0])}
    logits = model(batch)
    assert logits.shape == (5, 4)
    loss = torch.nn.functional.cross_entropy(logits, batch["label_ids"])
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())


def test_backbone_hidden_dataset_and_fast_predict():
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=8,
    )
    hidden = np.random.randn(12, model.backbone.hidden_size).astype(np.float32)
    labels = np.arange(12, dtype=np.int64) % 4
    ds = BackboneHiddenDataset(hidden, labels, indices=np.array([0, 2, 4]))
    batch = collate_hidden_batch([ds[i] for i in range(3)])
    logits = model(batch)
    assert logits.shape == (3, 4)

    device = torch.device("cpu")
    z = encode_projected_matrix(model, hidden, batch_size=4, device=device)
    assert z.shape == (12, 8)
    pred, probs, _, _, _ = predict_from_hidden_matrix(
        model, hidden, macros=["A0", "A1", "B", "C"], batch_size=4, device=device
    )
    assert len(pred) == 12 and probs.shape == (12, 4)


def test_load_backbone_hidden_for_corpus_from_csv(tmp_path):
    meta_df, emb_csv = _make_meta_and_emb_csv(tmp_path, n=3, dim=4)
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=8,
    )
    texts = meta_df["sentence"].astype(str).tolist()
    device = torch.device("cpu")

    with patch("supervised_macro_ft.embedding_cache.encode_backbone_matrix") as mock_encode:
        hidden = load_backbone_hidden_for_corpus(
            meta_df=meta_df,
            texts=texts,
            emb_csv=emb_csv,
            cache_path=tmp_path / "cache.npy",
            model=model,
            tokenizer=None,
            max_length=32,
            batch_size=2,
            device=device,
        )
        mock_encode.assert_not_called()

    assert hidden.shape == (3, 4)
    assert hidden.dtype == np.float32


def test_load_backbone_hidden_for_corpus_fallback_encodes_and_saves_cache(tmp_path):
    meta_df, _ = _make_meta_and_emb_csv(tmp_path, n=2, dim=3)
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=8,
    )
    texts = meta_df["sentence"].astype(str).tolist()
    device = torch.device("cpu")
    cache_path = tmp_path / "backbone_hidden_ood.npy"
    fake_hidden = np.arange(6, dtype=np.float32).reshape(2, 3)

    with patch(
        "supervised_macro_ft.embedding_cache.encode_backbone_matrix",
        return_value=fake_hidden,
    ) as mock_encode:
        hidden = load_backbone_hidden_for_corpus(
            meta_df=meta_df,
            texts=texts,
            emb_csv=tmp_path / "missing.csv",
            cache_path=cache_path,
            model=model,
            tokenizer=None,
            max_length=32,
            batch_size=2,
            device=device,
        )
        mock_encode.assert_called_once()

    assert np.allclose(hidden, fake_hidden)
    assert cache_path.is_file()
    reloaded = np.load(cache_path)
    assert reloaded.shape == (2, 3)


def test_load_backbone_hidden_for_corpus_reuses_npy_cache(tmp_path):
    meta_df, _ = _make_meta_and_emb_csv(tmp_path, n=2, dim=3)
    cache_path = tmp_path / "backbone_hidden_ood.npy"
    cached = np.ones((2, 3), dtype=np.float32)
    np.save(cache_path, cached)
    model = SupervisedMacroModel(
        backbone_name="__test_dummy__",
        num_classes=4,
        backbone_trainable=False,
        projection="linear",
        hiddim=8,
    )

    with patch("supervised_macro_ft.embedding_cache.encode_backbone_matrix") as mock_encode:
        hidden = load_backbone_hidden_for_corpus(
            meta_df=meta_df,
            texts=["a", "b"],
            emb_csv=None,
            cache_path=cache_path,
            model=model,
            tokenizer=None,
            max_length=32,
            batch_size=2,
            device=torch.device("cpu"),
        )
        mock_encode.assert_not_called()

    assert np.allclose(hidden, cached)
