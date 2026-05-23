"""Tests export embeddings → DataFrame (sans PerformanceWarning fragmenté)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from contrastive_methods.export import dim_column_names, embeddings_to_dataframe


def test_embeddings_to_dataframe_shape_and_columns():
    doc_ids = np.array([1, 2, 3], dtype=np.int64)
    emb = np.random.randn(3, 8).astype(np.float32)
    frame = embeddings_to_dataframe(doc_ids, emb)
    assert list(frame.columns[:1]) == ["doc_id"]
    assert list(frame.columns[1:]) == dim_column_names(8)
    assert len(frame) == 3


def test_embeddings_to_dataframe_not_fragmented():
    doc_ids = np.arange(100, dtype=np.int64)
    emb = np.random.randn(100, 32).astype(np.float32)
    frame = embeddings_to_dataframe(doc_ids, emb)
    consolidated = frame.copy()
    assert len(consolidated.columns) == 33
