"""Tests export DataMapPlot découplé du fit BERTopic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_transfer.bertopic_exports import (
    bertopic_macro_model_path,
    export_bertopic_datamaps_from_run,
    load_bertopic_macro_model,
    save_bertopic_macro_model,
)


TEXT_ROOT = Path(__file__).resolve().parents[1]


class _PicklableVec:
    stop_words_ = ["le"]


class _PicklableFakeBertopic:
    """Stub picklable (classe module-level)."""

    def __init__(self) -> None:
        self.embedding_model = object()
        self.representation_model = object()
        self.vectorizer_model = _PicklableVec()
        self.topic_id = 1


def test_export_datamaps_skips_when_models_missing(tmp_path: Path):
    out = tmp_path / "run"
    topics = out / "topics_bertopic"
    topics.mkdir(parents=True)
    pd.DataFrame(
        {
            "doc_idx": [0, 1],
            "macro": ["A0", "A0"],
            "topic_id": [0, 1],
        }
    ).to_csv(topics / "assignments.csv", index=False)

    meta = pd.DataFrame({"sentence": ["a", "b"]})
    emb = np.eye(2, dtype=np.float64)

    saved = export_bertopic_datamaps_from_run(
        out,
        meta,
        emb,
        macros=["A0"],
        show_progress=False,
    )
    assert saved == {}
    assert not bertopic_macro_model_path(out / "bertopic" / "A0").is_file()


def test_export_datamaps_requires_assignments(tmp_path: Path):
    meta = pd.DataFrame({"sentence": ["a"]})
    emb = np.zeros((1, 4), dtype=np.float64)
    with pytest.raises(FileNotFoundError, match="assignments"):
        export_bertopic_datamaps_from_run(tmp_path, meta, emb, show_progress=False)


def test_save_load_bertopic_macro_model_joblib_fallback(tmp_path: Path):
    model = _PicklableFakeBertopic()
    macro_dir = tmp_path / "bertopic" / "A0"
    out = save_bertopic_macro_model(model, macro_dir)
    assert out is not None and out.is_file()
    loaded = load_bertopic_macro_model(macro_dir)
    assert getattr(loaded, "topic_id", None) == 1
    assert model.embedding_model is not None
    assert model.representation_model is not None
