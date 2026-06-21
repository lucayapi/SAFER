"""Tests stop_metier et résumé topics macro_transfer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro_transfer.bertopic_utils import (
    DEFAULT_STOP_WORDS_FILE,
    bertopic_show_progress,
    load_stop_metier,
    resolve_stop_words_file,
    umap_enabled,
)
from macro_transfer.topics_export import summarize_topics_by_macro

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_load_stop_metier_non_empty():
    words = load_stop_metier()
    assert len(words) >= 10
    assert "accident" in words
    assert "travail" in words


def test_resolve_stop_words_default():
    path = resolve_stop_words_file({})
    assert path == DEFAULT_STOP_WORDS_FILE
    assert path.is_file()


def test_umap_enabled_default_block():
    assert umap_enabled({"umap": {"enabled": True, "n_components": 5}}) is True
    assert umap_enabled({"umap": {"n_components": 5}}) is True


def test_umap_enabled_disabled():
    assert umap_enabled({"umap": None}) is False
    assert umap_enabled({"umap": False}) is False
    assert umap_enabled({"umap": {"enabled": False}}) is False


def test_umap_enabled_missing_key():
    assert umap_enabled({}) is True


def test_bertopic_show_progress_default_and_diagnostics():
    assert bertopic_show_progress({}) is True
    assert bertopic_show_progress({"diagnostics": {"show_progress": False}}) is False
    assert bertopic_show_progress({"show_progress": False}) is False


def test_summarize_topics_by_macro():
    themes = pd.DataFrame(
        {
            "macro": ["A0", "A0", "B"],
            "topic_id": [0, 1, 0],
            "n_units": [5, 3, 10],
        }
    )
    summary = summarize_topics_by_macro(themes)
    assert len(summary) == 4
    row_a0 = summary.loc[summary["macro"] == "A0"].iloc[0]
    assert row_a0["n_topics"] == 2
    assert row_a0["n_units"] == 8
