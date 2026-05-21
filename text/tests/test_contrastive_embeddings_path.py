"""Tests chemins embeddings contrastifs BTP vs test."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.test_corpus import (
    method_btp_results_dir,
    method_test_results_dir,
    resolve_contrastive_embeddings_csv,
)


def test_resolve_test_embeddings_path():
    p = resolve_contrastive_embeddings_csv(
        "supcon", "test", corpus_id="metallurgie", anchor=TEXT_ROOT
    )
    assert "output_test" in str(p).replace("\\", "/")
    assert "metallurgie" in str(p)
    assert "supcon" in str(p)
    assert p.name == "final_embeddings_test.csv"
    assert p.parent.name == "embeddings"


def test_resolve_btp_embeddings_path():
    p = resolve_contrastive_embeddings_csv("softtriple", "btp", anchor=TEXT_ROOT)
    assert "output" in str(p).replace("\\", "/")
    assert "softtriple" in str(p)
    assert "output_test" not in str(p).replace("\\", "/")
    assert p.suffix == ".csv"


def test_btp_and_test_dirs_differ():
    btp = method_btp_results_dir("supcon", anchor=TEXT_ROOT)
    test = method_test_results_dir("supcon", "metallurgie", anchor=TEXT_ROOT)
    assert btp != test
