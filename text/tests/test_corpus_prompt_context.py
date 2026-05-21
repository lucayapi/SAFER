"""Tests contexte corpus pour prompts OpenAI."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.corpus_context import (
    format_corpus_context_for_prompt,
    get_corpus_prompt_context,
    load_corpus_prompt_contexts,
    resolve_prompt_context_key,
)


def test_load_corpus_contexts():
    registry = load_corpus_prompt_contexts(anchor=TEXT_ROOT)
    assert "metallurgie" in registry
    assert "btp" in registry
    assert registry["metallurgie"].description


def test_resolve_prompt_context_key_metallurgie():
    assert resolve_prompt_context_key("metallurgie", anchor=TEXT_ROOT) == "metallurgie"


def test_format_corpus_context_metallurgie():
    block = format_corpus_context_for_prompt("metallurgie", anchor=TEXT_ROOT)
    assert block is not None
    assert "Métallurgie" in block
    assert "métallurgie" in block.lower()


def test_get_corpus_context_unknown():
    assert get_corpus_prompt_context("nonexistent_xyz", anchor=TEXT_ROOT) is None
