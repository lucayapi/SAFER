"""Tests BERTopic representation OpenAI (sans appel API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from macro_transfer.representation import (
    DEFAULT_FR_CHAT_PROMPT,
    build_representation_model,
    build_tiktoken_tokenizer,
    representation_enabled,
    _resolve_prompt,
)
from macro_transfer.bertopic_utils import format_topic_words, topic_label_from_model

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_representation_enabled_default():
    cfg = {"representation": {"enabled": True}}
    assert representation_enabled(cfg) is True


def test_representation_enabled_explicit_false():
    cfg = {"representation": {"enabled": False}}
    assert representation_enabled(cfg) is False


def test_resolve_prompt_injects_macro():
    prompt = _resolve_prompt({}, macro="A0", corpus_id=None, anchor=TEXT_ROOT)
    assert "[DOCUMENTS]" in prompt
    assert "[KEYWORDS]" in prompt
    assert "A0" in prompt
    assert "[MACRO_CONTEXT]" not in prompt
    assert "[CORPUS_CONTEXT]" not in prompt


def test_resolve_prompt_injects_corpus_and_macro():
    prompt = _resolve_prompt(
        {"include_corpus_context": True},
        macro="A0",
        corpus_id="metallurgie",
        anchor=TEXT_ROOT,
    )
    assert "Métallurgie" in prompt
    assert "Macro : A0" in prompt
    assert "non exhaustifs" in prompt
    assert "illustratifs" in prompt
    assert "[CORPUS_CONTEXT]" not in prompt
    assert "[MACRO_CONTEXT]" not in prompt


def test_build_representation_model_disabled():
    assert build_representation_model({"enabled": False}, macro="B") is None


def test_build_tiktoken_tokenizer_roundtrip():
    pytest.importorskip("tiktoken")
    tok = build_tiktoken_tokenizer("gpt-4o-mini")
    assert hasattr(tok, "encode") and hasattr(tok, "decode")
    tokens = tok.encode("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) >= 1
    assert "hello" in tok.decode(tokens)


def test_tiktoken_tokenizer_bertopic_truncate():
    """Aligné sur bertopic.representation._utils.truncate_document (0.16)."""
    pytest.importorskip("tiktoken")
    pytest.importorskip("bertopic")
    from bertopic.representation._utils import truncate_document

    tok = build_tiktoken_tokenizer("gpt-4o-mini")
    doc = "mot " * 500
    truncated = truncate_document(None, 20, tok, doc)
    assert len(tok.encode(truncated)) <= 20


@patch("macro_transfer.representation._get_client")
@patch("macro_transfer.representation.load_openai_dotenv")
def test_build_representation_model_mock(_mock_env, mock_client):
    mock_client.return_value = MagicMock()
    rep = build_representation_model(
        {
            "enabled": True,
            "model": "gpt-4o-mini",
            "chat": True,
            "nr_docs": 4,
            "doc_length": 100,
            "tokenizer_model": "gpt-4o-mini",
        },
        macro="C",
        anchor=TEXT_ROOT,
    )
    assert rep is not None
    assert rep.model == "gpt-4o-mini"
    assert "[DOCUMENTS]" in rep.prompt
    assert hasattr(rep.tokenizer, "encode") and hasattr(rep.tokenizer, "decode")


def test_topic_label_from_model_custom_name():
    model = MagicMock()
    model.get_topic_info.return_value = pd.DataFrame(
        {"Topic": [0], "CustomName": ["Chute de hauteur"]}
    )
    model.topic_representations_ = {0: [("Chute de hauteur", 1.0)]}
    model.representation_model = MagicMock()
    assert topic_label_from_model(model, 0) == "Chute de hauteur"
    assert topic_label_from_model(model, -1) == ""


def test_topic_label_ignores_bertopic_keyword_fallback():
    model = MagicMock()
    model.custom_labels_ = None
    model.get_topic_info.return_value = pd.DataFrame(
        {"Topic": [0], "Name": ["0_chute_hauteur_travail"]}
    )
    model.topic_representations_ = {0: [("chute", 0.5), ("hauteur", 0.4)]}
    model.representation_model = None
    assert topic_label_from_model(model, 0) == ""


def test_topic_label_from_llm_representation():
    model = MagicMock()
    model.get_topic_info.return_value = pd.DataFrame({"Topic": [1]})
    model.topic_representations_ = {1: [("Coincement de main", 1.0)]}
    model.representation_model = MagicMock()
    assert topic_label_from_model(model, 1) == "Coincement de main"


def test_format_topic_words_prefers_ctfidf_over_llm_label():
    pytest.importorskip("scipy")
    from scipy.sparse import csr_matrix

    model = MagicMock()
    model.topic_sizes_ = {0: 10}
    model.c_tf_idf_ = csr_matrix([[0.1, 0.5, 0.3]])
    model.vectorizer_model.get_feature_names_out.return_value = np.array(
        ["mot1", "chute", "hauteur"]
    )
    model.topic_representations_ = {0: [("Libellé LLM", 1.0)]}
    model.representation_model = MagicMock()
    assert format_topic_words(model, 0, top_k=3) == "chute hauteur mot1"


def test_default_fr_prompt_has_tags():
    assert "[DOCUMENTS]" in DEFAULT_FR_CHAT_PROMPT
    assert "[KEYWORDS]" in DEFAULT_FR_CHAT_PROMPT
    assert "[CORPUS_CONTEXT]" in DEFAULT_FR_CHAT_PROMPT
    assert "[MACRO_CONTEXT]" in DEFAULT_FR_CHAT_PROMPT
