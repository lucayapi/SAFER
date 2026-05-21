"""Tests BERTopic representation OpenAI (sans appel API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from macro_transfer.representation import (
    DEFAULT_FR_CHAT_PROMPT,
    build_representation_model,
    build_tiktoken_tokenizer,
    representation_enabled,
    _resolve_prompt,
)
from macro_transfer.bertopic_utils import topic_label_from_model

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_representation_enabled_default():
    cfg = {"representation": {"enabled": True}}
    assert representation_enabled(cfg) is True


def test_representation_enabled_explicit_false():
    cfg = {"representation": {"enabled": False}}
    assert representation_enabled(cfg) is False


def test_resolve_prompt_injects_macro():
    prompt = _resolve_prompt({}, macro="A0", anchor=TEXT_ROOT)
    assert "[DOCUMENTS]" in prompt
    assert "[KEYWORDS]" in prompt
    assert "A0" in prompt
    assert "[MACRO_CONTEXT]" not in prompt


def test_build_representation_model_disabled():
    assert build_representation_model({"enabled": False}, macro="B") is None


def test_build_tiktoken_tokenizer_roundtrip():
    tok = build_tiktoken_tokenizer("gpt-4o-mini")
    tokens = tok("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) >= 1


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


def test_topic_label_from_model_topic_labels_dict():
    model = MagicMock()
    model.topic_labels_ = {0: "Chute de hauteur", 1: "Manutention"}
    assert topic_label_from_model(model, 0) == "Chute de hauteur"
    assert topic_label_from_model(model, -1) == ""


def test_default_fr_prompt_has_tags():
    assert "[DOCUMENTS]" in DEFAULT_FR_CHAT_PROMPT
    assert "[KEYWORDS]" in DEFAULT_FR_CHAT_PROMPT
