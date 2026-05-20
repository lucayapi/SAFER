"""Tests prompts / parsing openai_theme_labels (sans appel API)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scgm_text.openai_theme_labels import (
    _fallback_row_labels,
    _labels_from_api_response,
    _split_example_sentences,
    build_user_prompt,
)


def test_split_example_sentences():
    raw = "phrase un || phrase deux || phrase trois"
    parts = _split_example_sentences(raw, 2)
    assert parts == ["phrase un", "phrase deux"]


def test_build_user_prompt_documents_only():
    prompt = build_user_prompt("doc A || doc B", n_example_texts=2)
    assert "Voici un topic qui contient les extraits suivants" in prompt
    assert "libelle" in prompt
    assert "doc A" in prompt
    assert "doc B" in prompt
    assert "Macro :" not in prompt
    assert "dominant_macro" not in prompt
    assert "z_id" not in prompt
    assert "TF-IDF" not in prompt


def test_build_user_prompt_oriented_a1():
    prompt = build_user_prompt("doc A", n_example_texts=1, macro="A1")
    assert "Macro : A1" in prompt
    assert "Facteurs contributifs" in prompt
    assert "refléter uniquement le thème de la macro" in prompt
    assert "événement dynamique (B)" in prompt or "événement (B)" in prompt
    assert "Contexte de travail" not in prompt


def test_one_row_passes_macro_to_prompt():
    from unittest.mock import MagicMock

    from scgm_text.openai_theme_labels import _one_row

    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs.get("messages")
        msg = MagicMock()
        msg.content = '{"libelle": "test libellé court"}'
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client = MagicMock()
    client.chat.completions.create = fake_create
    row = pd.Series(
        {
            "z_id": 0,
            "dominant_macro": "B",
            "top_words": "chute",
            "top_sentences": "La victime est tombée || perte d'équilibre",
        }
    )
    _one_row(
        client,
        "gpt-4o-mini",
        0.3,
        row,
        n_example_texts=2,
        summary_words_min=3,
        summary_words_max=12,
    )
    user_msg = captured["messages"][1]["content"]
    assert "Macro : B" in user_msg
    assert "Déviation" in user_msg or "événement accidentel" in user_msg


def test_labels_from_api_response_libelle_key():
    row = pd.Series({"z_id": 0, "top_words": "chute;échafaudage;sécurité"})
    out = _labels_from_api_response(
        {"libelle": "chute échafaudage sécurité chantier"},
        row,
        summary_words_min=3,
        summary_words_max=12,
    )
    assert out["theme_summary"] == "chute échafaudage sécurité chantier"


def test_labels_from_api_response_label_key_legacy():
    row = pd.Series({"z_id": 0, "top_words": "chute;échafaudage;sécurité"})
    out = _labels_from_api_response(
        {"label": "chute échafaudage sécurité chantier"},
        row,
        summary_words_min=3,
        summary_words_max=12,
    )
    assert out["theme_summary"] == "chute échafaudage sécurité chantier"
    assert out["theme_title"] == out["theme_summary"][:60]
    assert "chute" in out["theme_keywords"]


def test_labels_from_api_response_legacy_theme_summary():
    row = pd.Series({"z_id": 1, "top_words": "foo bar"})
    out = _labels_from_api_response(
        {"theme_summary": "incident manutention équipement lourd"},
        row,
        summary_words_min=3,
        summary_words_max=12,
    )
    assert "incident" in out["theme_summary"]


def test_fallback_row_labels():
    row = pd.Series(
        {
            "z_id": 2,
            "top_words": "gants;protection;coupe",
            "dominant_macro": "A0",
        }
    )
    out = _fallback_row_labels(row, summary_words_min=3, summary_words_max=12)
    assert out["theme_summary"]
    assert "gants" in out["theme_keywords"]
