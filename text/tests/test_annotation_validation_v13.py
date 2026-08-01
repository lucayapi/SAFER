"""Tests validation v13 (deux passes)."""

from __future__ import annotations

import pytest

from annotation.validation import validate_prediction, validate_prediction_v13


def _v13_payload(**overrides):
    base = {
        "label": "A0",
        "injury_mentioned": "NOT_MENTIONED",
        "hospitalized": "NOT_MENTIONED",
        "fatal": "NOT_MENTIONED",
        "confidence": 0.62,
        "ambiguous": True,
        "context_needed": True,
        "alternative_label": "B",
        "ambiguity_type": "ACTION_INTENT",
        "ambiguity_reason": "Le caractère volontaire ou accidentel n'est pas explicite.",
        "context_used": False,
        "justification": "Contexte utilisé: non. Indice principal: « est retiré ».",
    }
    base.update(overrides)
    return base


def test_validate_prediction_v13_pass1_ok():
    out = validate_prediction_v13(_v13_payload(), pass_mode="pass1")
    assert out["pred_ok"] is True
    assert out["pred_ambiguous"] is True
    assert out["pred_alternative_label"] == "B"


def test_validate_prediction_v13_pass1_rejects_context_used():
    with pytest.raises(ValueError, match="context_used doit être false"):
        validate_prediction_v13(_v13_payload(context_used=True), pass_mode="pass1")


def test_validate_prediction_v13_ambiguous_requires_alternative():
    with pytest.raises(ValueError, match="alternative_label requis"):
        validate_prediction_v13(
            _v13_payload(ambiguous=True, alternative_label="NONE"),
            pass_mode="pass1",
        )


def test_validate_prediction_v13_not_ambiguous_requires_none():
    with pytest.raises(ValueError, match="alternative_label doit être NONE"):
        validate_prediction_v13(
            _v13_payload(
                ambiguous=False,
                context_needed=False,
                alternative_label="B",
                ambiguity_type="NONE",
                ambiguity_reason="",
            ),
            pass_mode="pass1",
        )


def test_validate_prediction_v13_not_ambiguous_coerces_ambiguity_type():
    """ambiguous=false + ambiguity_type parasite → normalisé en NONE (pas de rejet)."""
    out = validate_prediction_v13(
        _v13_payload(
            ambiguous=False,
            context_needed=False,
            alternative_label="NONE",
            ambiguity_type="INSUFFICIENT_INFORMATION",
            ambiguity_reason="should be cleared",
        ),
        pass_mode="pass1",
    )
    assert out["pred_ok"] is True
    assert out["pred_label"] == "A0"
    assert out["pred_ambiguity_type"] == "NONE"
    assert out["pred_ambiguity_reason"] == ""


def test_validate_prediction_dispatch_v13():
    out = validate_prediction(
        _v13_payload(
            ambiguous=False,
            context_needed=False,
            alternative_label="NONE",
            ambiguity_type="NONE",
            ambiguity_reason="",
        ),
        prompt_version="v13_two_pass_ambiguity_context",
        pass_mode="pass1",
    )
    assert out["pred_ambiguous"] is False
    assert "pred_context_needed" in out
