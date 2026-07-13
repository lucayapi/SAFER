"""Tests validation annotation."""

from __future__ import annotations

import pytest

from annotation.validation import (
    extract_context_used,
    extract_json_object,
    normalize_mention_value,
    repair_justification,
    validate_prediction,
)


def test_extract_json_object_from_fenced_block():
    raw = '```json\n{"label": "B", "injury_mentioned": "NOT_MENTIONED", "hospitalized": "NOT_MENTIONED", "fatal": "NOT_MENTIONED", "confidence": 0.9, "justification": "Contexte utilisé: non. Indice principal: « chute »."}\n```'
    parsed = extract_json_object(raw)
    assert parsed["label"] == "B"


def test_validate_prediction_ok():
    parsed = {
        "label": "C",
        "injury_mentioned": "YES",
        "hospitalized": "NO",
        "fatal": "NOT_MENTIONED",
        "confidence": 0.85,
        "justification": "Contexte utilisé: oui. Indice principal: « fracture ».",
    }
    out = validate_prediction(parsed)
    assert out["pred_ok"] is True
    assert out["pred_label"] == "C"
    assert out["pred_context_used"] is True


def test_extract_context_used_from_justification():
    assert extract_context_used("Contexte utilisé: oui. Indice principal: « chute ».") is True
    assert extract_context_used("Contexte utilisé: non. Indice principal: « sol ».") is False


def test_validate_prediction_invalid_label():
    with pytest.raises(ValueError, match="Label invalide"):
        validate_prediction(
            {
                "label": "Z",
                "injury_mentioned": "YES",
                "hospitalized": "NO",
                "fatal": "NOT_MENTIONED",
                "confidence": 0.5,
                "justification": "Contexte utilisé: non. Indice principal: « x ».",
            }
        )


def test_normalize_mention_value_fixes_common_typo():
    assert normalize_mention_value("NOT_MOUNDED", field_name="fatal") == "NOT_MENTIONED"


def test_validate_prediction_repairs_missing_justification_fields():
    parsed = {
        "label": "B",
        "injury_mentioned": "NOT_MENTIONED",
        "hospitalized": "NOT_MENTIONED",
        "fatal": "NOT_MOUNDED",
        "confidence": 0.25,
        "justification": "Phrase technique sans format imposé.",
    }
    out = validate_prediction(parsed)
    assert out["pred_fatal"] == "NOT_MENTIONED"
    assert "Contexte utilisé: non" in out["pred_justification"]
    assert "Indice principal:" in out["pred_justification"]
    assert out["pred_context_used"] is False
