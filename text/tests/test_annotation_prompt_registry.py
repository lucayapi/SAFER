"""Tests registre de prompts."""

from __future__ import annotations

from pathlib import Path

from annotation.batch_request import build_chat_completion_body
from annotation.config import AnnotationConfig
from annotation.prompts import (
    V13_PROMPT_VERSION,
    build_artifact_slug,
    is_v13_prompt,
    resolve_prompt_bundle,
)


def test_is_v13_prompt():
    assert is_v13_prompt("v13_two_pass_ambiguity_context") is True
    assert is_v13_prompt("v10_macro_labels_independent_outcomes") is False


def test_build_artifact_slug():
    assert build_artifact_slug("v13_two_pass_ambiguity_context", "pass1") == (
        "v13_two_pass_ambiguity_context__pass1"
    )


def test_resolve_prompt_bundle_v13_pass1():
    bundle = resolve_prompt_bundle(V13_PROMPT_VERSION, pass_mode="pass1")
    assert bundle["pass_mode"] == "pass1"
    assert "DÉTECTION DE L'AMBIGUÏTÉ" in bundle["system_prompt"]


def test_build_chat_completion_body_pass1_without_narrative():
    from pathlib import Path

    cfg = AnnotationConfig(
        input_csv=Path("x.csv"),
        prompt_version=V13_PROMPT_VERSION,
        pass_mode="pass1",
    )
    row = {
        "accident_id": "A1",
        "fact_id": "1",
        "sentence": "Il chute.",
        "accident_summary": "Récit complet avec glissade.",
    }
    body = build_chat_completion_body(row, cfg)
    user = body["messages"][1]["content"]
    assert "<NARRATIVE>" not in user
    assert "PASSE 1" in user
    assert "Il chute." in user
