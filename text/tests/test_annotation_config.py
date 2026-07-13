"""Tests validation reasoning_effort dans AnnotationConfig."""

from __future__ import annotations

import pytest

from annotation.config import AnnotationConfig


def test_annotation_config_defaults_gpt54_mini_medium():
    cfg = AnnotationConfig(input_csv="x.csv")
    assert cfg.openai_model == "gpt-5.4-mini"
    assert cfg.reasoning_effort == "medium"
    assert cfg.max_output_tokens == 4000


def test_annotation_config_rejects_invalid_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning_effort invalide"):
        AnnotationConfig(input_csv="x.csv", reasoning_effort="turbo")
