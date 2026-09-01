import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recurrent_scenarios"))

from llm_labeling import extract_theme_item, normalize_llm_fields, parse_llm_payload


def test_extract_theme_without_topic_id():
    items = parse_llm_payload(
        '{"themes": [{"label": "Chute en hauteur", "description": "desc", "evidence": "preuve"}]}'
    )
    record = {"topic_id": "A0_006"}
    item = extract_theme_item(items, record)
    fields = normalize_llm_fields(item)
    assert fields["llm_label"] == "Chute en hauteur"
    assert fields["llm_description"] == "desc"
    assert fields["llm_evidence"] == "preuve"


def test_build_theme_label_chat_kwargs_gpt56_luna():
    from llm_labeling import build_theme_label_chat_kwargs

    kwargs = build_theme_label_chat_kwargs(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "test"}],
        reasoning_effort="low",
        max_output_tokens=4000,
    )
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] == 4000
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_extract_theme_with_topic_id():
    items = parse_llm_payload(
        '{"themes": [{"topic_id": "A0_006", "label": "Chute", "description": "d", "evidence": "e"}]}'
    )
    item = extract_theme_item(items, {"topic_id": "A0_006"})
    assert normalize_llm_fields(item)["llm_label"] == "Chute"
