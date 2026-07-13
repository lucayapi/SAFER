"""Tests helpers température OpenAI (gpt-5*)."""

from __future__ import annotations

from macro_transfer.openai_utils import (
    REASONING_MIN_COMPLETION_TOKENS,
    apply_openai_chat_max_output_tokens,
    apply_openai_chat_reasoning_effort,
    apply_openai_chat_temperature,
    extract_chat_message_content,
    openai_chat_accepts_custom_temperature,
    openai_chat_is_reasoning_model,
    openai_chat_uses_max_completion_tokens,
)


def test_gpt5_rejects_custom_temperature():
    assert not openai_chat_accepts_custom_temperature("gpt-5-mini")
    assert not openai_chat_accepts_custom_temperature("GPT-5")


def test_gpt4_accepts_custom_temperature():
    assert openai_chat_accepts_custom_temperature("gpt-4o-mini")


def test_apply_openai_chat_temperature_omits_for_gpt5():
    kwargs = {"model": "gpt-5-mini"}
    apply_openai_chat_temperature(kwargs, model="gpt-5-mini", temperature=0.2)
    assert "temperature" not in kwargs


def test_apply_openai_chat_temperature_sets_for_gpt4():
    kwargs = {"model": "gpt-4o-mini"}
    apply_openai_chat_temperature(kwargs, model="gpt-4o-mini", temperature=0.2)
    assert kwargs["temperature"] == 0.2


def test_gpt5_uses_max_completion_tokens():
    assert openai_chat_uses_max_completion_tokens("gpt-5-nano")


def test_gpt4_uses_max_tokens():
    assert not openai_chat_uses_max_completion_tokens("gpt-4o-mini")


def test_apply_openai_chat_max_output_tokens_gpt5_enforces_minimum():
    kwargs = {"model": "gpt-5-nano"}
    apply_openai_chat_max_output_tokens(kwargs, model="gpt-5-nano", max_output_tokens=400)
    assert kwargs["max_completion_tokens"] == REASONING_MIN_COMPLETION_TOKENS
    assert "max_tokens" not in kwargs


def test_apply_openai_chat_max_output_tokens_gpt4():
    kwargs = {"model": "gpt-4o-mini"}
    apply_openai_chat_max_output_tokens(kwargs, model="gpt-4o-mini", max_output_tokens=400)
    assert kwargs["max_tokens"] == 400
    assert "max_completion_tokens" not in kwargs


def test_apply_openai_chat_reasoning_effort_gpt5_default():
    kwargs = {"model": "gpt-5-nano"}
    apply_openai_chat_reasoning_effort(kwargs, model="gpt-5-nano")
    assert kwargs["reasoning_effort"] == "minimal"


def test_apply_openai_chat_reasoning_effort_gpt5_medium():
    kwargs = {"model": "gpt-5-mini"}
    apply_openai_chat_reasoning_effort(kwargs, model="gpt-5-mini", reasoning_effort="medium")
    assert kwargs["reasoning_effort"] == "medium"


def test_apply_openai_chat_reasoning_effort_ignored_for_gpt4():
    kwargs = {"model": "gpt-4o-mini"}
    apply_openai_chat_reasoning_effort(kwargs, model="gpt-4o-mini")
    assert "reasoning_effort" not in kwargs


def test_extract_chat_message_content_reads_message():
    from types import SimpleNamespace

    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=' {"label": "B"} ', refusal=None),
                finish_reason="stop",
            )
        ]
    )
    assert extract_chat_message_content(resp) == '{"label": "B"}'
