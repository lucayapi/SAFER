"""Tests helpers température OpenAI (gpt-5*)."""

from __future__ import annotations

from macro_transfer.openai_utils import (
    apply_openai_chat_temperature,
    openai_chat_accepts_custom_temperature,
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
