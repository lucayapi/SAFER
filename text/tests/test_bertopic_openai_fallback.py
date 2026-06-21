"""Tests repli OpenAI BERTopic / judge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from macro_transfer.bertopic_utils import fit_bertopic_subset, representation_fallback_on_error
from macro_transfer.openai_utils import is_openai_capacity_error


class _FakeRateLimitError(Exception):
    pass


def test_is_openai_capacity_error_message():
    assert is_openai_capacity_error(Exception("insufficient_quota"))
    assert is_openai_capacity_error(Exception("HTTP 429 Too Many Requests"))
    assert not is_openai_capacity_error(ValueError("bad json"))


def test_representation_fallback_on_error_default():
    cfg = {"representation": {"enabled": True}}
    assert representation_fallback_on_error(cfg) is True
    cfg["representation"]["fallback_on_error"] = False
    assert representation_fallback_on_error(cfg) is False


@patch("macro_transfer.bertopic_utils.build_bertopic_for_macro")
def test_fit_bertopic_subset_fallback_without_openai(mock_build):
    model = MagicMock()
    model.fit_transform.side_effect = [
        _FakeRateLimitError("insufficient_quota"),
        (np.array([0, 0]), np.array([[0.9, 0.1], [0.8, 0.2]])),
    ]
    mock_build.return_value = model

    bertopic_cfg = {
        "representation": {"enabled": True, "fallback_on_error": True},
        "min_topic_size": 2,
    }
    texts = ["a", "b"]
    emb = np.eye(2, dtype=np.float64)

    topic_ids, conf, out_model = fit_bertopic_subset(
        texts,
        emb,
        bertopic_cfg,
        macro="C",
        show_progress=False,
    )

    assert len(topic_ids) == 2
    assert mock_build.call_count == 2
    assert bertopic_cfg.get("_disable_representation") is not True
    second_cfg = mock_build.call_args_list[1].args[1]
    assert second_cfg.get("_disable_representation") is True
    assert out_model is model
