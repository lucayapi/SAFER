"""Tests évaluation LLM-judge des topics BERTopic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from macro_transfer.topic_judge import (
    JUDGE_CRITERIA,
    aggregate_judge_by_macro,
    build_topic_judge_prompt,
    compute_score_global,
    parse_judge_response,
    run_topic_judge_evaluation,
    sample_topic_examples,
)


def _valid_judge_payload(**overrides) -> dict:
    base = {
        "coherence_interne": 4,
        "homogeneite_accidentologique": 3,
        "alignement_role": 4,
        "specificite": 4,
        "nommabilite": 3,
        "utilite_reconstruction_scenario": 4,
        "label_propose": "Chute depuis échafaudage",
        "verdict": "conserver",
        "justification_courte": "Motif homogène.",
        "probleme_principal": "aucun",
        "score_global": 99.0,
    }
    base.update(overrides)
    return base


def test_compute_score_global_mean():
    scores = {c: 4 for c in JUDGE_CRITERIA}
    scores["homogeneite_accidentologique"] = 3
    assert compute_score_global(scores) == pytest.approx(3.8333333, rel=1e-4)


def test_parse_judge_response_ignores_llm_score_global():
    parsed = parse_judge_response(_valid_judge_payload())
    assert parsed["score_global"] == pytest.approx(3.6667, rel=1e-3)
    assert parsed["verdict"] == "conserver"


def test_parse_judge_response_missing_criterion_raises():
    payload = _valid_judge_payload()
    del payload["specificite"]
    with pytest.raises(KeyError, match="specificite"):
        parse_judge_response(payload)


def test_parse_judge_response_invalid_verdict_raises():
    with pytest.raises(ValueError, match="verdict"):
        parse_judge_response(_valid_judge_payload(verdict="garder"))


def test_build_topic_judge_prompt_contains_role_and_examples():
    prompt = build_topic_judge_prompt(
        "A1",
        12,
        ["fuite hydraulique sur presse"],
        ["manque de formation opérateur"],
    )
    assert "A1" in prompt
    assert "facteur défavorable" in prompt.lower() or "A1 :" in prompt
    assert "fuite hydraulique" in prompt
    assert "manque de formation" in prompt


def test_sample_topic_examples_from_top_sentences_and_assignments():
    themes_row = pd.Series(
        {
            "top_sentences": "phrase A || phrase B",
            "n_units": 10,
        }
    )
    assignments = pd.DataFrame(
        {
            "macro": ["A0", "A0", "A0"],
            "topic_id": [1, 1, 1],
            "doc_idx": [0, 1, 2],
        }
    )
    meta = pd.DataFrame({"sentence": ["s0", "s1", "s2"]})
    rep, rand_ex = sample_topic_examples(
        assignments,
        meta,
        "A0",
        1,
        "sentence",
        n_rep=2,
        n_random=1,
        themes_row=themes_row,
        rng=np.random.default_rng(0),
    )
    assert "phrase A" in rep[0]
    assert len(rand_ex) >= 1


def test_aggregate_judge_by_macro():
    scores = pd.DataFrame(
        [
            {"macro": "A0", "score_global": 4.0, "verdict": "conserver"},
            {"macro": "A0", "score_global": 2.0, "verdict": "rejeter"},
            {"macro": "B", "score_global": 3.5, "verdict": "fusionner"},
        ]
    )
    agg = aggregate_judge_by_macro(scores)
    assert len(agg) == 2
    a0 = agg.loc[agg["macro"] == "A0"].iloc[0]
    assert a0["n_topics_judged"] == 2
    assert a0["mean_score_global"] == pytest.approx(3.0)
    assert a0["pct_conserver"] == pytest.approx(50.0)
    assert a0["pct_rejeter"] == pytest.approx(50.0)


@patch("macro_transfer.topic_judge._get_client")
def test_run_topic_judge_evaluation_mock_openai(mock_get_client, tmp_path):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    payload = _valid_judge_payload()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
    mock_client.chat.completions.create.return_value = mock_resp

    themes = pd.DataFrame(
        [
            {
                "macro": "A0",
                "topic_id": 0,
                "n_units": 10,
                "top_sentences": "exemple représentatif",
            }
        ]
    )
    assignments = pd.DataFrame(
        {"macro": ["A0"], "topic_id": [0], "doc_idx": [0]}
    )
    meta = pd.DataFrame({"sentence": ["phrase test"]})

    result = run_topic_judge_evaluation(
        tmp_path,
        meta,
        assignments,
        themes,
        cfg={
            "enabled": True,
            "model": "gpt-5-mini",
            "n_representative": 1,
            "n_random": 1,
            "min_n_units": 8,
            "show_progress": False,
        },
        text_col="sentence",
        seed=42,
        client=mock_client,
        force=True,
    )
    assert result["n_topics"] == 1
    scores_path = tmp_path / "summary" / "topic_judge_scores.csv"
    assert scores_path.is_file()
    scores = pd.read_csv(scores_path)
    assert len(scores) == 1
    assert scores.iloc[0]["macro"] == "A0"
    assert float(scores.iloc[0]["score_global"]) == pytest.approx(3.6667, rel=1e-3)
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_kwargs
