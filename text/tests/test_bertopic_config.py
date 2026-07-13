"""Tests fusion config BERTopic (judge_enable, shared yaml)."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from macro_transfer.bertopic_config import enrich_run_config_bertopic, resolve_bertopic_run_config


def test_judge_enable_false_overrides_shared_enabled():
    cfg = {
        "bertopic_shared": "configs/bertopic_macro_shared.yaml",
        "judge_enable": False,
    }
    enriched = enrich_run_config_bertopic(cfg, anchor=TEXT_ROOT)
    assert enriched["topic_judge"]["enabled"] is False
    assert enriched["judge_enable"] is False


def test_judge_enable_true_keeps_judge_on():
    cfg = {
        "bertopic_shared": "configs/bertopic_macro_shared.yaml",
        "judge_enable": True,
    }
    enriched = enrich_run_config_bertopic(cfg, anchor=TEXT_ROOT)
    assert enriched["topic_judge"]["enabled"] is True


def test_explicit_topic_judge_enabled_wins_over_judge_enable():
    cfg = {
        "bertopic_shared": "configs/bertopic_macro_shared.yaml",
        "judge_enable": False,
        "topic_judge": {"enabled": True},
    }
    _, _, topic_judge = resolve_bertopic_run_config(cfg, anchor=TEXT_ROOT)
    assert topic_judge["enabled"] is True


def test_supervised_baseline_yaml_loads_judge_enable():
    from safer_core.io import load_yaml

    spec = load_yaml(TEXT_ROOT / "configs" / "supervised_macro_baseline.yaml")
    enriched = enrich_run_config_bertopic(spec, anchor=TEXT_ROOT)
    assert enriched["topic_judge"]["enabled"] is True
