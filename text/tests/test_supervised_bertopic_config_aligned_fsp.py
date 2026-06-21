"""BERTopic supervisé aligné sur frozen_source_prototypes.yaml (config partagée)."""

from __future__ import annotations

from pathlib import Path

from macro_transfer.bertopic_config import resolve_bertopic_run_config
from safer_core.io import load_yaml

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_supervised_bertopic_matches_frozen_source_prototypes():
    fsp = load_yaml(TEXT_ROOT / "configs" / "frozen_source_prototypes.yaml")
    sup = load_yaml(TEXT_ROOT / "configs" / "supervised_macro_baseline.yaml")
    fsp_bertopic, fsp_topics, fsp_judge = resolve_bertopic_run_config(fsp, anchor=TEXT_ROOT)
    sup_bertopic, sup_topics, sup_judge = resolve_bertopic_run_config(sup, anchor=TEXT_ROOT)
    assert fsp_bertopic == sup_bertopic
    assert fsp_topics == sup_topics
    assert fsp_judge == sup_judge
