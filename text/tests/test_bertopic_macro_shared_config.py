"""Tests config BERTopic partagée FSP / supervisé."""

from __future__ import annotations

from pathlib import Path

import yaml

from macro_transfer.bertopic_config import (
    enrich_run_config_bertopic,
    load_bertopic_macro_shared,
    resolve_bertopic_run_config,
)

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_shared_bertopic_uses_eom_and_nr_docs_50():
    shared = load_bertopic_macro_shared(anchor=TEXT_ROOT)
    rep = shared["bertopic"]["representation"]
    assert rep["nr_docs"] == 50
    assert rep["model"] == "gpt-5-mini"
    judge = shared["topic_judge"]
    assert judge["enabled"] is True
    assert judge["model"] == "gpt-5-mini"
    for macro in ("A0", "A1", "B", "C"):
        block = shared["bertopic"]["macro_params"][macro]
        assert block["cluster_selection_method"] == "eom"
    umap = shared["bertopic"]["umap"]
    assert umap["enabled"] is True
    assert umap["n_components"] == 15
    assert umap["metric"] == "cosine"
    assert shared["bertopic"]["diagnostics"]["show_progress"] is True
    assert shared["bertopic"]["diagnostics"]["save_datamap"] is False
    assert shared["bertopic"]["diagnostics"]["save_model"] is True


def test_supervised_and_fsp_resolve_same_bertopic():
    fsp_raw = yaml.safe_load(
        (TEXT_ROOT / "configs" / "frozen_source_prototypes.yaml").read_text(encoding="utf-8")
    )
    sup_raw = yaml.safe_load(
        (TEXT_ROOT / "configs" / "supervised_macro_baseline.yaml").read_text(encoding="utf-8")
    )
    fsp_bertopic, fsp_topics, fsp_judge = resolve_bertopic_run_config(fsp_raw, anchor=TEXT_ROOT)
    sup_bertopic, sup_topics, sup_judge = resolve_bertopic_run_config(sup_raw, anchor=TEXT_ROOT)
    assert fsp_bertopic == sup_bertopic
    assert fsp_topics == sup_topics
    assert fsp_judge == sup_judge


def test_enrich_run_config_injects_bertopic():
    raw = {"bertopic_shared": "configs/bertopic_macro_shared.yaml"}
    cfg = enrich_run_config_bertopic(raw, anchor=TEXT_ROOT)
    assert cfg["bertopic"]["macro_params"]["A0"]["cluster_selection_method"] == "eom"
    assert cfg["topics_export"]["top_k_words"] == 12
    assert cfg["topic_judge"]["model"] == "gpt-5-mini"
