"""Tests statistiques article multi-corpus."""

from __future__ import annotations

import json

import pandas as pd

from annotation.article_stats import (
    build_label_distribution_long,
    build_summary_table,
    corpus_display_name,
    corpus_annotation_summary,
    discover_annotation_runs,
    find_annotated_xlsx,
)


def _write_annotated_xlsx(path, rows):
    pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")


def test_discover_annotation_runs_finds_annotated(tmp_path):
    run_dir = tmp_path / "run_all_btp"
    run_dir.mkdir()
    annotated = run_dir / "model__prompt__pass1__annotated.xlsx"
    _write_annotated_xlsx(
        annotated,
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "pred_label": "A0",
                "pred_ok": True,
            },
            {
                "accident_id": "A1",
                "fact_id": 2,
                "pred_label": "B",
                "pred_ok": True,
            },
        ],
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_id": "run_all_btp", "input_csv": "btp_sentence_accidents.csv"}),
        encoding="utf-8",
    )

    runs = discover_annotation_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["corpus"] == "BTP"
    assert runs[0]["annotated_path"] == annotated


def test_corpus_annotation_summary_percentages():
    df = pd.DataFrame(
        [
            {"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
            {"accident_id": "A1", "fact_id": 2, "pred_label": "A0", "pred_ok": True},
            {"accident_id": "A2", "fact_id": 3, "pred_label": "B", "pred_ok": True},
            {"accident_id": "A2", "fact_id": 4, "pred_label": "C", "pred_ok": False},
        ]
    )
    summary = corpus_annotation_summary(df, corpus="Test")
    assert summary["n_recits"] == 2
    assert summary["n_units"] == 4
    assert summary["n_annotated"] == 3
    assert summary["pct_A0"] == round(200 / 3, 2)
    assert summary["pct_B"] == round(100 / 3, 2)


def test_build_summary_and_distribution_tables(tmp_path):
    for run_id, corpus, rows in [
        (
            "run_all_btp",
            "BTP",
            [
                {"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
                {"accident_id": "A1", "fact_id": 2, "pred_label": "B", "pred_ok": True},
            ],
        ),
        (
            "run_all_metallurgie",
            "Métallurgie",
            [
                {"accident_id": "M1", "fact_id": 1, "pred_label": "C", "pred_ok": True},
            ],
        ),
    ]:
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        _write_annotated_xlsx(
            run_dir / "model__annotated.xlsx",
            rows,
        )
        (run_dir / "run_config.json").write_text(
            json.dumps({"run_id": run_id}),
            encoding="utf-8",
        )

    runs = discover_annotation_runs(tmp_path)
    summary = build_summary_table(runs)
    assert len(summary) == 2
    assert set(summary["corpus"]) == {"BTP", "Métallurgie"}

    dist = build_label_distribution_long(runs)
    assert len(dist) == 8
    assert set(dist["label"]) == {"A0", "A1", "B", "C"}


def test_corpus_display_name_from_input_csv():
    name = corpus_display_name(
        "custom_run",
        {"input_csv": "caou_chimie_plas_sentence_accidents.csv"},
    )
    assert name == "Caou Chimie Plas"
