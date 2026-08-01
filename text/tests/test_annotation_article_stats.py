"""Tests statistiques article multi-corpus."""

from __future__ import annotations

import json

import pandas as pd

from annotation.article_stats import (
    ARTICLE_PLOT_COLORS,
    article_confusion_cmap,
    article_plot_colors,
    build_label_distribution_long,
    build_label_distribution_sensitivity,
    build_mixed_combination_long,
    build_mixed_units_table,
    build_summary_table,
    corpus_display_name,
    corpus_annotation_summary,
    discover_annotation_runs,
    discover_dataset_corpora,
    find_annotated_xlsx,
    mixed_units_mask,
    mixed_units_summary,
    read_annotation_table,
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
            # Récit sans aucune unité valide → exclu de n_recits.
            {"accident_id": "A3", "fact_id": 5, "pred_label": None, "pred_ok": False},
        ]
    )
    summary = corpus_annotation_summary(df, corpus="Test")
    assert summary["n_recits"] == 2
    assert summary["n_units"] == 5
    assert summary["n_annotated"] == 3
    assert summary["n_missing_or_failed"] == 2
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
    summary = build_summary_table(runs, include_overall=False)
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


def test_corpus_display_name_nicollin():
    assert corpus_display_name("run_all_nicollin") == "Company corpus"


def test_discover_full_runs_only_excludes_partial(tmp_path):
    for run_id, rows in [
        (
            "run_all_btp",
            [{"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True}],
        ),
        (
            "run_partial_btp",
            [{"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True}],
        ),
    ]:
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        _write_annotated_xlsx(run_dir / "model__annotated.xlsx", rows)
        (run_dir / "run_config.json").write_text(
            json.dumps({"run_id": run_id}),
            encoding="utf-8",
        )

    all_runs = discover_annotation_runs(tmp_path)
    assert len(all_runs) == 2
    full = discover_annotation_runs(tmp_path, full_runs_only=True)
    assert len(full) == 1
    assert full[0]["run_id"] == "run_all_btp"


def test_build_summary_includes_overall(tmp_path):
    run_dir = tmp_path / "run_all_btp"
    run_dir.mkdir()
    _write_annotated_xlsx(
        run_dir / "model__annotated.xlsx",
        [
            {"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
            {"accident_id": "A1", "fact_id": 2, "pred_label": "B", "pred_ok": True},
        ],
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_id": "run_all_btp"}),
        encoding="utf-8",
    )
    runs = discover_annotation_runs(tmp_path, full_runs_only=True)
    summary = build_summary_table(runs, include_overall=True)
    assert "Overall" in set(summary["corpus"])
    overall = summary.loc[summary["corpus"] == "Overall"].iloc[0]
    assert int(overall["n_units"]) == 2
    assert int(overall["n_annotated"]) == 2


def test_mixed_units_summary_and_mask():
    df = pd.DataFrame(
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "pred_label": "A0",
                "pred_ok": True,
                "pred_ambiguity_type": "NONE",
            },
            {
                "accident_id": "A1",
                "fact_id": 2,
                "pred_label": "B",
                "pred_ok": True,
                "pred_ambiguity_type": "B_C",
            },
            {
                "accident_id": "A1",
                "fact_id": 3,
                "pred_label": "A1",
                "pred_ok": True,
                "pred_ambiguity_type": "A1_B",
            },
            {
                "accident_id": "A1",
                "fact_id": 4,
                "pred_label": "C",
                "pred_ok": False,
                "pred_ambiguity_type": "B_C",
            },
            {
                "accident_id": "A1",
                "fact_id": 5,
                "pred_label": "A0",
                "pred_ok": True,
                "pred_ambiguity_type": "REFERENCE",
            },
        ]
    )
    mask = mixed_units_mask(df)
    assert int(mask.sum()) == 2
    summary = mixed_units_summary(df, corpus="Test")
    assert summary["n_annotated"] == 4
    assert summary["n_mixed"] == 2
    assert summary["pct_mixed"] == 50.0
    assert summary["n_B_C"] == 1
    assert summary["n_A1_B"] == 1
    assert summary["n_A0_A1"] == 0


def test_mixed_tables_and_sensitivity(tmp_path):
    run_dir = tmp_path / "run_all_btp"
    run_dir.mkdir()
    _write_annotated_xlsx(
        run_dir / "model__annotated.xlsx",
        [
            {
                "accident_id": "A1",
                "fact_id": 1,
                "pred_label": "A0",
                "pred_ok": True,
                "pred_ambiguity_type": "NONE",
            },
            {
                "accident_id": "A1",
                "fact_id": 2,
                "pred_label": "B",
                "pred_ok": True,
                "pred_ambiguity_type": "B_C",
            },
            {
                "accident_id": "A1",
                "fact_id": 3,
                "pred_label": "C",
                "pred_ok": True,
                "pred_ambiguity_type": "A0_A1",
            },
            {
                "accident_id": "A1",
                "fact_id": 4,
                "pred_label": "A1",
                "pred_ok": True,
                "pred_ambiguity_type": "NONE",
            },
        ],
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_id": "run_all_btp"}),
        encoding="utf-8",
    )
    runs = discover_annotation_runs(tmp_path, full_runs_only=True)

    mixed_table = build_mixed_units_table(runs, include_overall=True)
    assert "Overall" in set(mixed_table["corpus"])
    btp = mixed_table.loc[mixed_table["corpus"] == "BTP"].iloc[0]
    assert int(btp["n_mixed"]) == 2
    assert float(btp["pct_mixed"]) == 50.0

    comb = build_mixed_combination_long(runs)
    assert set(comb["combination"]) >= {"A0/A1", "A1/B", "B/C", "MULTIPLE_ROLES"}
    bc = comb.loc[comb["combination"] == "B/C"].iloc[0]
    assert int(bc["n_units"]) == 1
    assert float(bc["pct_of_mixed"]) == 50.0

    sens = build_label_distribution_sensitivity(runs)
    assert len(sens) == 1
    row = sens.iloc[0]
    assert int(row["n_excluded_mixed"]) == 2
    assert int(row["n_after_exclusion"]) == 2
    # Sans mixtes : A0 et A1 seulement → 50 % chacun
    assert float(row["pct_A0_excl_mixed"]) == 50.0
    assert float(row["pct_A1_excl_mixed"]) == 50.0
    assert float(row["pct_B_excl_mixed"]) == 0.0
    assert float(row["pct_C_excl_mixed"]) == 0.0


def test_discover_dataset_corpora_and_summary(tmp_path):
    for corpus_id, rows in [
        (
            "btp",
            [
                {"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
                {"accident_id": "A1", "fact_id": 2, "pred_label": "B", "pred_ok": True},
            ],
        ),
        (
            "caou",
            [
                {"accident_id": "C1", "fact_id": 1, "pred_label": "C", "pred_ok": True},
            ],
        ),
    ]:
        pd.DataFrame(rows).to_csv(tmp_path / f"data_{corpus_id}.csv", index=False)

    # Nicollin fourni uniquement en XLSX (comme en local).
    pd.DataFrame(
        [
            {"accident_id": "N1", "fact_id": 1, "pred_label": "A0", "pred_ok": True},
            {"accident_id": "N1", "fact_id": 2, "pred_label": "B", "pred_ok": True},
        ]
    ).to_excel(tmp_path / "data_nicollin.xlsx", index=False)

    corpora = discover_dataset_corpora(tmp_path)
    assert [c["corpus_id"] for c in corpora] == ["btp", "caou", "nicollin"]
    assert corpora[0]["corpus"] == "Construction"
    assert corpora[1]["corpus"] == "Chemistry--plastics"
    assert corpora[2]["corpus"] == "Company corpus"
    assert corpora[2]["annotated_path"].suffix.lower() == ".xlsx"

    summary = build_summary_table(corpora, include_overall=True)
    assert set(summary["corpus"]) >= {
        "Construction",
        "Chemistry--plastics",
        "Company corpus",
        "Overall",
    }

    dist = build_label_distribution_long(corpora)
    assert len(dist) == 12

    colors = article_plot_colors(3)
    assert len(colors) == 3
    assert colors[0] == ARTICLE_PLOT_COLORS[0]
    cmap = article_confusion_cmap()
    assert cmap.N >= 2


def test_discover_dataset_prefers_csv_over_xlsx(tmp_path):
    rows = [{"accident_id": "A1", "fact_id": 1, "pred_label": "A0", "pred_ok": True}]
    pd.DataFrame(rows).to_excel(tmp_path / "data_btp.xlsx", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "data_btp.csv", index=False)
    corpora = discover_dataset_corpora(tmp_path)
    assert len(corpora) == 1
    assert corpora[0]["annotated_path"].name == "data_btp.csv"
