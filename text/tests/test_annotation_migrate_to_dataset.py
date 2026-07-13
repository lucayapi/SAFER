"""Tests migration annotation XLSX → dataset CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from annotation.migrate_to_dataset import (
    build_migration_plan,
    find_best_annotated_xlsx,
    migrate_one,
    resolve_dataset_id,
    validate_annotation_table,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accident_id": ["a1", "a1", "a2"],
            "fact_id": [1, 2, 1],
            "sentence": ["u1", "u2", "u3"],
            "accident_summary": ["r1", "r1", "r2"],
            "pred_label": ["A0", "A1", "B"],
            "pred_ok": [True, True, False],
        }
    )


def test_resolve_dataset_id_known():
    assert resolve_dataset_id("run_all_btp") == "btp"
    assert resolve_dataset_id("run_all_caou_chimie_plas") == "caou"


def test_resolve_dataset_id_override():
    assert resolve_dataset_id("run_all_btp", "custom") == "custom"


def test_find_best_annotated_prefers_final(tmp_path):
    run_dir = tmp_path / "run_all_btp"
    run_dir.mkdir()
    pass1 = run_dir / "model__pass1__annotated.xlsx"
    final = run_dir / "model__annotated_final.xlsx"
    _sample_df().to_excel(pass1, index=False)
    _sample_df().to_excel(final, index=False)
    assert find_best_annotated_xlsx(run_dir).name == final.name


def test_validate_required_columns(tmp_path):
    df = _sample_df().drop(columns=["pred_ok"])
    issues = validate_annotation_table(df, source=tmp_path / "x.xlsx")
    assert any("pred_ok" in issue for issue in issues)


def test_migrate_writes_csv(tmp_path):
    run_dir = tmp_path / "outputs" / "run_all_metallurgie"
    dataset_dir = tmp_path / "dataset"
    run_dir.mkdir(parents=True)
    xlsx = run_dir / "gpt__pass1__annotated.xlsx"
    _sample_df().to_excel(xlsx, index=False)

    run = {
        "run_id": "run_all_metallurgie",
        "run_dir": run_dir,
        "annotated_path": xlsx,
    }
    plan = build_migration_plan(run, dataset_dir=dataset_dir)
    assert plan.dest_csv.name == "data_metallurgie.csv"

    result = migrate_one(plan, dry_run=False, backup=False)
    assert result["dest"] == str(plan.dest_csv)
    assert plan.dest_csv.is_file()

    loaded = pd.read_csv(plan.dest_csv)
    assert len(loaded) == 3
    assert "pred_label" in loaded.columns


def test_migrate_dry_run_no_write(tmp_path):
    run_dir = tmp_path / "outputs" / "run_all_btp"
    dataset_dir = tmp_path / "dataset"
    run_dir.mkdir(parents=True)
    xlsx = run_dir / "annotated.xlsx"
    _sample_df().to_excel(xlsx, index=False)

    plan = build_migration_plan(
        {"run_id": "run_all_btp", "run_dir": run_dir, "annotated_path": xlsx},
        dataset_dir=dataset_dir,
    )
    migrate_one(plan, dry_run=True, backup=False)
    assert not plan.dest_csv.exists()
