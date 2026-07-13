"""Tests export XLSX annotation."""

from __future__ import annotations

import pandas as pd

from annotation.export_io import sanitize_excel_cell_value, save_annotation_table


def test_save_annotation_table_writes_xlsx(tmp_path):
    df = pd.DataFrame({"accident_id": ["A1"], "pred_label": ["B"]})
    path = tmp_path / "annotated.xlsx"
    out = save_annotation_table(df, path)
    assert out == path
    assert path.is_file()
    loaded = pd.read_excel(path, engine="openpyxl")
    assert loaded["pred_label"].iloc[0] == "B"


def test_save_annotation_table_strips_illegal_control_chars(tmp_path):
    dirty = "lanc\x01ée métallique"
    df = pd.DataFrame({"sentence": [dirty], "pred_rationale": [dirty]})
    path = tmp_path / "annotated.xlsx"
    save_annotation_table(df, path)
    loaded = pd.read_excel(path, engine="openpyxl")
    assert loaded["sentence"].iloc[0] == sanitize_excel_cell_value(dirty)
    assert "\x01" not in str(loaded["pred_rationale"].iloc[0])
