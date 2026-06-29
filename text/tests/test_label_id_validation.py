"""Tests filtrage labels (évite NaN → int64 invalide en CE)."""

from __future__ import annotations

import pandas as pd
import pytest

from scgm_text.data_metadata import load_filtered_metadata
from scgm_text.dataset_text_raw import TextRawDataset


def test_load_filtered_metadata_excludes_nan_pred_label(tmp_path):
  csv = tmp_path / "mini.csv"
  pd.DataFrame(
      {
          "accident_id": ["a1", "a2", "a3"],
          "sentence": ["t1", "t2", "t3"],
          "pred_label": ["A0", "nan", "B"],
          "pred_ok": [True, True, True],
      }
  ).to_csv(csv, index=False)
  meta = load_filtered_metadata(str(csv), text_col="sentence")
  assert len(meta) == 2
  assert set(meta["pred_label"]) == {"A0", "B"}
  assert meta["label_id"].between(0, 3).all()


def test_text_raw_dataset_rejects_nan_label_id(tmp_path):
  csv = tmp_path / "bad.csv"
  pd.DataFrame(
      {
          "accident_id": ["a1"],
          "sentence": ["t1"],
          "pred_label": ["A0"],
          "pred_ok": [True],
          "label_id": [float("nan")],
          "row_id": [0],
      }
  ).to_csv(csv, index=False)
  df = pd.read_csv(csv)
  with pytest.raises(ValueError, match="NaN"):
      TextRawDataset(str(csv), metadata_df=df, text_col="sentence")
