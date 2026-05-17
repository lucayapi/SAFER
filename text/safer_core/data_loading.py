"""Chargement données BTP / métallurgie (wrappers)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from scgm_text.dataset_text_embeddings import load_filtered_metadata, merge_metadata_with_embeddings
from safer_core.paths import DATASET_DIR, TEXT_ROOT


def resolve_dataset_path(dataset_path: str | Path) -> Path:
    p = Path(dataset_path)
    if p.is_absolute():
        return p
    return (TEXT_ROOT / p).resolve()


def load_btp_metadata(
    dataset_path: Optional[str | Path] = None,
    *,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
) -> pd.DataFrame:
    path = resolve_dataset_path(dataset_path or DATASET_DIR / "data_btp.csv")
    return load_filtered_metadata(
        data_csv=str(path),
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
    )


def load_metadata_with_embeddings(
    data_csv: str | Path,
    emb_csv: str | Path,
    **kwargs,
) -> Tuple[pd.DataFrame, list]:
    meta = load_filtered_metadata(data_csv=str(resolve_dataset_path(data_csv)), **kwargs)
    emb_path = resolve_dataset_path(emb_csv)
    merged, dim_cols = merge_metadata_with_embeddings(meta, str(emb_path))
    return merged, dim_cols
