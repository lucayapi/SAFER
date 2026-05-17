"""Métriques géométriques post-entraînement."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.export import dim_column_names
from metrics.geometry import build_geometry_metrics_row
from safer_core.io import save_metrics_geometry

METHOD_DISPLAY = {
    "batch_triplet": "Batch Triplet",
    "softtriple": "SoftTriple",
    "supcon": "SupCon",
}


def compute_and_save_geometry_metrics(
    embeddings_csv: Path,
    cfg: ContrastiveConfig,
    metrics_dir: Path,
) -> Dict[str, object]:
    emb_df = pd.read_csv(embeddings_csv)
    dim_cols = [c for c in emb_df.columns if c.startswith("dim_")]
    if not dim_cols:
        dim_cols = dim_column_names(emb_df.shape[1] - 1)
    from scgm_text.utils_io import create_doc_id_if_missing

    meta = create_doc_id_if_missing(pd.read_csv(cfg.dataset_path))
    merged = meta.merge(emb_df, on="doc_id", how="inner")
    emb = merged[dim_cols].to_numpy(dtype=float)
    labels = merged[cfg.label_col].to_numpy()
    display = METHOD_DISPLAY.get(cfg.method_name, cfg.method_name)
    row = build_geometry_metrics_row(emb, labels, method=display, l2_normalize=True)
    save_metrics_geometry(row, metrics_dir)
    return row
