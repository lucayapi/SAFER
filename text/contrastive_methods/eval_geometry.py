"""Évaluation géométrique val (δ_macro = 100×η²) pour sélection de checkpoint."""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from metrics.geometry import build_geometry_metrics_row

SELECTION_METRIC_DEFAULT = "delta_macro_pct"


def selection_score(row: Dict[str, Any], metric: str = SELECTION_METRIC_DEFAULT) -> float:
    value = float(row.get(metric, float("nan")))
    if not np.isfinite(value):
        return float("-inf")
    return value


def evaluate_embeddings_geometry(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "val",
) -> Dict[str, Any]:
    return build_geometry_metrics_row(
        embeddings,
        labels,
        method=method,
        l2_normalize=True,
    )


def evaluate_st_val_geometry(
    model,
    val_df: pd.DataFrame,
    cfg: ContrastiveConfig,
    text_col: str,
) -> Dict[str, Any]:
    texts = val_df[text_col].astype(str).tolist()
    labels = val_df[cfg.label_col].to_numpy()
    emb = model.encode(
        texts,
        batch_size=cfg.eval_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return evaluate_embeddings_geometry(np.asarray(emb), labels, method="val")


def evaluate_hf_val_geometry(
    encoder,
    val_df: pd.DataFrame,
    cfg: ContrastiveConfig,
    text_col: str,
    device: str,
) -> Dict[str, Any]:
    from contrastive_methods.losses.softtriple import encode_texts_with_hf_encoder

    texts = val_df[text_col].astype(str).tolist()
    labels = val_df[cfg.label_col].to_numpy()
    emb = encode_texts_with_hf_encoder(
        encoder,
        texts,
        batch_size=cfg.eval_batch_size,
        device=device,
        normalize_embeddings=True,
        max_length=cfg.max_seq_length,
    )
    return evaluate_embeddings_geometry(emb, labels, method="val")
