"""Post-évaluation classification (logistic sklearn) sur embeddings contrastifs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.hf_training_common import encode_texts, load_contrastive_checkpoint
from safer_core.classification_eval import (
    CLASSIFICATION_METRIC_KEYS,
    CV_CLASSIFICATION_METRIC_KEYS,
    evaluate_classifier_on_embeddings,
    fit_logistic_on_embeddings,
    save_classification_metrics_csv,
)

__all__ = [
    "CLASSIFICATION_METRIC_KEYS",
    "CV_CLASSIFICATION_METRIC_KEYS",
    "evaluate_classifier_on_embeddings",
    "fit_classifier_on_embeddings",
    "run_post_eval_on_fold",
    "run_post_eval_on_corpus",
]


def post_eval_balance_cfg(cfg: ContrastiveConfig) -> Dict[str, Any]:
    return {
        "oversampling": cfg.post_eval_oversampling,
        "class_weight": cfg.post_eval_class_weight,
    }


def fit_classifier_on_embeddings(
    X_train: np.ndarray,
    y_train_int: np.ndarray,
    cfg: ContrastiveConfig,
    *,
    seed: int = 42,
):
    return fit_logistic_on_embeddings(
        X_train,
        y_train_int,
        classifier=cfg.post_eval_classifier,
        class_weight=cfg.post_eval_class_weight,
        oversampling=cfg.post_eval_oversampling,
        seed=seed,
    )


def run_post_eval_on_fold(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    device: str,
) -> Dict[str, float]:
    """Fit logistic sur train fold, évalue sur val fold."""
    if not cfg.post_eval_enabled:
        return {}
    encoder = load_contrastive_checkpoint(cfg, Path(checkpoint_dir), device)
    train_texts = train_df[text_col].astype(str).tolist()
    val_texts = val_df[text_col].astype(str).tolist()
    X_train = encode_texts(encoder, train_texts, cfg, device)
    X_val = encode_texts(encoder, val_texts, cfg, device)
    y_train_int = train_df["label_id"].astype(int).to_numpy()
    y_val_macro = val_df[cfg.label_col].astype(str).to_numpy()
    pipe = fit_classifier_on_embeddings(X_train, y_train_int, cfg, seed=cfg.seed)
    metrics = evaluate_classifier_on_embeddings(pipe, X_val, y_val_macro)
    return {f"val_{k}": float(metrics.get(k, float("nan"))) for k in CLASSIFICATION_METRIC_KEYS}


def run_post_eval_on_corpus(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    text_col: str,
    device: str,
    *,
    corpus: str = "btp",
    metrics_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit logistic sur train (BTP), évalue sur eval_df (BTP ou test)."""
    if not cfg.post_eval_enabled:
        return {}
    encoder = load_contrastive_checkpoint(cfg, Path(checkpoint_dir), device)
    train_texts = train_df[text_col].astype(str).tolist()
    eval_texts = eval_df[text_col].astype(str).tolist()
    X_train = encode_texts(encoder, train_texts, cfg, device)
    X_eval = encode_texts(encoder, eval_texts, cfg, device)
    y_train_int = train_df["label_id"].astype(int).to_numpy()
    y_eval_macro = eval_df[cfg.label_col].astype(str).to_numpy()
    pipe = fit_classifier_on_embeddings(X_train, y_train_int, cfg, seed=cfg.seed)
    metrics = evaluate_classifier_on_embeddings(pipe, X_eval, y_eval_macro)
    row = {
        "corpus": corpus,
        "classifier": cfg.post_eval_classifier,
        **{k: float(metrics.get(k, float("nan"))) for k in CLASSIFICATION_METRIC_KEYS},
    }
    if metrics_dir is not None:
        stem = "metrics_classification_btp" if corpus == "btp" else f"metrics_classification_test_{corpus}"
        save_classification_metrics_csv(row, metrics_dir / f"{stem}.csv")
    return row
