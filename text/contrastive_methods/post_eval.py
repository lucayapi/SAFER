"""Post-évaluation classification (logistic sklearn) sur embeddings contrastifs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.hf_training_common import encode_texts, load_contrastive_checkpoint
from macro_transfer.frozen_source_prototypes import evaluate_macro_predictions
from macro_transfer.supervised_baseline import (
    CV_METRIC_KEYS,
    _fit_pipeline,
    _labels_to_int,
    _predict_with_probs,
    build_classifier_pipeline,
)
from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID
from supervised_macro_ft.class_balance import balanced_oversample_arrays
from safer_core.io import save_metrics_geometry

CLASSIFICATION_METRIC_KEYS: Tuple[str, ...] = CV_METRIC_KEYS
CV_CLASSIFICATION_METRIC_KEYS: Tuple[str, ...] = tuple(f"val_{k}" for k in CLASSIFICATION_METRIC_KEYS)


def _macro_names() -> List[str]:
    return [ID2LABEL[i] for i in range(len(LABEL2ID))]


def post_eval_balance_cfg(cfg: ContrastiveConfig) -> Dict[str, Any]:
    return {
        "oversampling": cfg.post_eval_oversampling,
        "class_weight": cfg.post_eval_class_weight,
    }


def _classifier_params(cfg: ContrastiveConfig) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if cfg.post_eval_class_weight == "balanced":
        params["class_weight"] = "balanced"
    return params


def fit_classifier_on_embeddings(
    X_train: np.ndarray,
    y_train_int: np.ndarray,
    cfg: ContrastiveConfig,
    *,
    seed: int = 42,
):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train_int, dtype=np.int64)
    if cfg.post_eval_oversampling:
        X, y = balanced_oversample_arrays(X, y, seed=seed)
    pipe = build_classifier_pipeline(
        cfg.post_eval_classifier,
        _classifier_params(cfg),
        seed=seed,
    )
    _fit_pipeline(pipe, X, y, seed=seed)
    return pipe


def evaluate_classifier_on_embeddings(
    pipe,
    X: np.ndarray,
    y_macro: Sequence[str],
    *,
    macros: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    macros_list = list(macros or _macro_names())
    pred_macro, probs, _, _, _ = _predict_with_probs(pipe, np.asarray(X, dtype=np.float64), macros_list)
    return evaluate_macro_predictions(
        np.asarray(y_macro, dtype=object).astype(str),
        pred_macro,
        probs,
        macros_list,
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
    macros = _macro_names()
    y_train_int = train_df["label_id"].astype(int).to_numpy()
    y_val_macro = val_df[cfg.label_col].astype(str).to_numpy()
    pipe = fit_classifier_on_embeddings(X_train, y_train_int, cfg, seed=cfg.seed)
    metrics = evaluate_classifier_on_embeddings(pipe, X_val, y_val_macro, macros=macros)
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
    macros = _macro_names()
    y_train_int = train_df["label_id"].astype(int).to_numpy()
    y_eval_macro = eval_df[cfg.label_col].astype(str).to_numpy()
    pipe = fit_classifier_on_embeddings(X_train, y_train_int, cfg, seed=cfg.seed)
    metrics = evaluate_classifier_on_embeddings(pipe, X_eval, y_eval_macro, macros=macros)
    row = {
        "corpus": corpus,
        "classifier": cfg.post_eval_classifier,
        **{k: float(metrics.get(k, float("nan"))) for k in CLASSIFICATION_METRIC_KEYS},
    }
    if metrics_dir is not None:
        save_metrics_geometry(row, metrics_dir, stem=f"metrics_classification_{corpus}")
    return row


def classification_summary_row(row: Mapping[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in CLASSIFICATION_METRIC_KEYS:
        src = f"{prefix}{key}" if prefix else key
        if src in row:
            out[src] = row[src]
    return out
