"""Évaluation classification partagée (LR sklearn, embeddings, OOD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from safer_core.classification_metrics import evaluate_macro_predictions
from macro_transfer.supervised_baseline import (
    _fit_pipeline,
    _predict_with_probs,
    build_classifier_pipeline,
)
from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID
from supervised_macro_ft.class_balance import balanced_oversample_arrays

CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = ("accuracy", "macro_f1", "balanced_accuracy")
CV_CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = tuple(f"val_{k}" for k in CLASSIFICATION_METRIC_KEYS)
KFOLD_CLASSIFICATION_AGGREGATE_KEYS: tuple[str, ...] = CV_CLASSIFICATION_METRIC_KEYS + ("train_wall_time_sec",)
DEFAULT_CLASSIFIER = "logistic_regression"
DEFAULT_SELECTION_METRIC = "balanced_accuracy"

EMBEDDING_STEMS: tuple[str, ...] = ("btp", "metallurgie", "caou")


def resolve_test_corpora(cfg: Mapping[str, Any]) -> list[str]:
    corpora = cfg.get("test_corpora")
    if corpora:
        return [str(c) for c in corpora]
    legacy = cfg.get("test_corpus")
    if legacy:
        return [str(legacy)]
    return ["metallurgie"]


def macro_names() -> list[str]:
    return [ID2LABEL[i] for i in range(len(LABEL2ID))]


def classifier_params(*, class_weight: Optional[str] = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if class_weight == "balanced":
        params["class_weight"] = "balanced"
    return params


def fit_logistic_on_embeddings(
    X_train: np.ndarray,
    y_train_int: np.ndarray,
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    class_weight: Optional[str] = None,
    oversampling: bool = False,
    seed: int = 42,
):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train_int, dtype=np.int64)
    if oversampling:
        X, y = balanced_oversample_arrays(X, y, seed=seed)
    pipe = build_classifier_pipeline(classifier, classifier_params(class_weight=class_weight), seed=seed)
    _fit_pipeline(pipe, X, y, seed=seed)
    return pipe


def evaluate_classifier_on_embeddings(
    pipe,
    X: np.ndarray,
    y_macro: Sequence[str],
    *,
    macros: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    macros_list = list(macros or macro_names())
    pred_macro, probs, _, _, _ = _predict_with_probs(pipe, np.asarray(X, dtype=np.float64), macros_list)
    metrics = evaluate_macro_predictions(
        np.asarray(y_macro, dtype=object).astype(str),
        pred_macro,
        probs,
        macros_list,
    )
    return {
        k: float(metrics.get(k, float("nan")))
        for k in CLASSIFICATION_METRIC_KEYS
        if k in metrics
    }


def fit_logistic_and_evaluate(
    X_train: np.ndarray,
    y_train_int: np.ndarray,
    X_eval: np.ndarray,
    y_eval_macro: Sequence[str],
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    class_weight: Optional[str] = None,
    oversampling: bool = False,
    seed: int = 42,
    macros: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    pipe = fit_logistic_on_embeddings(
        X_train,
        y_train_int,
        classifier=classifier,
        class_weight=class_weight,
        oversampling=oversampling,
        seed=seed,
    )
    return evaluate_classifier_on_embeddings(pipe, X_eval, y_eval_macro, macros=macros)


def export_projected_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    emb_dir: Path,
    stem: str,
    *,
    label_col: Optional[str] = None,
    group_col: Optional[str] = None,
    text_col: Optional[str] = None,
) -> tuple[Path, Path]:
    """Écrit ``projected_<stem>.npy`` + metadata CSV."""
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    npy_path = emb_dir / f"projected_{stem}.npy"
    meta_path = emb_dir / f"projected_{stem}_metadata.csv"
    np.save(npy_path, np.asarray(embeddings, dtype=np.float64))

    cols = []
    for col in (group_col, "doc_id", text_col, label_col):
        if col and col in metadata_df.columns:
            cols.append(col)
    meta = metadata_df[cols].copy() if cols else metadata_df.copy()
    meta.to_csv(meta_path, index=False)
    return npy_path, meta_path


def save_classification_metrics_csv(row: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(path, index=False)
    return path


def build_all_test_corpora_metrics_table(
    test_metrics_by_corpus: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for corpus_id, metrics in test_metrics_by_corpus.items():
        row: dict[str, Any] = {"corpus": str(corpus_id)}
        for key in CLASSIFICATION_METRIC_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _cv_ba_from_summary(cv_summary: pd.DataFrame) -> tuple[float, float]:
    if cv_summary.empty:
        return float("nan"), float("nan")
    row = cv_summary.iloc[0]
    for mean_key, std_key in (
        ("mean_balanced_accuracy", "std_balanced_accuracy"),
        ("mean_val_balanced_accuracy", "std_val_balanced_accuracy"),
    ):
        if mean_key in row.index:
            return float(row.get(mean_key, float("nan"))), float(row.get(std_key, float("nan")))
    return float("nan"), float("nan")


def summarize_ood_classification(
    test_metrics_by_corpus: Mapping[str, Mapping[str, Any]],
    cv_summary: pd.DataFrame,
    *,
    model_name: str = "model",
) -> pd.DataFrame:
    if not test_metrics_by_corpus:
        return pd.DataFrame()

    cv_mean, cv_std = _cv_ba_from_summary(cv_summary)
    ba_values = [
        float(metrics.get("balanced_accuracy", float("nan")))
        for metrics in test_metrics_by_corpus.values()
    ]
    ba_values = [v for v in ba_values if np.isfinite(v)]

    record: dict[str, Any] = {
        "model": model_name,
        "cv_ba_mean": cv_mean,
        "cv_ba_std": cv_std,
        "cv_ba": f"{cv_mean:.2f} ± {cv_std:.2f}",
        "ba_ood_avg": float(np.mean(ba_values)) if ba_values else float("nan"),
        "ba_ood_worst": float(np.min(ba_values)) if ba_values else float("nan"),
    }
    for corpus_id, metrics in test_metrics_by_corpus.items():
        cid = str(corpus_id)
        for key in CLASSIFICATION_METRIC_KEYS:
            record[f"{key}_{cid}"] = metrics.get(key)
    return pd.DataFrame([record])


def save_classification_outputs(
    out_dir: Path,
    *,
    method_name: str,
    metrics_by_corpus: Mapping[str, Mapping[str, Any]],
    cv_summary: pd.DataFrame,
    classifier: str = DEFAULT_CLASSIFIER,
) -> dict[str, Path]:
    """Écrit CSV classification + agrégats OOD sous ``metrics/``."""
    metrics_dir = Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for corpus_id, metrics in metrics_by_corpus.items():
        cid = str(corpus_id)
        if cid == "btp":
            fname = "metrics_classification_btp.csv"
        else:
            fname = f"metrics_classification_test_{cid}.csv"
        row = {"corpus": cid, "classifier": classifier, **metrics}
        paths[cid] = save_classification_metrics_csv(row, metrics_dir / fname)

    ood_only = {k: v for k, v in metrics_by_corpus.items() if str(k) != "btp"}
    all_test = build_all_test_corpora_metrics_table(ood_only)
    if not all_test.empty:
        p = metrics_dir / "all_test_corpora_metrics.csv"
        all_test.to_csv(p, index=False)
        paths["all_test_corpora"] = p

    cross = summarize_ood_classification(ood_only, cv_summary, model_name=method_name)
    if not cross.empty:
        p = metrics_dir / "cross_domain_generalization.csv"
        cross.to_csv(p, index=False)
        paths["cross_domain"] = p
    return paths


def build_cv_summary_from_kfold(kfold_summary: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    """Convertit kfold_summary (val_*) en cv_summary (mean_balanced_accuracy)."""
    if kfold_summary.empty:
        return pd.DataFrame()
    row = kfold_summary.iloc[0].to_dict()
    out: dict[str, Any] = {"model": model_name, "n_folds": row.get("n_folds")}
    mapping = {
        "balanced_accuracy": ("mean_val_balanced_accuracy", "std_val_balanced_accuracy"),
        "accuracy": ("mean_val_accuracy", "std_val_accuracy"),
        "macro_f1": ("mean_val_macro_f1", "std_val_macro_f1"),
    }
    for metric, (mean_key, std_key) in mapping.items():
        if mean_key in row:
            out[f"mean_{metric}"] = row.get(mean_key)
            out[f"std_{metric}"] = row.get(std_key)
    return pd.DataFrame([out])
