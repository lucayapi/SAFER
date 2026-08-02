"""Évaluation classification partagée (LR sklearn, embeddings, OOD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from safer_core.classification_metrics import evaluate_macro_predictions
from macro_transfer.supervised_baseline import (
    _fit_pipeline,
    _predict_with_probs,
    build_classifier_pipeline,
    build_predictions_dataframe,
)
from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID
from supervised_macro_ft.class_balance import balanced_oversample_arrays

CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = ("accuracy", "macro_f1", "balanced_accuracy")
CV_CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = tuple(f"val_{k}" for k in CLASSIFICATION_METRIC_KEYS)
KFOLD_CLASSIFICATION_AGGREGATE_KEYS: tuple[str, ...] = CV_CLASSIFICATION_METRIC_KEYS + ("train_wall_time_sec",)
DEFAULT_CLASSIFIER = "logistic_regression"
DEFAULT_SELECTION_METRIC = "balanced_accuracy"

EMBEDDING_STEMS: tuple[str, ...] = ("btp", "metallurgie", "caou", "nicollin")

PredictionDetails = Dict[str, Any]

__all__ = [
    "CLASSIFICATION_METRIC_KEYS",
    "CV_CLASSIFICATION_METRIC_KEYS",
    "KFOLD_CLASSIFICATION_AGGREGATE_KEYS",
    "DEFAULT_CLASSIFIER",
    "DEFAULT_SELECTION_METRIC",
    "EMBEDDING_STEMS",
    "PredictionDetails",
    "resolve_test_corpora",
    "predictions_path",
    "save_corpus_predictions",
    "save_transfer_predictions_alias",
    "load_saved_predictions",
    "build_and_save_predictions",
    "evaluate_classifier_on_embeddings",
    "build_cv_summary_from_kfold",
]


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


def classifier_params(
    *,
    class_weight: Optional[str] = None,
    classifier_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if class_weight == "balanced":
        params["class_weight"] = "balanced"
    params.update(dict(classifier_overrides or {}))
    return params


def fit_logistic_on_embeddings(
    X_train: np.ndarray,
    y_train_int: np.ndarray,
    *,
    classifier: str = DEFAULT_CLASSIFIER,
    class_weight: Optional[str] = None,
    oversampling: bool = False,
    seed: int = 42,
    classifier_overrides: Optional[Mapping[str, Any]] = None,
):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train_int, dtype=np.int64)
    if oversampling:
        X, y = balanced_oversample_arrays(X, y, seed=seed)
    pipe = build_classifier_pipeline(
        classifier,
        classifier_params(
            class_weight=class_weight,
            classifier_overrides=classifier_overrides,
        ),
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
    return_details: bool = False,
) -> Union[dict[str, Any], Tuple[dict[str, Any], PredictionDetails]]:
    macros_list = list(macros or macro_names())
    pred_macro, probs, confidence, margin, entropy = _predict_with_probs(
        pipe, np.asarray(X, dtype=np.float64), macros_list
    )
    metrics_full = evaluate_macro_predictions(
        np.asarray(y_macro, dtype=object).astype(str),
        pred_macro,
        probs,
        macros_list,
    )
    metrics = {
        k: float(metrics_full.get(k, float("nan")))
        for k in CLASSIFICATION_METRIC_KEYS
        if k in metrics_full
    }
    if not return_details:
        return metrics
    details: PredictionDetails = {
        "pred_macro": pred_macro,
        "probs": probs,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "macros": macros_list,
    }
    return metrics, details


def predictions_path(out_dir: Path, corpus_id: str) -> Path:
    """Chemin canonique ``predictions/predictions_<corpus>.csv``."""
    return Path(out_dir) / "predictions" / f"predictions_{corpus_id}.csv"


def save_corpus_predictions(
    preds_df: pd.DataFrame,
    out_dir: Path,
    corpus_id: str,
    *,
    also_transfer_alias: bool = False,
) -> Path:
    """Écrit ``predictions/predictions_<corpus>.csv`` (+ alias transfer optionnel)."""
    out = Path(out_dir)
    path = predictions_path(out, corpus_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = preds_df.copy()
    if "corpus" not in df.columns:
        df.insert(0, "corpus", str(corpus_id))
    df.to_csv(path, index=False)
    if also_transfer_alias:
        save_transfer_predictions_alias(df, out)
    return path


def save_transfer_predictions_alias(preds_df: pd.DataFrame, out_dir: Path) -> Path:
    """Écrit ``transfer/target_macro_predictions.csv`` (compatible BERTopic / BN)."""
    transfer = Path(out_dir) / "transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    path = transfer / "target_macro_predictions.csv"
    df = preds_df.copy()
    if "pred_macro" in df.columns and "m_hat" not in df.columns:
        df["m_hat"] = df["pred_macro"].astype(str)
    if "confidence" in df.columns and "q_conf" not in df.columns:
        df["q_conf"] = pd.to_numeric(df["confidence"], errors="coerce")
    df.to_csv(path, index=False)
    return path


def load_saved_predictions(
    results_dir: Union[str, Path],
    corpus_id: str,
) -> Optional[pd.DataFrame]:
    """Charge ``predictions/predictions_<corpus>.csv`` si présent."""
    path = predictions_path(Path(results_dir), corpus_id)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def build_and_save_predictions(
    meta: pd.DataFrame,
    details: PredictionDetails,
    out_dir: Path,
    corpus_id: str,
    *,
    method_name: str,
    text_col: str = "sentence",
    group_col: str = "accident_id",
    label_col: Optional[str] = "pred_label",
    also_transfer_alias: bool = False,
) -> Tuple[pd.DataFrame, Path]:
    """Construit le DataFrame standard et l'écrit sous ``predictions/``."""
    macros = list(details.get("macros") or macro_names())
    preds = build_predictions_dataframe(
        meta,
        details["pred_macro"],
        details["probs"],
        details["confidence"],
        details["margin"],
        details["entropy"],
        macros=macros,
        method_name=method_name,
        text_col=text_col if text_col in meta.columns else (
            "sentence" if "sentence" in meta.columns else meta.columns[0]
        ),
        group_col=group_col,
        label_col=label_col,
    )
    path = save_corpus_predictions(
        preds, out_dir, corpus_id, also_transfer_alias=also_transfer_alias
    )
    return preds, path


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
