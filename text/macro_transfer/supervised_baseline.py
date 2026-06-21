"""Baseline supervisée macro sur embeddings Qwen bruts (GroupKFold + BERTopic)."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macro_transfer.bertopic_phase import run_bertopic_phase
from macro_transfer.constants import LABEL2ID, MACRO_NAMES
from macro_transfer.encode import load_target_metadata
from macro_transfer.frozen_source_prototypes import (
    _build_gating_from_predictions,
    evaluate_macro_predictions,
)
from safer_core.data_loading import load_metadata_with_embeddings
from macro_transfer.bertopic_config import enrich_run_config_bertopic
from safer_core.io import load_yaml
from safer_core.kfold_eval import group_kfold_splits
from safer_core.paths import TEXT_ROOT, resolve_repo_path
from safer_core.test_corpus import default_test_corpus_id, resolve_test_corpus
from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
from scgm_text.utils_io import create_doc_id_if_missing

logger = logging.getLogger(__name__)

CV_METRIC_KEYS: Tuple[str, ...] = ("accuracy", "macro_f1", "balanced_accuracy")
DEFAULT_SELECTION_METRIC = "macro_f1"

DEFAULT_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "use_scaler": True,
        "params": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "solver": "lbfgs",
        },
    },
    "random_forest": {
        "use_scaler": False,
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "class_weight": "balanced",
        },
    },
    "xgboost": {
        "use_scaler": False,
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
    },
    "mlp": {
        "use_scaler": True,
        "params": {
            "hidden_layer_sizes": (256, 128),
            "max_iter": 500,
            "early_stopping": True,
            "class_weight": "balanced",
        },
    },
}


def supervised_baseline_output_dir(
    corpus_id: Optional[str] = None,
    *,
    anchor: Optional[Path] = None,
) -> Path:
    """``output_test/<corpus>/supervised_baseline/``."""
    root = anchor or TEXT_ROOT
    cid = corpus_id or default_test_corpus_id()
    return (root / "output_test" / str(cid) / "supervised_baseline").resolve()


def merge_model_registry(
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fusionne ``DEFAULT_MODEL_REGISTRY`` avec des surcharges YAML/notebook."""
    reg = deepcopy(DEFAULT_MODEL_REGISTRY)
    if not overrides:
        return reg
    for key, block in overrides.items():
        if key not in reg:
            reg[key] = {"use_scaler": False, "params": {}}
        src = dict(block or {})
        if "params" in src:
            reg[key]["params"] = {**reg[key].get("params", {}), **dict(src["params"] or {})}
        for k, v in src.items():
            if k != "params":
                reg[key][k] = v
    return reg


def _labels_to_int(labels: Sequence[str], macros: Sequence[str]) -> np.ndarray:
    out = np.array([LABEL2ID.get(str(v), -1) for v in labels], dtype=np.int64)
    if (out < 0).any():
        bad = sorted({str(labels[i]) for i in np.where(out < 0)[0]})
        raise ValueError(f"Labels hors macros {list(macros)} : {bad[:5]}")
    return out


def _build_xgb_classifier(params: Dict[str, Any], seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost est requis pour le modèle 'xgboost'. Installez-le : pip install xgboost"
        ) from exc
    p = dict(params)
    p.setdefault("random_state", int(seed))
    p.setdefault("n_jobs", -1)
    return XGBClassifier(**p)


def build_classifier_pipeline(
    model_key: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    seed: int = 42,
    use_scaler: Optional[bool] = None,
) -> Pipeline:
    """Construit un pipeline sklearn pour un modèle du registre."""
    key = str(model_key)
    if key not in DEFAULT_MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu : {key!r}. Connus : {sorted(DEFAULT_MODEL_REGISTRY)}")
    spec = DEFAULT_MODEL_REGISTRY[key]
    p = {**dict(spec.get("params") or {}), **dict(params or {})}
    scale = bool(spec.get("use_scaler", False)) if use_scaler is None else bool(use_scaler)
    steps: List[Tuple[str, Any]] = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if key == "logistic_regression":
        est = LogisticRegression(random_state=int(seed), **p)
    elif key == "random_forest":
        est = RandomForestClassifier(random_state=int(seed), n_jobs=-1, **p)
    elif key == "xgboost":
        est = _build_xgb_classifier(p, seed)
    elif key == "mlp":
        mlp_class_weight = p.pop("class_weight", None)
        est = MLPClassifier(random_state=int(seed), **p)
    else:
        raise ValueError(f"Modèle non implémenté : {key!r}")
    steps.append(("clf", est))
    pipe = Pipeline(steps)
    if key == "mlp":
        pipe._mlp_class_weight = mlp_class_weight  # noqa: SLF001 — consommé par _fit_pipeline
    return pipe


def _resample_train_for_class_weight(
    X: np.ndarray,
    y: np.ndarray,
    class_weight: Any,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sur-échantillonnage équilibré (sklearn 1.5.x : MLP sans ``sample_weight``)."""
    if class_weight is None:
        return X, y
    if str(class_weight).strip().lower() != "balanced":
        raise ValueError(
            f"class_weight MLP : seule la valeur 'balanced' est supportée, reçu {class_weight!r}"
        )
    rng = np.random.RandomState(int(seed))
    classes = np.unique(y)
    if len(classes) == 0:
        return X, y
    counts = np.array([np.sum(y == cls) for cls in classes], dtype=np.int64)
    target_n = int(counts.max())
    parts: List[np.ndarray] = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            continue
        parts.append(rng.choice(cls_idx, size=target_n, replace=True))
    all_idx = np.concatenate(parts)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def _fit_pipeline(pipe: Pipeline, X: np.ndarray, y: np.ndarray, *, seed: int = 42) -> Pipeline:
    """Fit pipeline ; ``class_weight`` MLP → sur-échantillonnage (pas de ``sample_weight`` en 1.5.x)."""
    class_weight = getattr(pipe, "_mlp_class_weight", None)
    if class_weight is not None:
        X_fit, y_fit = _resample_train_for_class_weight(X, y, class_weight, seed=seed)
        pipe.fit(X_fit, y_fit)
    else:
        pipe.fit(X, y)
    return pipe


def _proba_as_macro_matrix(proba: np.ndarray, classes_: np.ndarray, macros: Sequence[str]) -> np.ndarray:
    """Réordonne ``predict_proba`` selon l'ordre ``macros`` (A0…C)."""
    cls = np.asarray(classes_, dtype=np.int64)
    out = np.zeros((proba.shape[0], len(macros)), dtype=np.float64)
    for i, macro in enumerate(macros):
        mid = LABEL2ID[str(macro)]
        pos = int(np.where(cls == mid)[0][0])
        out[:, i] = proba[:, pos]
    return out


def _predict_with_probs(
    model: Pipeline,
    X: np.ndarray,
    macros: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proba_raw = model.predict_proba(X)
    classes_ = model.named_steps["clf"].classes_
    probs = _proba_as_macro_matrix(proba_raw, classes_, macros)
    pred_ids = probs.argmax(axis=1)
    pred_macro = np.array([str(macros[i]) for i in pred_ids], dtype=object)
    confidence = probs.max(axis=1)
    sort_p = np.sort(probs, axis=1)
    margin = sort_p[:, -1] - sort_p[:, -2] if probs.shape[1] >= 2 else np.zeros(len(probs))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return pred_macro, probs, confidence, margin, entropy


def load_supervised_datasets(
    cfg: Mapping[str, Any],
    *,
    anchor: Optional[Path] = None,
) -> Dict[str, Any]:
    """Charge BTP (filtré) et corpus test avec embeddings Qwen bruts."""
    root = anchor or TEXT_ROOT
    source_cfg = dict(cfg.get("source") or {})
    target_cfg = dict(cfg.get("target") or {})
    corpus = str(cfg.get("corpus") or default_test_corpus_id())

    source_data = resolve_repo_path(
        str(source_cfg.get("dataset_path", "dataset/data_btp.csv")),
        repo_root=root,
    )
    source_emb = resolve_repo_path(
        str(source_cfg.get("emb_csv", "embeddings/Qwen3-Embedding-0.6B_btp.csv")),
        repo_root=root,
    )
    label_col = str(source_cfg.get("label_col", "pred_label"))
    group_col = str(source_cfg.get("group_col", "accident_id"))
    text_col = str(source_cfg.get("text_col", "sentence"))

    btp_meta, dim_cols = load_metadata_with_embeddings(
        source_data,
        source_emb,
        label_col=label_col,
        pred_ok_col=str(source_cfg.get("pred_ok_col", "pred_ok")),
        group_col=group_col,
    )
    X_btp = btp_meta[dim_cols].to_numpy(dtype=np.float64)
    y_btp = btp_meta[label_col].astype(str).to_numpy()
    groups_btp = btp_meta[group_col].astype(str).to_numpy()

    test_spec = resolve_test_corpus(corpus, anchor=root)
    target_data = resolve_repo_path(
        str(target_cfg.get("dataset_path") or test_spec.data_csv_str()),
        repo_root=root,
    )
    target_emb = resolve_repo_path(
        str(target_cfg.get("emb_csv") or test_spec.emb_csv_str()),
        repo_root=root,
    )
    target_text_col = str(target_cfg.get("text_col", text_col))
    target_label_col = str(target_cfg.get("label_col", label_col))
    target_group_col = str(target_cfg.get("group_col", group_col))

    test_meta = load_target_metadata(str(target_data), text_col=target_text_col)
    test_meta = create_doc_id_if_missing(test_meta)
    if target_group_col and target_group_col not in test_meta.columns:
        test_meta[target_group_col] = np.arange(len(test_meta))
    test_meta = test_meta[test_meta[target_text_col].astype(str).str.strip().ne("")].reset_index(drop=True)
    slim = test_meta.drop(columns=[c for c in test_meta.columns if c.startswith("dim_")], errors="ignore")
    test_merged, test_dim_cols = merge_metadata_with_embeddings(slim, str(target_emb))
    if len(test_merged) != len(test_meta):
        raise ValueError(
            f"Alignement embeddings test : metadata={len(test_meta)}, merged={len(test_merged)}"
        )
    X_test = test_merged[test_dim_cols].to_numpy(dtype=np.float64)

    return {
        "corpus_id": corpus,
        "macros": list(MACRO_NAMES),
        "dim_cols": dim_cols,
        "label_col": label_col,
        "text_col": text_col,
        "group_col": group_col,
        "target_label_col": target_label_col,
        "target_text_col": target_text_col,
        "target_group_col": target_group_col,
        "btp_meta": btp_meta,
        "X_btp": X_btp,
        "y_btp": y_btp,
        "groups_btp": groups_btp,
        "test_meta": test_merged,
        "X_test": X_test,
    }


def run_model_group_kfold_cv(
    model_key: str,
    X: np.ndarray,
    y: Sequence[str],
    groups: Sequence[str],
    *,
    macros: Sequence[str] = MACRO_NAMES,
    n_folds: int = 5,
    seed: int = 42,
    params: Optional[Mapping[str, Any]] = None,
    use_scaler: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """CV GroupKFold pour un modèle → une ligne de métriques par fold."""
    y_arr = np.asarray(y, dtype=object).astype(str)
    y_int = _labels_to_int(y_arr, macros)
    splits = group_kfold_splits(np.asarray(groups), n_folds, seed)
    rows: List[Dict[str, Any]] = []
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        pipe = build_classifier_pipeline(
            model_key,
            params,
            seed=seed,
            use_scaler=use_scaler,
        )
        _fit_pipeline(pipe, X[tr_idx], y_int[tr_idx], seed=seed + int(fold_id))
        pred_macro, probs, _, _, _ = _predict_with_probs(pipe, X[va_idx], macros)
        metrics = evaluate_macro_predictions(y_arr[va_idx], pred_macro, probs, macros)
        row: Dict[str, Any] = {
            "model": str(model_key),
            "fold_id": int(fold_id),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
        }
        for key in CV_METRIC_KEYS:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)
    return rows


def run_all_models_group_kfold_cv(
    model_keys: Sequence[str],
    X: np.ndarray,
    y: Sequence[str],
    groups: Sequence[str],
    *,
    model_registry: Optional[Mapping[str, Any]] = None,
    macros: Sequence[str] = MACRO_NAMES,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """CV pour chaque modèle du registre."""
    reg = merge_model_registry(model_registry)
    all_rows: List[Dict[str, Any]] = []
    for key in model_keys:
        spec = reg.get(str(key), {})
        rows = run_model_group_kfold_cv(
            str(key),
            X,
            y,
            groups,
            macros=macros,
            n_folds=n_folds,
            seed=seed,
            params=spec.get("params"),
            use_scaler=spec.get("use_scaler"),
        )
        all_rows.extend(rows)
    return all_rows


def aggregate_cv_metrics(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = CV_METRIC_KEYS,
) -> pd.DataFrame:
    """Agrège les folds → mean ± std par modèle."""
    if not fold_rows:
        return pd.DataFrame()
    df = pd.DataFrame(list(fold_rows))
    records: List[Dict[str, Any]] = []
    for model, grp in df.groupby("model", sort=True):
        rec: Dict[str, Any] = {"model": str(model), "n_folds": int(len(grp))}
        for key in metric_keys:
            vals = pd.to_numeric(grp[key], errors="coerce").dropna()
            rec[f"mean_{key}"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"std_{key}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        records.append(rec)
    return pd.DataFrame(records)


def select_best_model(
    cv_summary: pd.DataFrame,
    *,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> str:
    """Sélectionne le modèle avec le meilleur ``mean_<selection_metric>``."""
    if cv_summary.empty:
        raise ValueError("cv_summary vide : impossible de sélectionner un modèle.")
    col = f"mean_{selection_metric}"
    if col not in cv_summary.columns:
        raise KeyError(f"Colonne {col!r} absente du résumé CV.")
    idx = cv_summary[col].astype(float).idxmax()
    return str(cv_summary.loc[idx, "model"])


def build_predictions_dataframe(
    test_meta: pd.DataFrame,
    pred_macro: Sequence[str],
    probs: np.ndarray,
    confidence: Sequence[float],
    margin: Sequence[float],
    entropy: Sequence[float],
    *,
    macros: Sequence[str],
    method_name: str,
    text_col: str,
    group_col: str,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """Export compatible FSP / BERTopic gating."""
    preds = pd.DataFrame(
        {
            "method": method_name,
            "pred_macro": np.asarray(pred_macro, dtype=object),
            "confidence": np.asarray(confidence, dtype=np.float64),
            "margin": np.asarray(margin, dtype=np.float64),
            "entropy": np.asarray(entropy, dtype=np.float64),
            "sentence": test_meta[text_col].astype(str).to_numpy(),
        }
    )
    if "index" in test_meta.columns:
        preds["index"] = test_meta["index"].to_numpy()
    if group_col in test_meta.columns:
        preds[group_col] = test_meta[group_col].to_numpy()
    if "fact_id" in test_meta.columns:
        preds["fact_id"] = test_meta["fact_id"].to_numpy()
    for i, m in enumerate(macros):
        preds[f"prob_{m}"] = probs[:, i]
    if label_col and label_col in test_meta.columns:
        preds["true_macro"] = test_meta[label_col].astype(str).to_numpy()
    return preds


def fit_final_and_predict_test(
    model_key: str,
    X_btp: np.ndarray,
    y_btp: Sequence[str],
    X_test: np.ndarray,
    test_meta: pd.DataFrame,
    *,
    macros: Sequence[str] = MACRO_NAMES,
    seed: int = 42,
    params: Optional[Mapping[str, Any]] = None,
    use_scaler: Optional[bool] = None,
    method_name: Optional[str] = None,
    text_col: str = "sentence",
    group_col: str = "accident_id",
    label_col: Optional[str] = "pred_label",
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """Réentraîne sur 100 % BTP, prédit le corpus test, calcule les métriques."""
    y_int = _labels_to_int(y_btp, macros)
    pipe = build_classifier_pipeline(model_key, params, seed=seed, use_scaler=use_scaler)
    _fit_pipeline(pipe, X_btp, y_int, seed=seed)
    pred_macro, probs, confidence, margin, entropy = _predict_with_probs(pipe, X_test, macros)
    display_name = method_name or f"supervised_{model_key}_raw_qwen"
    preds = build_predictions_dataframe(
        test_meta,
        pred_macro,
        probs,
        confidence,
        margin,
        entropy,
        macros=macros,
        method_name=display_name,
        text_col=text_col,
        group_col=group_col,
        label_col=label_col,
    )
    metrics_out: Dict[str, Any] = {
        "method": display_name,
        "model": str(model_key),
        "n_source": int(len(X_btp)),
        "n_target": int(len(X_test)),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "accuracy": float("nan"),
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else float("nan"),
        "mean_entropy": float(np.mean(entropy)) if len(entropy) else float("nan"),
    }
    if label_col and label_col in test_meta.columns:
        eval_metrics = evaluate_macro_predictions(
            test_meta[label_col].astype(str).to_numpy(),
            pred_macro,
            probs,
            macros,
        )
        cm = np.asarray(eval_metrics.pop("confusion_matrix"))
        cls_rep = eval_metrics.pop("classification_report")
        metrics_out.update(eval_metrics)
        metrics_out["_confusion_matrix"] = cm
        metrics_out["_classification_report"] = cls_rep
    return pipe, preds, metrics_out


def export_cv_results(
    out_dir: Path,
    fold_rows: Sequence[Mapping[str, Any]],
    cv_summary: pd.DataFrame,
) -> None:
    """Écrit ``cv/cv_per_fold.csv`` et ``cv/cv_summary.csv``."""
    cv_dir = Path(out_dir) / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(fold_rows)).to_csv(cv_dir / "cv_per_fold.csv", index=False)
    cv_summary.to_csv(cv_dir / "cv_summary.csv", index=False)


def export_test_results(
    out_dir: Path,
    preds: pd.DataFrame,
    metrics: Mapping[str, Any],
    *,
    macros: Sequence[str],
) -> Path:
    """Écrit transfer/target_macro_predictions.csv, metrics.json, etc."""
    transfer_dir = Path(out_dir) / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(transfer_dir / "target_macro_predictions.csv", index=False)

    metrics_json = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
    with open(transfer_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    cm = metrics.get("_confusion_matrix")
    if cm is not None:
        pd.DataFrame(np.asarray(cm), index=macros, columns=macros).to_csv(
            transfer_dir / "confusion_matrix.csv"
        )
    cls_rep = metrics.get("_classification_report")
    if cls_rep:
        pd.DataFrame(cls_rep).T.to_csv(transfer_dir / "classification_report.csv", index=True)

    bertopic_cols = ["pred_macro", "confidence"] + [f"prob_{m}" for m in macros]
    group_col = "accident_id" if "accident_id" in preds.columns else None
    base_cols = [c for c in [group_col, "fact_id", "sentence"] if c and c in preds.columns]
    bertopic_df = preds[base_cols + bertopic_cols]
    bertopic_df.to_csv(transfer_dir / "bertopic_input_all.csv", index=False)
    for m in macros:
        bertopic_df[bertopic_df["pred_macro"] == m].to_csv(
            transfer_dir / f"bertopic_input_{m}.csv",
            index=False,
        )
    return transfer_dir


def summarize_all_models_test_metrics(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Tableau comparatif des métriques test pour tous les modèles."""
    rows: list[dict[str, Any]] = []
    for model_key, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": str(model_key),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "mean_confidence": metrics.get("mean_confidence"),
                "mean_entropy": metrics.get("mean_entropy"),
            }
        )
    return pd.DataFrame(rows)


def _write_model_test_artifacts(
    model_dir: Path,
    preds: pd.DataFrame,
    metrics: Mapping[str, Any],
    *,
    macros: Sequence[str],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(model_dir / "target_macro_predictions.csv", index=False)
    metrics_json = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    cm = metrics.get("_confusion_matrix")
    if cm is not None:
        pd.DataFrame(np.asarray(cm), index=macros, columns=macros).to_csv(
            model_dir / "confusion_matrix.csv"
        )
    cls_rep = metrics.get("_classification_report")
    if cls_rep:
        pd.DataFrame(cls_rep).T.to_csv(model_dir / "classification_report.csv", index=True)


def export_all_models_test_results(
    out_dir: Path,
    preds_by_model: Mapping[str, pd.DataFrame],
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    *,
    macros: Sequence[str],
    best_model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Écrit ``transfer/models/<model>/`` + ``transfer/all_models_test_metrics.csv``.

    Duplique aussi le meilleur modèle sous ``transfer/`` (rétrocompatibilité).
    """
    models_root = Path(out_dir) / "transfer" / "models"
    for model_key, preds in preds_by_model.items():
        _write_model_test_artifacts(
            models_root / str(model_key),
            preds,
            metrics_by_model[model_key],
            macros=macros,
        )
    summary = summarize_all_models_test_metrics(metrics_by_model)
    summary.to_csv(Path(out_dir) / "transfer" / "all_models_test_metrics.csv", index=False)
    if best_model and str(best_model) in preds_by_model:
        export_test_results(
            out_dir,
            preds_by_model[str(best_model)],
            metrics_by_model[str(best_model)],
            macros=macros,
        )
    return summary


def evaluate_all_models_on_test(
    model_keys: Sequence[str],
    model_registry: Mapping[str, Mapping[str, Any]],
    X_btp: np.ndarray,
    y_btp: Sequence[str],
    X_test: np.ndarray,
    test_meta: pd.DataFrame,
    *,
    macros: Sequence[str] = MACRO_NAMES,
    seed: int = 42,
    text_col: str = "sentence",
    group_col: str = "accident_id",
    label_col: Optional[str] = "pred_label",
    method_prefix: str = "supervised_macro_baseline",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """Réentraîne chaque modèle sur 100 % BTP et évalue sur le corpus test."""
    preds_by_model: Dict[str, pd.DataFrame] = {}
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for model_key in model_keys:
        spec = model_registry[model_key]
        _, preds, metrics = fit_final_and_predict_test(
            str(model_key),
            X_btp,
            y_btp,
            X_test,
            test_meta,
            macros=macros,
            seed=seed,
            params=spec.get("params"),
            use_scaler=spec.get("use_scaler"),
            method_name=f"{method_prefix}/{model_key}",
            text_col=text_col,
            group_col=group_col,
            label_col=label_col,
        )
        preds_by_model[str(model_key)] = preds
        metrics_by_model[str(model_key)] = metrics
    return preds_by_model, metrics_by_model


def load_cached_test_results_for_model(
    out_dir: Path,
    model_key: str,
    *,
    macros: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Recharge les prédictions test d'un modèle depuis ``transfer/models/<model>/``."""
    model_dir = Path(out_dir) / "transfer" / "models" / str(model_key)
    preds_path = model_dir / "target_macro_predictions.csv"
    metrics_path = model_dir / "metrics.json"
    if not preds_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Artefacts test manquants pour {model_key!r} : {model_dir}")
    preds = pd.read_csv(preds_path)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    cm_path = model_dir / "confusion_matrix.csv"
    if cm_path.is_file():
        metrics["_confusion_matrix"] = pd.read_csv(cm_path, index_col=0).to_numpy()
    rep_path = model_dir / "classification_report.csv"
    if rep_path.is_file():
        rep_df = pd.read_csv(rep_path, index_col=0)
        metrics["_classification_report"] = rep_df.to_dict(orient="index")
    return preds, metrics


def load_cached_all_models_test_results(
    out_dir: Path,
    model_keys: Sequence[str],
    *,
    macros: Sequence[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Recharge tous les modèles + tableau récapitulatif."""
    preds_by_model: Dict[str, pd.DataFrame] = {}
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for model_key in model_keys:
        preds, metrics = load_cached_test_results_for_model(
            out_dir, model_key, macros=macros
        )
        preds_by_model[str(model_key)] = preds
        metrics_by_model[str(model_key)] = metrics
    summary_path = Path(out_dir) / "transfer" / "all_models_test_metrics.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
    else:
        summary = summarize_all_models_test_metrics(metrics_by_model)
    return preds_by_model, metrics_by_model, summary


RUN_MANIFEST_NAME = "run_manifest.json"


def supervised_run_manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / RUN_MANIFEST_NAME


def save_supervised_run_manifest(
    out_dir: Path,
    *,
    best_model: str,
    selection_metric: str,
    seed: int,
    n_folds: int,
    test_corpus: str,
    model_keys: Optional[Sequence[str]] = None,
) -> Path:
    """Métadonnées du run (meilleur modèle, pour rechargement ``RESTIMATE=False``)."""
    path = supervised_run_manifest_path(out_dir)
    payload = {
        "best_model": str(best_model),
        "selection_metric": str(selection_metric),
        "seed": int(seed),
        "n_folds": int(n_folds),
        "test_corpus": str(test_corpus),
    }
    if model_keys is not None:
        payload["model_keys"] = [str(k) for k in model_keys]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_supervised_run_manifest(out_dir: Path) -> Dict[str, Any]:
    path = supervised_run_manifest_path(out_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Manifeste run absent : {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def supervised_ml_artifacts_exist(out_dir: Path) -> bool:
    root = Path(out_dir)
    return all(
        (root / rel).is_file()
        for rel in (
            "cv/cv_summary.csv",
            "cv/cv_per_fold.csv",
            "transfer/target_macro_predictions.csv",
            "transfer/metrics.json",
            RUN_MANIFEST_NAME,
        )
    )


def supervised_bertopic_artifacts_exist(out_dir: Path) -> bool:
    root = Path(out_dir)
    return (root / "topics_bertopic" / "assignments.csv").is_file()


def load_cached_cv_results(out_dir: Path) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Recharge CV depuis ``cv/``."""
    root = Path(out_dir)
    per_fold_path = root / "cv" / "cv_per_fold.csv"
    summary_path = root / "cv" / "cv_summary.csv"
    if not per_fold_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Artefacts CV manquants sous {root / 'cv'}")
    per_fold = pd.read_csv(per_fold_path).to_dict("records")
    summary = pd.read_csv(summary_path)
    return per_fold, summary


def load_cached_fold_rows_for_model(
    out_dir: Path,
    model_key: str,
) -> List[Dict[str, Any]]:
    per_fold, _ = load_cached_cv_results(out_dir)
    return [r for r in per_fold if str(r.get("model")) == str(model_key)]


def load_cached_test_results(
    out_dir: Path,
    *,
    macros: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Recharge prédictions test + métriques (avec CM / report si présents)."""
    transfer = Path(out_dir) / "transfer"
    preds_path = transfer / "target_macro_predictions.csv"
    metrics_path = transfer / "metrics.json"
    if not preds_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Artefacts transfer manquants sous {transfer}")
    preds = pd.read_csv(preds_path)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    cm_path = transfer / "confusion_matrix.csv"
    if cm_path.is_file():
        metrics["_confusion_matrix"] = pd.read_csv(cm_path, index_col=0).to_numpy()
    rep_path = transfer / "classification_report.csv"
    if rep_path.is_file():
        rep_df = pd.read_csv(rep_path, index_col=0)
        metrics["_classification_report"] = rep_df.to_dict(orient="index")
    return preds, metrics


def require_supervised_cache(out_dir: Path, *, include_bertopic: bool = True) -> None:
    """Lève si le cache disque est incomplet."""
    if not supervised_ml_artifacts_exist(out_dir):
        raise FileNotFoundError(
            f"Cache ML incomplet sous {out_dir}. Lancez avec RESTIMATE=True."
        )
    if include_bertopic and not supervised_bertopic_artifacts_exist(out_dir):
        raise FileNotFoundError(
            f"BERTopic cache absent sous {out_dir / 'topics_bertopic'}. "
            "Lancez avec RESTIMATE=True."
        )


def run_supervised_bertopic_phase(
    out_dir: Path,
    *,
    test_meta: pd.DataFrame,
    preds: pd.DataFrame,
    X_test: np.ndarray,
    macros: Sequence[str],
    bertopic_cfg: Mapping[str, Any],
    topics_export_cfg: Mapping[str, Any],
    text_col: str,
    corpus_id: str,
    method_name: str,
    anchor: Optional[Path] = None,
    topic_judge_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Lance BERTopic intra-macro sur les prédictions supervisées."""
    if not bertopic_cfg or bertopic_cfg.get("enabled", True) is False:
        return {}
    bertopic_cfg_dict = dict(bertopic_cfg)
    judge_cfg = dict(topic_judge_cfg or bertopic_cfg_dict.pop("topic_judge", None) or {})
    gating = _build_gating_from_predictions(preds, macros)
    meta_t = test_meta.copy()
    meta_t["m_hat"] = preds["pred_macro"].astype(str).to_numpy()
    return run_bertopic_phase(
        out=Path(out_dir),
        meta_t=meta_t,
        gating_adapted=gating,
        h_t=np.asarray(X_test, dtype=np.float64),
        h_t_adapted=np.asarray(X_test, dtype=np.float64),
        method_name=method_name,
        bertopic_cfg=bertopic_cfg_dict,
        topics_export_cfg=dict(topics_export_cfg),
        text_col_t=text_col,
        repo_anchor=anchor or TEXT_ROOT,
        corpus_id=corpus_id,
        topic_embedding_mode=None,
        topic_alpha=None,
        run_bertopic_grid=False,
        grid_macros=None,
        skip_compression_diagnostics=True,
        topic_judge_cfg=judge_cfg or None,
    )


def load_supervised_config(config_path: str | Path) -> Dict[str, Any]:
    """Charge le YAML baseline supervisée (+ fusion config BERTopic partagée)."""
    anchor = Path(__file__).resolve().parents[1]
    cfg = load_yaml(Path(config_path))
    return enrich_run_config_bertopic(cfg, anchor=anchor)


def run_supervised_baseline_from_config(config_path: str | Path) -> Dict[str, Any]:
    """Pipeline complet (CV → sélection → test → BERTopic) depuis un YAML."""
    cfg_path = Path(config_path)
    cfg = load_supervised_config(cfg_path)
    anchor = Path(__file__).resolve().parents[1]
    data = load_supervised_datasets(cfg, anchor=anchor)

    n_folds = int(cfg.get("n_folds", 5))
    seed = int(cfg.get("seed", 42))
    selection_metric = str(cfg.get("selection_metric", DEFAULT_SELECTION_METRIC))
    model_registry = merge_model_registry(cfg.get("models"))
    model_keys = list(model_registry.keys())

    fold_rows = run_all_models_group_kfold_cv(
        model_keys,
        data["X_btp"],
        data["y_btp"],
        data["groups_btp"],
        model_registry=model_registry,
        macros=data["macros"],
        n_folds=n_folds,
        seed=seed,
    )
    cv_summary = aggregate_cv_metrics(fold_rows)
    best_model = select_best_model(cv_summary, selection_metric=selection_metric)

    corpus_id = str(data["corpus_id"])
    out_dir = supervised_baseline_output_dir(corpus_id, anchor=anchor)
    export_cv_results(out_dir, fold_rows, cv_summary)

    method_name = str(cfg.get("method_name", "supervised_macro_baseline"))
    preds_by_model, metrics_by_model = evaluate_all_models_on_test(
        model_keys,
        model_registry,
        data["X_btp"],
        data["y_btp"],
        data["X_test"],
        data["test_meta"],
        macros=data["macros"],
        seed=seed,
        text_col=data["target_text_col"],
        group_col=data["target_group_col"],
        label_col=data["target_label_col"],
        method_prefix=method_name,
    )
    test_summary = export_all_models_test_results(
        out_dir,
        preds_by_model,
        metrics_by_model,
        macros=data["macros"],
        best_model=best_model,
    )
    preds = preds_by_model[best_model]
    metrics = metrics_by_model[best_model]
    save_supervised_run_manifest(
        out_dir,
        best_model=best_model,
        selection_metric=selection_metric,
        seed=seed,
        n_folds=n_folds,
        test_corpus=corpus_id,
        model_keys=model_keys,
    )

    bertopic_summary = run_supervised_bertopic_phase(
        out_dir,
        test_meta=data["test_meta"],
        preds=preds,
        X_test=data["X_test"],
        macros=data["macros"],
        bertopic_cfg=dict(cfg.get("bertopic") or {}),
        topics_export_cfg=dict(cfg.get("topics_export") or {}),
        topic_judge_cfg=dict(cfg.get("topic_judge") or {}),
        text_col=data["target_text_col"],
        corpus_id=corpus_id,
        method_name=f"{method_name}/{best_model}",
        anchor=anchor,
    )

    return {
        "output_dir": str(out_dir),
        "best_model": best_model,
        "cv_summary": cv_summary,
        "fold_rows": fold_rows,
        "metrics": metrics,
        "test_summary": test_summary,
        "bertopic_summary": bertopic_summary,
    }
