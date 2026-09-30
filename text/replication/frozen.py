"""Source-only CV and OOD evaluation for frozen exported embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from safer_core.classification_eval import (build_and_save_predictions, evaluate_classifier_on_embeddings, fit_logistic_on_embeddings, per_role_metrics_from_predictions, save_classification_outputs)
from safer_core.data_loading import load_metadata_with_embeddings
from safer_core.embedding_paths import embedding_export_path
from safer_core.io import ensure_dir
from safer_core.kfold_eval import group_kfold_splits
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.dataset_text_embeddings import LABEL2ID


def _load_corpus(corpus: str, spec: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    registry = resolve_test_corpus(corpus, require_files=True, require_emb_csv=False)
    emb = embedding_export_path(corpus, backbone_name=str(spec["backbone_name"]), backbone_id=spec.get("backbone_id"), output_root=str(spec.get("embeddings_root", "embeddings")))
    if not emb.is_file():
        raise FileNotFoundError(f"Frozen embeddings missing for {corpus}: {emb}")
    meta, dims = load_metadata_with_embeddings(registry.data_csv, emb, label_col=str(spec.get("label_col", "pred_label")), pred_ok_col=str(spec.get("pred_ok_col", "pred_ok")), group_col=str(spec.get("group_col", "accident_id")))
    return meta, meta[dims].to_numpy(dtype=np.float64)


def _labels(meta: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    names = meta[label_col].astype(str).to_numpy()
    unknown = sorted(set(names) - set(LABEL2ID))
    if unknown:
        raise ValueError(f"Unknown role labels: {unknown}")
    return names, np.asarray([LABEL2ID[name] for name in names], dtype=np.int64)


def run_frozen_replication(spec: Mapping[str, Any], *, model_id: str, run_dir: Path, training_seed: int, split_seed: int, n_folds: int, source_corpus: str, target_corpora: Sequence[str]) -> dict[str, Any]:
    """Select logistic regularisation only on source grouped folds, then predict targets."""
    source_meta, source_X = _load_corpus(source_corpus, spec)
    label_col, group_col = str(spec.get("label_col", "pred_label")), str(spec.get("group_col", "accident_id"))
    source_y, source_y_int = _labels(source_meta, label_col)
    groups = source_meta[group_col].astype(str).to_numpy()
    c_grid = [float(c) for c in spec.get("classifier_c_grid", [0.001, 0.01, 0.1, 1.0])]
    rows: list[dict[str, Any]] = []
    for c in c_grid:
        for fold, (tr, va) in enumerate(group_kfold_splits(groups, int(n_folds), int(split_seed))):
            if set(groups[tr]) & set(groups[va]):
                raise RuntimeError(f"accident_id leakage in frozen fold {fold}")
            t_fit = time.perf_counter()
            pipe = fit_logistic_on_embeddings(source_X[tr], source_y_int[tr], class_weight=spec.get("class_weight"), oversampling=bool(spec.get("oversampling", False)), seed=training_seed + fold, classifier_overrides={"C": c})
            rows.append({"C": c, "fold": fold, "fit_wall_time_sec": time.perf_counter() - t_fit, **evaluate_classifier_on_embeddings(pipe, source_X[va], source_y[va])})
    cv = pd.DataFrame(rows)
    aggregate = cv.groupby("C", as_index=False).agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), std_balanced_accuracy=("balanced_accuracy", "std"), mean_accuracy=("accuracy", "mean"), mean_macro_f1=("macro_f1", "mean")).sort_values(["mean_balanced_accuracy", "C"], ascending=[False, True])
    selected_c = float(aggregate.iloc[0]["C"])
    cv_dir = ensure_dir(run_dir / "cv")
    cv.to_csv(cv_dir / "cv_per_fold.csv", index=False)
    summary = aggregate.iloc[[0]].copy()
    summary.insert(0, "model", model_id); summary.insert(1, "source_corpus", source_corpus)
    summary.insert(2, "training_seed", training_seed); summary.insert(3, "split_seed", split_seed); summary.insert(4, "n_folds", n_folds)
    summary["mean_cv_fit_wall_time_sec"] = float(cv[cv["C"].eq(selected_c)]["fit_wall_time_sec"].mean())
    summary.to_csv(cv_dir / "cv_summary.csv", index=False)
    t_final_fit = time.perf_counter()
    pipe = fit_logistic_on_embeddings(source_X, source_y_int, class_weight=spec.get("class_weight"), oversampling=bool(spec.get("oversampling", False)), seed=training_seed, classifier_overrides={"C": selected_c})
    final_fit_wall_time_sec = time.perf_counter() - t_final_fit
    metrics_by_corpus: dict[str, dict[str, Any]] = {}; per_role: dict[str, pd.DataFrame] = {}
    for corpus in [source_corpus, *target_corpora]:
        meta, X = (source_meta, source_X) if corpus == source_corpus else _load_corpus(corpus, spec)
        y, _ = _labels(meta, label_col)
        metrics, details = evaluate_classifier_on_embeddings(pipe, X, y, return_details=True)
        predictions, _ = build_and_save_predictions(meta, details, run_dir, corpus, method_name="frozen", group_col=group_col, label_col=label_col)
        metrics_by_corpus[corpus], per_role[corpus] = metrics, per_role_metrics_from_predictions(predictions)
    save_classification_outputs(run_dir, method_name="frozen", metrics_by_corpus=metrics_by_corpus, cv_summary=summary, classifier="logistic_regression", source_corpus=source_corpus, per_role_by_corpus=per_role)
    (run_dir / "best_logistic_params.json").write_text(json.dumps({"C": selected_c, "class_weight": spec.get("class_weight")}, indent=2), encoding="utf-8")
    return {"selected_C": selected_c, "final_fit_train_wall_time_sec": final_fit_wall_time_sec, "checkpoint_dir": None}
