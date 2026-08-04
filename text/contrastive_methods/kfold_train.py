"""K-fold groupé pour entraînement contrastif (train simple et tuning)."""

from __future__ import annotations

import dataclasses
import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from contrastive_methods.config import (
    ContrastiveConfig,
    load_contrastive_config_from_dict,
    merge_config_dict,
)
from contrastive_methods.data import get_group_kfold_splits, prepare_text_dataset, train_val_metadata
from contrastive_methods.eval_corpus import run_final_classification_eval
from contrastive_methods.post_eval import (
    CV_CLASSIFICATION_METRIC_KEYS,
    run_post_eval_grid_on_fold,
    run_post_eval_on_fold,
)
from contrastive_methods.results import TrainingResult
from contrastive_methods.hf_training_common import get_device
from safer_core.classification_eval import DEFAULT_SELECTION_METRIC
from safer_core.kfold_eval import (
    KFOLD_AGGREGATE_METRIC_KEYS,
    aggregate_fold_rows,
    record_final_fit_wall_time,
    save_kfold_tables,
)
from safer_core.paths import layout_method_output


def get_contrastive_runner(method_name: str):
    dispatch = {
        "batch_triplet": "contrastive_methods.training_triplet:run_batch_triplet",
        "softtriple": "contrastive_methods.training_softtriple:run_softtriple",
        "supcon": "contrastive_methods.training_supcon:run_supcon",
    }
    target = dispatch.get(method_name)
    if not target:
        raise ValueError(f"Méthode inconnue : {method_name}")
    mod_name, attr = target.split(":")
    import importlib

    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _fold_output_path(base: Path, fold_id: int) -> Path:
    return base / "folds" / f"fold_{fold_id}"


def _inner_early_stopping_split(
    dataset,
    outer_train_idx: np.ndarray,
    cfg: ContrastiveConfig,
    fold_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split only the outer training partition for early-stopping monitoring."""
    outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
    groups = np.asarray(dataset.get_groups())
    metadata = dataset.get_metadata_df()
    if groups.ndim != 1 or groups.shape[0] <= int(outer_train_idx.max(initial=-1)):
        return outer_train_idx, np.array([], dtype=np.int64)
    try:
        labels = np.asarray(metadata.iloc[outer_train_idx]["label_id"])
    except (AttributeError, KeyError, IndexError, TypeError):
        return outer_train_idx, np.array([], dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(outer_train_idx):
        return outer_train_idx, np.array([], dtype=np.int64)
    unique_labels = set(labels.tolist())
    if len(unique_labels) < 2 or len(outer_train_idx) < 4:
        return outer_train_idx, np.array([], dtype=np.int64)

    ratio = min(max(float(cfg.early_stopping_inner_val_ratio), 0.01), 0.5)
    for attempt in range(20):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=ratio,
            random_state=int(cfg.seed) + fold_id * 100 + attempt,
        )
        local_train, local_val = next(
            splitter.split(outer_train_idx, labels, groups=groups[outer_train_idx])
        )
        inner_train_idx = outer_train_idx[local_train]
        inner_val_idx = outer_train_idx[local_val]
        val_labels = set(np.asarray(metadata.iloc[inner_val_idx]["label_id"]).tolist())
        train_labels = set(np.asarray(metadata.iloc[inner_train_idx]["label_id"]).tolist())
        val_counts = metadata.iloc[inner_val_idx]["label_id"].value_counts()
        if len(train_labels) >= 2 and len(val_labels) >= 2 and (val_counts >= 2).all():
            return inner_train_idx, inner_val_idx
    return outer_train_idx, np.array([], dtype=np.int64)


def _safe_group_count(dataset, indices: np.ndarray) -> Optional[int]:
    groups = np.asarray(dataset.get_groups())
    if groups.ndim != 1 or len(indices) == 0:
        return 0 if len(indices) == 0 else None
    try:
        return len(set(groups[np.asarray(indices, dtype=np.int64)].tolist()))
    except (IndexError, TypeError):
        return None


def run_kfold_loop(
    cfg: ContrastiveConfig,
    runner: Callable[[ContrastiveConfig], TrainingResult],
    *,
    fold_dir_fn: Optional[Callable[[int], str]] = None,
    log_prefix: str = "kfold",
    save_tables: bool = True,
    metrics_dir: Optional[Path] = None,
    post_eval_grid: Optional[List[Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Exécute K folds (validation uniquement) → agrégat μ±σ classification."""
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    if metrics_dir is None:
        metrics_dir = Path(layout["metrics"])
    cv_dir = root / "cv"

    dataset = prepare_text_dataset(cfg)
    splits = get_group_kfold_splits(dataset, cfg)
    fold_rows: List[Dict[str, Any]] = []
    device = get_device()

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        if fold_dir_fn is not None:
            fold_out = fold_dir_fn(fold_id)
        else:
            fold_out = str(_fold_output_path(root, fold_id))

        fold_cfg = dataclasses.replace(
            cfg,
            output_dir=fold_out,
            final_fit_full_data=False,
        )
        fold_cfg.extra = dict(cfg.extra)
        fold_cfg.extra["fold_train_idx"] = train_idx
        fold_cfg.extra["fold_val_idx"] = val_idx
        inner_train_idx, inner_val_idx = _inner_early_stopping_split(
            dataset, train_idx, cfg, fold_id
        )
        fold_cfg.extra["early_stopping_train_idx"] = inner_train_idx
        fold_cfg.extra["early_stopping_val_idx"] = inner_val_idx

        print(f"[{log_prefix}] fold {fold_id} → {fold_out}", flush=True)
        result = runner(fold_cfg)
        train_df, val_df = train_val_metadata(dataset, train_idx, val_idx)
        row: Dict[str, Any] = {
            "fold_id": fold_id,
            "best_train_loss": result.best_train_loss,
            "train_wall_time_sec": float(result.train_wall_time_sec),
            "best_epoch": result.best_epoch,
            "epochs_ran": result.epochs_ran,
            "max_epochs": cfg.epochs,
            "early_stopped": bool(result.epochs_ran is not None and result.epochs_ran < cfg.epochs),
            "n_outer_train": len(train_idx),
            "n_outer_val": len(val_idx),
            "n_inner_train": len(inner_train_idx),
            "n_inner_val": len(inner_val_idx),
            "n_outer_train_groups": _safe_group_count(dataset, train_idx),
            "n_outer_val_groups": _safe_group_count(dataset, val_idx),
            "n_inner_train_groups": _safe_group_count(dataset, inner_train_idx),
            "n_inner_val_groups": _safe_group_count(dataset, inner_val_idx),
        }
        ckpt = Path(fold_cfg.output_dir) / "checkpoints" / "best_model"
        if cfg.post_eval_enabled and ckpt.is_dir():
            try:
                if post_eval_grid:
                    grid_metrics = run_post_eval_grid_on_fold(
                        cfg,
                        ckpt,
                        train_df,
                        val_df,
                        cfg.text_col,
                        device,
                        post_eval_grid,
                    )
                    for grid_id, metrics in grid_metrics.items():
                        for metric, value in metrics.items():
                            row[f"lr_{grid_id}_val_{metric}"] = value
                else:
                    row.update(
                        run_post_eval_on_fold(cfg, ckpt, train_df, val_df, cfg.text_col, device)
                    )
            except Exception as exc:
                print(f"[{log_prefix}] post_eval fold {fold_id} ignoré : {exc}", flush=True)
        fold_rows.append(row)
        del result
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    selection = f"val_{DEFAULT_SELECTION_METRIC}"
    agg = aggregate_fold_rows(
        fold_rows,
        metric_keys=KFOLD_AGGREGATE_METRIC_KEYS,
        selection_metric=selection,
    )
    if save_tables:
        save_kfold_tables(
            fold_rows,
            metrics_dir,
            selection_metric=selection,
            cv_dir=cv_dir,
        )
    return fold_rows, agg


def run_contrastive_kfold(cfg: ContrastiveConfig) -> Dict[str, Any]:
    """Train simple — étape 1 : K-fold validation (μ±σ), sans éval test."""
    runner = get_contrastive_runner(cfg.method_name)
    _, agg = run_kfold_loop(cfg, runner, log_prefix=cfg.method_name)
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    metrics_dir = Path(layout["metrics"])
    print(
        f"[{cfg.method_name}] K-fold val → {metrics_dir / 'kfold_summary.csv'} | "
        f"mean_val_balanced_accuracy={agg.get('mean_val_balanced_accuracy', float('nan')):.2f} "
        f"± {agg.get('std_val_balanced_accuracy', float('nan')):.2f}",
        flush=True,
    )
    return agg


def run_contrastive_final_fit_and_eval(cfg: ContrastiveConfig) -> None:
    """Train simple — étape 2 : fit 100 % BTP puis classification multi-corpus."""
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    runner = get_contrastive_runner(cfg.method_name)

    cfg_final = dataclasses.replace(
        cfg,
        output_dir=str(root),
        final_fit_full_data=True,
    )
    cfg_final.extra = dict(cfg.extra)
    cfg_final.extra.pop("fold_train_idx", None)
    cfg_final.extra.pop("fold_val_idx", None)

    print(f"[{cfg.method_name}] Réentraînement final 100 % BTP…", flush=True)
    t0 = time.perf_counter()
    result = runner(cfg_final)
    metrics_dir = Path(layout_method_output(cfg.method_name, cfg.resolved_output_dir)["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    record_final_fit_wall_time(metrics_dir, time.perf_counter() - t0)
    ckpt = result.output_root / "checkpoints" / "best_model"
    if not ckpt.exists():
        print(f"[{cfg.method_name}] Checkpoint final absent : {ckpt}", flush=True)
        return
    paths = run_final_classification_eval(cfg_final, ckpt, result.output_root)
    print(f"[{cfg.method_name}] Fit final — embeddings : {result.embeddings_path}", flush=True)
    if paths.get("btp"):
        print(f"[{cfg.method_name}] Métriques BTP : {paths['btp']}", flush=True)
    if paths.get("cross_domain"):
        print(f"[{cfg.method_name}] Cross-domain : {paths['cross_domain']}", flush=True)


def run_tuning_combo_kfold(
    method_name: str,
    runner,
    cfg_base: Dict[str, Any],
    overrides: Dict[str, Any],
    combo_id: str,
    tuning_output: str,
    n_folds: int,
    seed: int,
    selection_metric: str,
) -> Dict[str, Any]:
    """K-fold pour une combinaison de la grille tuning (validation uniquement)."""
    merged = merge_config_dict(cfg_base, overrides)
    merged_prep = {**merged, "method_name": method_name, "seed": seed, "n_folds": n_folds}
    cfg = load_contrastive_config_from_dict(method_name, merged_prep)

    tuning_root = Path(tuning_output)

    def fold_dir_fn(fold_id: int) -> str:
        return str(tuning_root / "combos" / combo_id / f"fold_{fold_id}")

    _, agg = run_kfold_loop(
        cfg,
        runner,
        fold_dir_fn=fold_dir_fn,
        log_prefix=f"tuning/{combo_id}",
        save_tables=False,
    )
    return {
        "combo_id": combo_id,
        "selection_metric": selection_metric,
        "selection_score": agg.get("selection_score", float("nan")),
        **{k: agg.get(k) for k in agg if k.startswith("mean_") or k.startswith("std_")},
    }
