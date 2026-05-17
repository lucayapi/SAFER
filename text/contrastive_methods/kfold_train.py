"""K-fold groupé pour entraînement contrastif (train simple et tuning)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from contrastive_methods.config import (
    ContrastiveConfig,
    load_contrastive_config_from_dict,
    merge_config_dict,
)
from contrastive_methods.data import get_group_kfold_splits, prepare_text_dataset
from contrastive_methods.eval_corpus import evaluate_btp_and_test
from contrastive_methods.results import TrainingResult
from safer_core.kfold_eval import aggregate_fold_rows, save_kfold_tables
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


def run_kfold_loop(
    cfg: ContrastiveConfig,
    runner: Callable[[ContrastiveConfig], TrainingResult],
    *,
    fold_dir_fn: Optional[Callable[[int], str]] = None,
    log_prefix: str = "kfold",
    save_tables: bool = True,
    metrics_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Exécute K folds (validation uniquement) → agrégat μ±σ."""
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    root = Path(layout["root"])
    if metrics_dir is None:
        metrics_dir = Path(layout["metrics"])

    dataset = prepare_text_dataset(cfg)
    splits = get_group_kfold_splits(dataset, cfg)
    fold_rows: List[Dict[str, Any]] = []

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

        print(f"[{log_prefix}] fold {fold_id} → {fold_out}", flush=True)
        result = runner(fold_cfg)
        row: Dict[str, Any] = {"fold_id": fold_id, **(result.val_geometry or {})}
        row["fold_selection_score"] = result.best_delta_macro_pct
        fold_rows.append(row)

    agg = aggregate_fold_rows(fold_rows)
    if save_tables:
        save_kfold_tables(fold_rows, metrics_dir)
    return fold_rows, agg


def run_contrastive_kfold(cfg: ContrastiveConfig) -> Dict[str, Any]:
    """Train simple — étape 1 : K-fold validation (μ±σ), sans éval test."""
    runner = get_contrastive_runner(cfg.method_name)
    _, agg = run_kfold_loop(cfg, runner, log_prefix=cfg.method_name)
    layout = layout_method_output(cfg.method_name, cfg.resolved_output_dir)
    metrics_dir = Path(layout["metrics"])
    print(
        f"[{cfg.method_name}] K-fold val → {metrics_dir / 'kfold_summary.csv'} | "
        f"mean_delta_macro_pct={agg.get('mean_delta_macro_pct', float('nan')):.2f} "
        f"± {agg.get('std_delta_macro_pct', float('nan')):.2f}",
        flush=True,
    )
    return agg


def run_contrastive_final_fit_and_eval(cfg: ContrastiveConfig) -> None:
    """Train simple — étape 2 : fit 100 % BTP puis métriques BTP + test."""
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
    result = runner(cfg_final)
    ckpt = result.output_root / "checkpoints" / "best_model"
    if not ckpt.exists():
        print(f"[{cfg.method_name}] Checkpoint final absent : {ckpt}", flush=True)
        return
    paths = evaluate_btp_and_test(cfg_final, ckpt, result.output_root)
    print(f"[{cfg.method_name}] Fit final — embeddings : {result.embeddings_path}", flush=True)
    if paths.get("btp"):
        print(f"[{cfg.method_name}] Métriques BTP : {paths['btp']}", flush=True)
    if paths.get("test"):
        print(f"[{cfg.method_name}] Métriques test : {paths['test']}", flush=True)


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
        **{f"mean_{k}": v for k, v in agg.items() if k.startswith("mean_")},
        **{f"std_{k}": v for k, v in agg.items() if k.startswith("std_")},
        **{k.replace(".", "_"): v for k, v in overrides.items()},
    }
