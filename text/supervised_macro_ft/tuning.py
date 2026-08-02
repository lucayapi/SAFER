"""Grille hyperparamètres supervised_macro_ft (CV GroupKFold, balanced accuracy)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_methods.config import merge_config_dict
from safer_core.io import ensure_dir, load_yaml
from safer_core.paths import TEXT_ROOT
from supervised_macro_ft.train_runner import (
    run_supervised_macro_ft_cv,
    run_supervised_macro_ft_training,
)

MACRO_FT_GRID_PREFIXES = ("training.", "model.")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def validate_macro_ft_grid_keys(grid: Dict[str, Any]) -> None:
    for key in grid:
        if key == "seed" or any(key.startswith(prefix) for prefix in MACRO_FT_GRID_PREFIXES):
            continue
        raise ValueError(
            f"Clé grille supervised_macro_ft invalide : {key!r}. "
            f"Préfixes autorisés : {', '.join(MACRO_FT_GRID_PREFIXES)}"
        )


def expand_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values_list = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values_list)]


def is_valid_macro_ft_tuning_model_cfg(model_cfg: Dict[str, Any]) -> bool:
    """Exclut les combinaisons hors scope article (FT only, class_weight, MLP/null)."""
    from supervised_macro_ft.backbone_scaler import should_standardize_backbone
    from supervised_macro_ft.class_balance import resolve_train_balance
    from supervised_macro_ft.model import validate_macro_ft_projection

    model = dict(model_cfg or {})
    try:
        validate_macro_ft_projection(model.get("projection", "mlp_sklearn"))
        resolve_train_balance(model)
    except ValueError:
        return False

    if not bool(model.get("backbone_trainable", False)):
        return False

    train_last_n = model.get("train_last_n_layers")
    if train_last_n is not None:
        try:
            n = int(train_last_n)
        except (TypeError, ValueError):
            return False
        if n < 1:
            return False

    if should_standardize_backbone(model):
        return False

    if bool(model.get("cache_backbone_embeddings", False)):
        return False

    projection = validate_macro_ft_projection(model.get("projection", "mlp_sklearn"))
    if projection not in (None, "mlp_sklearn"):
        return False
    if projection == "mlp_sklearn":
        hiddim = model.get("hiddim")
        if hiddim is not None and int(hiddim) != 128:
            return False

    return True


def encoder_scope_label(train_last_n_layers: Any) -> str:
    """Libellé article pour le scope d'update encodeur."""
    if train_last_n_layers is None:
        return "Full encoder"
    try:
        n = int(train_last_n_layers)
    except (TypeError, ValueError):
        return str(train_last_n_layers)
    if n == 1:
        return "Last 1 layer"
    return f"Last {n} layers"


def projector_label(projection: Any) -> str:
    """Yes = mlp_sklearn ; No = null/none."""
    from supervised_macro_ft.model import validate_macro_ft_projection

    proj = validate_macro_ft_projection(projection)
    return "Yes" if proj == "mlp_sklearn" else "No"


def filter_macro_ft_tuning_combos(
    combos: List[Dict[str, Any]],
    base_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Filtre la grille après fusion avec la config de base."""
    kept: List[Dict[str, Any]] = []
    for overrides in combos:
        merged = _merge_overrides(base_cfg, overrides)
        if is_valid_macro_ft_tuning_model_cfg(dict(merged.get("model") or {})):
            kept.append(overrides)
    return kept


def apply_full_encoder_training_overrides(
    overrides: Dict[str, Any],
    base_cfg: Dict[str, Any],
    full_encoder_training_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Applique les réglages de stabilité seulement aux variantes full encoder."""
    if not full_encoder_training_overrides:
        return dict(overrides)
    if any(not key.startswith("training.") for key in full_encoder_training_overrides):
        raise ValueError("full_encoder_training_overrides accepte uniquement des clés training.*")

    merged = _merge_overrides(base_cfg, overrides)
    model_cfg = dict(merged.get("model") or {})
    if not bool(model_cfg.get("backbone_trainable", False)):
        return dict(overrides)
    if model_cfg.get("train_last_n_layers") is not None:
        return dict(overrides)
    return {**overrides, **full_encoder_training_overrides}


def _combo_id(overrides: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(overrides.keys()):
        val = overrides[key]
        short = key.split(".")[-1]
        if isinstance(val, float):
            parts.append(f"{short}{val:.0e}".replace("+", ""))
        else:
            parts.append(f"{short}{val}")
    readable = "_".join(parts)[:100] if parts else "default"
    digest = hashlib.sha1(
        json.dumps(overrides, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    return f"{readable}_{digest}"


def _merge_overrides(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return merge_config_dict(base_cfg, overrides)


def _mean_balanced_accuracy_column(selection_metric: str) -> str:
    metric = str(selection_metric).strip().lower()
    if metric in ("balanced_accuracy", "mean_balanced_accuracy"):
        return "mean_balanced_accuracy"
    if metric.startswith("mean_"):
        return metric
    return f"mean_{metric}"


def _run_combo_cv(
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    *,
    combo_output_dir: Path,
    backbone_hidden,
    shared_cache_dir: Path,
    n_folds: int,
    seed: int,
    selection_metric: str,
) -> Dict[str, Any]:
    merged = _merge_overrides(base_cfg, overrides)
    train_section = dict(merged.get("training") or {})
    train_section["n_folds"] = n_folds
    train_section["seed"] = seed
    merged = {**merged, "training": train_section}
    row = run_supervised_macro_ft_cv(
        merged,
        combo_output_dir=combo_output_dir,
        backbone_hidden=backbone_hidden,
        shared_cache_dir=shared_cache_dir,
    )
    col = _mean_balanced_accuracy_column(selection_metric)
    row["selection_score"] = float(row.get(col, row.get("selection_score", float("nan"))))
    return row


def _load_ba_from_metrics_csv(path: Path) -> float:
    if not path.is_file():
        return float("nan")
    df = pd.read_csv(path)
    if "balanced_accuracy" in df.columns:
        return float(df["balanced_accuracy"].iloc[0])
    if "metric" in df.columns and "value" in df.columns:
        hit = df.loc[df["metric"].astype(str) == "balanced_accuracy"]
        if not hit.empty:
            return float(hit.iloc[0]["value"])
    return float("nan")


def _find_ood_ba(combo_dir: Path, corpus_id: str) -> float:
    metrics_dir = Path(combo_dir) / "metrics"
    candidates = [
        metrics_dir / f"metrics_classification_test_{corpus_id}.csv",
        metrics_dir / f"metrics_classification_{corpus_id}.csv",
        metrics_dir / f"{corpus_id}_metrics.csv",
    ]
    for path in candidates:
        ba = _load_ba_from_metrics_csv(path)
        if ba == ba:
            return ba
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob(f"*{corpus_id}*.csv")):
            ba = _load_ba_from_metrics_csv(path)
            if ba == ba:
                return ba
    return float("nan")


def build_variants_results_summary(
    combo_rows: List[Dict[str, Any]],
    *,
    test_corpora: List[str],
) -> pd.DataFrame:
    """Tableau article : encoder scope × projector × CV ± OOD."""
    rows: List[Dict[str, Any]] = []
    for row in combo_rows:
        combo_dir = Path(str(row.get("combo_output_dir") or ""))
        ood_vals: List[float] = []
        out: Dict[str, Any] = {
            "encoder_scope": row.get("encoder_scope"),
            "projector": row.get("projector"),
            "combo_id": row.get("combo_id"),
            "cv_ba_mean": float(row.get("mean_balanced_accuracy", row.get("selection_score", float("nan")))),
            "cv_ba_std": float(row.get("std_balanced_accuracy", float("nan"))),
            "train_last_n_layers": row.get("model_train_last_n_layers", row.get("train_last_n_layers")),
            "projection": row.get("model_projection", row.get("projection")),
            "combo_dir": str(combo_dir),
        }
        for corpus_id in test_corpora:
            ba = float(row.get(f"ba_ood_{corpus_id}", float("nan")))
            if ba != ba:
                ba = _find_ood_ba(combo_dir, corpus_id)
            out[f"ba_ood_{corpus_id}"] = ba
            if ba == ba:
                ood_vals.append(ba)
        out["ba_ood_avg"] = float(sum(ood_vals) / len(ood_vals)) if ood_vals else float("nan")
        out["ba_ood_worst"] = float(min(ood_vals)) if ood_vals else float("nan")
        rows.append(out)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    scope_order = {
        "Last 1 layer": 0,
        "Last 2 layers": 1,
        "Last 3 layers": 2,
        "Full encoder": 3,
    }
    proj_order = {"Yes": 0, "No": 1}
    summary["_scope_ord"] = summary["encoder_scope"].map(lambda s: scope_order.get(str(s), 99))
    summary["_proj_ord"] = summary["projector"].map(lambda s: proj_order.get(str(s), 99))
    summary = summary.sort_values(["_scope_ord", "_proj_ord"], kind="mergesort").drop(
        columns=["_scope_ord", "_proj_ord"]
    )
    return summary.reset_index(drop=True)


def run_supervised_macro_ft_tuning(argv: Optional[List[str]] = None) -> int:
    """Campagne de variantes FT : CV + fit + preds pour chaque combo (pas de best-only)."""
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Campagne variantes supervised_macro_ft (tableau article)."
    )
    parser.add_argument(
        "--grid-config",
        type=str,
        default="configs/tuning/supervised_macro_ft_grid.yaml",
    )
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument(
        "--skip-final-fit",
        action="store_true",
        help="CV seule (pas de fit final ni preds OOD) — debug uniquement",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed global")
    tune_args = parser.parse_args(argv)

    spec = load_yaml(TEXT_ROOT / tune_args.grid_config)
    if os.environ.get("TEST_CORPORA"):
        corpora = [c.strip() for c in os.environ["TEST_CORPORA"].split(",") if c.strip()]
        spec = {**spec, "test_corpora": corpora}
    elif os.environ.get("TEST_CORPUS"):
        spec = {**spec, "test_corpora": [os.environ["TEST_CORPUS"]]}
    base_config_path = TEXT_ROOT / str(
        spec.get("base_config", "configs/methods/supervised_macro_ft.yaml")
    )
    base_cfg = load_yaml(base_config_path)
    test_corpora = list(
        spec.get("test_corpora")
        or base_cfg.get("test_corpora")
        or ["metallurgie", "caou", "nicollin"]
    )
    base_cfg = {**base_cfg, "test_corpora": test_corpora}

    grid = spec.get("grid") or {}
    validate_macro_ft_grid_keys(grid)
    n_folds = int(spec.get("n_folds", base_cfg.get("training", {}).get("n_folds", 3)))
    selection_metric = str(spec.get("selection_metric", "balanced_accuracy"))
    variants_output = str(
        spec.get("output_dir", "output/supervised_macro_ft/variants")
    )
    seed = int(tune_args.seed if tune_args.seed is not None else spec.get("seed", 42))
    save_checkpoint = bool(spec.get("save_checkpoint", False))
    save_btp_embeddings = bool(spec.get("save_btp_embeddings", False))
    full_encoder_training_overrides = dict(spec.get("full_encoder_training_overrides") or {})

    combos = expand_grid(grid)
    combos = filter_macro_ft_tuning_combos(combos, base_cfg)
    combos = [
        apply_full_encoder_training_overrides(
            overrides,
            base_cfg,
            full_encoder_training_overrides,
        )
        for overrides in combos
    ]
    if tune_args.max_combos is not None:
        combos = combos[: tune_args.max_combos]

    variants_root = TEXT_ROOT / variants_output
    ensure_dir(variants_root / "combos")

    log = logging.getLogger(__name__)
    log.info(
        "[macro_ft variants] %d combos, n_folds=%d, corpora=%s",
        len(combos),
        n_folds,
        test_corpora,
    )

    summary_rows: List[Dict[str, Any]] = []

    for combo_idx, overrides in enumerate(combos, start=1):
        cid = _combo_id(overrides)
        combo_dir = variants_root / "combos" / cid
        merged = _merge_overrides(base_cfg, overrides)
        model_section = dict(merged.get("model") or {})
        train_section = dict(merged.get("training") or {})
        train_section["n_folds"] = n_folds
        train_section["seed"] = seed
        exports = dict(merged.get("exports") or {})
        exports["save_checkpoint"] = save_checkpoint
        exports["save_btp_embeddings"] = save_btp_embeddings
        exports["save_fold_checkpoints"] = False
        merged = {
            **merged,
            "model": model_section,
            "training": train_section,
            "exports": exports,
            "test_corpora": test_corpora,
            "output_dir": str(combo_dir.relative_to(TEXT_ROOT)).replace("\\", "/"),
        }

        scope = encoder_scope_label(model_section.get("train_last_n_layers"))
        proj = projector_label(model_section.get("projection"))
        log.info(
            "[macro_ft variants] combo %d/%d: %s | %s | projector=%s",
            combo_idx,
            len(combos),
            cid,
            scope,
            proj,
        )
        t0 = time.perf_counter()

        if tune_args.skip_final_fit:
            row = _run_combo_cv(
                base_cfg,
                overrides,
                combo_output_dir=combo_dir,
                backbone_hidden=None,
                shared_cache_dir=variants_root / "_unused_cache",
                n_folds=n_folds,
                seed=seed,
                selection_metric=selection_metric,
            )
        else:
            result = run_supervised_macro_ft_training(
                None,
                cfg=merged,
                output_dir_override=combo_dir,
                backbone_hidden=None,
                shared_cache_dir=None,
            )
            cv_summary = pd.DataFrame(result.get("cv_summary") or [])
            row = {"combo_output_dir": str(combo_dir)}
            if not cv_summary.empty:
                row.update(cv_summary.iloc[0].to_dict())
            row["selection_score"] = float(
                row.get("mean_balanced_accuracy", float("nan"))
            )
            # BA OOD depuis métriques écrites
            for corpus_id in test_corpora:
                row[f"ba_ood_{corpus_id}"] = _find_ood_ba(combo_dir, corpus_id)

        elapsed = time.perf_counter() - t0
        row["combo_id"] = cid
        row["encoder_scope"] = scope
        row["projector"] = proj
        row.update({k.replace(".", "_"): v for k, v in overrides.items()})
        summary_rows.append(row)
        log.info(
            "[macro_ft variants] fin %d/%d %s / %s CV_BA=%.4f (%.1f min)",
            combo_idx,
            len(combos),
            scope,
            proj,
            float(row.get("selection_score", float("nan"))),
            elapsed / 60,
        )

    grid_df = pd.DataFrame(summary_rows)
    grid_df.to_csv(variants_root / "grid_summary.csv", index=False)

    results = build_variants_results_summary(summary_rows, test_corpora=test_corpora)
    results_path = variants_root / "results_summary.csv"
    results.to_csv(results_path, index=False)
    (variants_root / "results_summary.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    log.info("[macro_ft variants] Tableau article → %s", results_path)
    if not results.empty:
        cols = [
            c
            for c in (
                "encoder_scope",
                "projector",
                "cv_ba_mean",
                "cv_ba_std",
                *[f"ba_ood_{c}" for c in test_corpora],
                "ba_ood_avg",
                "ba_ood_worst",
            )
            if c in results.columns
        ]
        print("\n=== Results summary (article) ===", flush=True)
        print(results[cols].to_string(index=False), flush=True)
    return 0
