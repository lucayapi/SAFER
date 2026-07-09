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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_methods.config import merge_config_dict
from safer_core.io import ensure_dir, load_yaml
from safer_core.paths import TEXT_ROOT
from supervised_macro_ft.train_runner import (
    prepare_shared_backbone_hidden,
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
    """Exclut les combinaisons YAML impossibles ou redondantes."""
    from supervised_macro_ft.backbone_scaler import should_standardize_backbone
    from supervised_macro_ft.class_balance import resolve_train_balance
    from supervised_macro_ft.model import validate_macro_ft_projection

    model = dict(model_cfg or {})
    try:
        validate_macro_ft_projection(model.get("projection", "mlp_sklearn"))
        resolve_train_balance(model)
    except ValueError:
        return False

    backbone_trainable = bool(model.get("backbone_trainable", False))
    train_last_n = model.get("train_last_n_layers")
    if not backbone_trainable and train_last_n is not None:
        try:
            if int(train_last_n) > 0:
                return False
        except (TypeError, ValueError):
            return False

    if backbone_trainable and should_standardize_backbone(model):
        return False

    projection = validate_macro_ft_projection(model.get("projection", "mlp_sklearn"))
    if projection == "mlp_sklearn":
        hiddim = model.get("hiddim")
        if hiddim is not None and int(hiddim) != 128:
            return False

    return True


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


def run_supervised_macro_ft_tuning(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-config",
        type=str,
        default="configs/tuning/supervised_macro_ft_grid.yaml",
    )
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--skip-final-fit", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override seed global")
    tune_args = parser.parse_args(argv)

    spec = load_yaml(TEXT_ROOT / tune_args.grid_config)
    if os.environ.get("TEST_CORPUS"):
        spec = {**spec, "test_corpus": os.environ["TEST_CORPUS"]}
    base_config_path = TEXT_ROOT / str(spec.get("base_config", "configs/methods/supervised_macro_ft.yaml"))
    base_cfg = load_yaml(base_config_path)
    if spec.get("test_corpus"):
        base_cfg = {**base_cfg, "test_corpus": spec["test_corpus"]}

    grid = spec.get("grid") or {}
    validate_macro_ft_grid_keys(grid)
    n_folds = int(spec.get("n_folds", 3))
    selection_metric = str(spec.get("selection_metric", "balanced_accuracy"))
    tuning_output = str(spec.get("output_dir", "output/supervised_macro_ft/tuning"))
    final_output = str(spec.get("final_output_dir", "output/supervised_macro_ft"))
    seed = int(tune_args.seed if tune_args.seed is not None else spec.get("seed", 42))

    combos = expand_grid(grid)
    combos = filter_macro_ft_tuning_combos(combos, base_cfg)
    if tune_args.max_combos is not None:
        combos = combos[: tune_args.max_combos]

    tuning_root = TEXT_ROOT / tuning_output
    ensure_dir(tuning_root / "combos")
    shared_cache_dir = tuning_root / "_shared_cache"
    ensure_dir(shared_cache_dir)

    log = logging.getLogger(__name__)
    log.info(
        "[macro_ft tuning] %d combos valides, n_folds=%d, selection=%s",
        len(combos),
        n_folds,
        selection_metric,
    )

    log.info("[macro_ft tuning] Chargement cache backbone partagé…")
    backbone_hidden = prepare_shared_backbone_hidden(base_cfg, cache_dir=shared_cache_dir)
    if backbone_hidden is not None:
        log.info("[macro_ft tuning] cache backbone : shape=%s", backbone_hidden.shape)

    summary_rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_combo_id: Optional[str] = None
    best_overrides: Dict[str, Any] = {}

    for combo_idx, overrides in enumerate(combos, start=1):
        cid = _combo_id(overrides)
        combo_dir = tuning_root / "combos" / cid
        log.info("[macro_ft tuning] combo %d/%d: %s", combo_idx, len(combos), cid)
        t0 = time.perf_counter()
        row = _run_combo_cv(
            base_cfg,
            overrides,
            combo_output_dir=combo_dir,
            backbone_hidden=backbone_hidden,
            shared_cache_dir=shared_cache_dir,
            n_folds=n_folds,
            seed=seed,
            selection_metric=selection_metric,
        )
        elapsed = time.perf_counter() - t0
        row["combo_id"] = cid
        row.update({k.replace(".", "_"): v for k, v in overrides.items()})
        summary_rows.append(row)
        score = float(row.get("selection_score", float("nan")))
        log.info(
            "[macro_ft tuning] fin combo %d/%d score=%.4f (%.1f min)",
            combo_idx,
            len(combos),
            score,
            elapsed / 60,
        )
        if score == score and score > best_score:
            best_score = score
            best_combo_id = cid
            best_overrides = dict(overrides)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tuning_root / "grid_summary.csv", index=False)

    best_info = {
        "combo_id": best_combo_id,
        "selection_metric": selection_metric,
        "selection_score": best_score,
        "n_folds": n_folds,
        "overrides": best_overrides,
    }
    with open(tuning_root / "best_combo.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)

    if not tune_args.skip_final_fit and best_combo_id:
        final_cfg = _merge_overrides(base_cfg, best_overrides)
        train_section = dict(final_cfg.get("training") or {})
        train_section["seed"] = seed
        final_cfg = {**final_cfg, "training": train_section, "output_dir": final_output}
        log.info("[macro_ft tuning] Réentraînement final 100 %% BTP…")
        run_supervised_macro_ft_training(
            None,
            cfg=final_cfg,
            output_dir_override=TEXT_ROOT / final_output,
            backbone_hidden=backbone_hidden,
            shared_cache_dir=shared_cache_dir,
        )
        configs_dir = TEXT_ROOT / final_output / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        with open(configs_dir / "best_combo.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(best_info, f, sort_keys=False, allow_unicode=True)

    log.info("[macro_ft tuning] Meilleur : %s (score=%.4f)", best_combo_id, best_score)
    return 0
