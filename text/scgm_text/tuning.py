"""Grille hyperparamètres SCGM avec K-fold groupé."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safer_core.io import load_yaml
from safer_core.paths import TEXT_ROOT, ensure_dir


def expand_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values_list = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values_list)]


def _combo_id(overrides: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(overrides.keys()):
        val = overrides[key]
        short = key.split(".")[-1]
        if isinstance(val, float):
            parts.append(f"{short}{val:.0e}".replace("+", ""))
        else:
            parts.append(f"{short}{val}")
    return "_".join(parts)[:120] if parts else "default"


def _apply_overrides(
    args: argparse.Namespace,
    overrides: Dict[str, Any],
    *,
    base_config_path: Path,
) -> argparse.Namespace:
    from contrastive_methods.config import merge_config_dict
    from scgm_text.fidelity import flatten_config_yaml

    merged = merge_config_dict(load_yaml(base_config_path), overrides)
    flat = flatten_config_yaml(merged)
    out = argparse.Namespace(**vars(args))
    for key, value in flat.items():
        key_norm = key.replace("-", "_")
        if hasattr(out, key_norm):
            setattr(out, key_norm, value)
    return out


def _run_combo(
    base_args: argparse.Namespace,
    overrides: Dict[str, Any],
    *,
    base_config_path: Path,
    output_dir: str,
    n_folds: int,
) -> Dict[str, Any]:
    from scripts.train_scgm_text import run_kfold, run_post_train_eval, run_training

    combo_args = _apply_overrides(base_args, overrides, base_config_path=base_config_path)
    combo_args.output_dir = output_dir
    combo_args.kfold = n_folds if n_folds > 1 else 0

    if n_folds > 1:
        run_kfold(combo_args)
        # Agrégation depuis kfold_summary si présent
        summary_path = Path(output_dir) / "metrics" / "kfold_summary.csv"
        if summary_path.is_file():
            row = pd.read_csv(summary_path).iloc[0].to_dict()
            return {"selection_score": row.get("selection_score", float("nan")), **row}
        return {"selection_score": float("nan")}

    metrics = run_training(combo_args)
    run_post_train_eval(combo_args)
    return {
        "selection_score": metrics.get(
            "eta2_macro_balanced_perc", metrics.get("best_checkpoint_score")
        ),
        **metrics,
    }


def run_scgm_tuning(argv: Optional[List[str]] = None) -> int:
    from scripts.train_scgm_text import apply_config, finalize_args, parse_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-config", type=str, default="configs/tuning/scgm_text_grid.yaml")
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--skip-final-fit", action="store_true")
    tune_args, _ = parser.parse_known_args(argv)

    spec = load_yaml(TEXT_ROOT / tune_args.grid_config)
    base_config = TEXT_ROOT / str(spec.get("base_config", "configs/scgm_text_strict_fidelity.yaml"))
    grid = spec.get("grid") or {}
    n_folds = int(spec.get("n_folds", 5))
    selection_metric = str(spec.get("selection_metric", "eta2_macro_balanced_perc"))
    tuning_output = str(spec.get("output_dir", "resultats/scgm_text/tuning"))
    final_output = str(spec.get("final_output_dir", "resultats/scgm_text"))

    base_args = parse_args()
    apply_config(base_args, str(base_config))
    finalize_args(base_args)
    base_args.best_checkpoint_metric = selection_metric

    combos = expand_grid(grid)
    if tune_args.max_combos is not None:
        combos = combos[: tune_args.max_combos]

    tuning_root = TEXT_ROOT / tuning_output
    ensure_dir(tuning_root / "combos")

    summary_rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_combo_id = None
    best_overrides: Dict[str, Any] = {}

    for overrides in combos:
        cid = _combo_id(overrides)
        combo_dir = str(tuning_root / "combos" / cid)
        print(f"[scgm tuning] {cid}", flush=True)
        row = _run_combo(
            base_args,
            overrides,
            base_config_path=base_config,
            output_dir=combo_dir,
            n_folds=n_folds,
        )
        row["combo_id"] = cid
        row.update({k.replace(".", "_"): v for k, v in overrides.items()})
        summary_rows.append(row)
        score = float(row.get("selection_score", float("nan")))
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
        from scripts.train_scgm_text import run_post_train_eval, run_training

        final_args = _apply_overrides(base_args, best_overrides, base_config_path=base_config)
        final_args.output_dir = final_output
        final_args.kfold = 0
        print("[scgm tuning] Réentraînement final 100 % BTP…", flush=True)
        run_training(final_args)
        run_post_train_eval(final_args)
        configs_dir = TEXT_ROOT / final_output / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        with open(configs_dir / "best_combo.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(best_info, f, sort_keys=False, allow_unicode=True)

    print(f"[scgm tuning] Meilleur : {best_combo_id} (score={best_score:.2f})")
    return 0
