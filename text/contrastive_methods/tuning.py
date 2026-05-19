"""Grid search contrastif avec K-fold — sélection par mean eta2_macro_balanced_perc."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from contrastive_methods.config import load_contrastive_config_from_dict, merge_config_dict
from contrastive_methods.eval_corpus import evaluate_btp_and_test
from contrastive_methods.eval_geometry import selection_score
from contrastive_methods.kfold_train import get_contrastive_runner, run_tuning_combo_kfold
from metrics.geometry import GEOMETRY_METRIC_KEYS, PRIMARY_SELECTION_METRIC
from safer_core.io import ensure_dir, load_yaml, save_config_resolved
from safer_core.paths import TEXT_ROOT


def _format_combo_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.0e}".replace("+", "")
    return str(value)


def combo_id_from_overrides(overrides: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(overrides.keys()):
        short = key.split(".")[-1]
        parts.append(f"{short}{_format_combo_value(overrides[key])}")
    return "_".join(parts)[:120] if parts else "default"


def expand_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values_list = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values_list)]


def run_tuning(method_name: str, argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Tuning contrastif : {method_name}")
    parser.add_argument(
        "--grid-config",
        type=str,
        default=f"configs/tuning/{method_name}_grid.yaml",
    )
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--skip-final-fit", action="store_true")
    args, _ = parser.parse_known_args(argv)

    spec = load_yaml(TEXT_ROOT / args.grid_config)
    base_path = TEXT_ROOT / str(spec.get("base_config", f"configs/methods/{method_name}.yaml"))
    base_raw = load_yaml(base_path)
    grid = spec.get("grid") or {}
    selection_metric = str(spec.get("selection_metric", PRIMARY_SELECTION_METRIC))
    n_folds = int(spec.get("n_folds", 5))
    seed = int(spec.get("seed", base_raw.get("seed", 42)))
    tuning_output = str(spec.get("output_dir", f"resultats/{method_name}/tuning"))
    final_output = str(spec.get("final_output_dir", f"resultats/{method_name}"))
    final_fit = bool(spec.get("final_fit_full_data", True)) and not args.skip_final_fit
    test_dataset = str(spec.get("test_dataset_path", "dataset/test/data_metallurgie.csv"))

    combos = expand_grid(grid)
    if args.max_combos is not None:
        combos = combos[: args.max_combos]

    runner = get_contrastive_runner(method_name)
    tuning_root = TEXT_ROOT / tuning_output
    ensure_dir(tuning_root / "combos")

    summary_rows: List[Dict[str, Any]] = []
    best_combo_id = None
    best_score = float("-inf")
    best_overrides: Dict[str, Any] = {}

    for overrides in combos:
        cid = combo_id_from_overrides(overrides)
        if n_folds > 1:
            row = run_tuning_combo_kfold(
                method_name, runner, base_raw, overrides, cid, tuning_output, n_folds, seed, selection_metric
            )
        else:
            merged = merge_config_dict(base_raw, overrides)
            merged["output_dir"] = str(Path(tuning_output) / "combos" / cid)
            merged["final_fit_full_data"] = False
            merged["selection_metric"] = selection_metric
            merged["method_name"] = method_name
            cfg = load_contrastive_config_from_dict(method_name, merged, config_path=str(args.grid_config))
            result = runner(cfg)
            row = {
                "combo_id": cid,
                "selection_metric": selection_metric,
                "selection_score": result.best_eta2_macro_balanced_perc,
                **{k.replace(".", "_"): v for k, v in overrides.items()},
            }
            if result.val_geometry:
                for key in GEOMETRY_METRIC_KEYS:
                    row[f"val_{key}"] = result.val_geometry.get(key)
        summary_rows.append(row)
        score = float(row.get("selection_score", row.get("selection_score", float("-inf"))))
        if not (score == score):  # NaN check
            score = selection_score(row, selection_metric)
        if score > best_score:
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

    print(f"[tuning] Meilleur combo : {best_combo_id} (score={best_score:.2f})")

    if final_fit and best_combo_id:
        merged_final = merge_config_dict(base_raw, best_overrides)
        merged_final["output_dir"] = final_output
        merged_final["final_fit_full_data"] = True
        merged_final["selection_metric"] = selection_metric
        merged_final["method_name"] = method_name
        merged_final["test_dataset_path"] = test_dataset
        cfg_final = load_contrastive_config_from_dict(
            method_name, merged_final, config_path=str(args.grid_config)
        )
        print("[tuning] Réentraînement final 100 % BTP…", flush=True)
        final_result = runner(cfg_final)
        ckpt = final_result.output_root / "checkpoints" / "best_model"
        evaluate_btp_and_test(cfg_final, ckpt, final_result.output_root)
        save_config_resolved(
            {
                **merged_final,
                "best_combo_id": best_combo_id,
                "best_tuning_score": best_score,
                "tuning_grid_config": args.grid_config,
            },
            final_result.output_root,
        )
        with open(final_result.output_root / "configs" / "best_combo.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(best_info, f, sort_keys=False, allow_unicode=True)
        print(f"[tuning] Métriques BTP/test : {final_result.output_root / 'metrics'}")

    return 0
