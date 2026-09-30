"""Apply source-only selected architecture/LR settings to the final recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _best_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    for column in ("selection_score", "cv_ba_mean", "mean_balanced_accuracy"):
        if column in df:
            return df.sort_values([column, "combo_id"] if "combo_id" in df else [column], ascending=[False, True] if "combo_id" in df else False).iloc[0]
    raise ValueError(f"No selection score in {path}")


def _scope(row: pd.Series) -> int | None:
    raw = row.get("train_last_n_layers", row.get("model_train_last_n_layers"))
    if pd.notna(raw):
        return int(raw)
    label = str(row.get("encoder_scope", "")).strip().lower()
    if label in {"full", "full encoder"}:
        return None
    if label in {"last 1 layer", "last_1", "last_1_layer"}:
        return 1
    if label in {"last 2 layers", "last_2", "last_2_layers"}:
        return 2
    if label in {"last 3 layers", "last_3", "last_3_layers"}:
        return 3
    raise ValueError(f"Unknown encoder scope in selection summary: {label!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", default="output/replication_recipes/backbone_source_factorial.yaml")
    p.add_argument("--tuning-root", default="output/backbone_source_factorial/source_only_tuning")
    p.add_argument("--write", action="store_true", help="Write the selected overrides into the recipe.")
    args = p.parse_args()
    recipe_path = ROOT / args.recipe; tuning_root = ROOT / args.tuning_root
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    changes = []
    for model_id, model in recipe["models"].items():
        method = next((name for name in ("cross_entropy", "softtriple", "supcon", "frozen") if model_id.endswith("_" + name)), None)
        if method is None:
            raise ValueError(f"Cannot infer method from model id: {model_id}")
        if method == "frozen":
            continue
        if bool(model.get("reuse_existing_selection", False)):
            continue
        prefix = model_id.rsplit("_" + method, 1)[0]
        # Tuning output naming includes the whole method for cross_entropy.
        tuning_dir = tuning_root / f"{prefix}_{method}"
        summary = tuning_dir / ("results_summary.csv" if method == "cross_entropy" else "grid_summary.csv")
        if not summary.is_file():
            raise FileNotFoundError(f"Selection summary missing for {model_id}: {summary}")
        row = _best_row(summary)
        overrides = model.setdefault("overrides", {})
        model_override = overrides.setdefault("model", {})
        model_override["train_last_n_layers"] = _scope(row)
        yes = str(row.get("projector", "Yes")).strip().lower() in {"yes", "true", "1"}
        model_override["use_projector"] = yes
        if not yes:
            model_override["projection"] = None
        if method in {"supcon", "softtriple"} and pd.notna(row.get("best_lr_C")):
            model.setdefault("classifier", {})["C"] = float(row["best_lr_C"])
        model["source_only_selection"] = {"summary": str(summary.relative_to(ROOT)).replace("\\", "/"), "score": float(row.get("selection_score", row.get("cv_ba_mean", row.get("mean_balanced_accuracy"))))}
        changes.append(model_id)
    if args.write:
        recipe_path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("\n".join(changes))


if __name__ == "__main__":
    main()
