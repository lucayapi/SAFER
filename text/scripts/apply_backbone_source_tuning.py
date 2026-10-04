"""Apply source-only selected architecture/LR settings to the final recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from replication.runner import _model_config
from safer_core.embedding_paths import backbone_storage_id

ROOT = Path(__file__).resolve().parents[1]

# These values are part of the source-only grid, so the final repeated-seed
# recipe must use the same scope-specific optimisation budget.  In particular,
# full encoder adaptation is deliberately restricted to three epochs.
EPOCHS_BY_ENCODER_SCOPE = {1: 30, 2: 15, 3: 10, None: 3}


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


def _apply_scope_training_overrides(model: dict, *, method: str, scope: int | None) -> None:
    """Keep final repeated-seed optimisation consistent with source-only CV."""
    training = model.setdefault("overrides", {}).setdefault("training", {})
    training["epochs"] = EPOCHS_BY_ENCODER_SCOPE[scope]
    if scope is None:
        training["use_amp"] = False
        if method == "cross_entropy":
            training["lr_backbone"] = 2.0e-6
        else:
            training["learning_rate"] = 2.0e-6
    else:
        # Make repeated applications of this script deterministic if a new
        # selection replaces an earlier full-encoder selection.
        training["use_amp"] = True
        if method == "cross_entropy":
            training["lr_backbone"] = 2.0e-5
        else:
            training["learning_rate"] = 2.0e-5


def _selection_stem(recipe: dict, model_id: str, method: str) -> str:
    """Return the source-only tuning directory stem for a recipe model."""
    spec = _model_config(recipe, model_id)
    backbone_name = spec.get("backbone_name") or spec.get("overrides", {}).get("model", {}).get("backbone_name")
    if not backbone_name:
        raise ValueError(f"Cannot determine backbone for {model_id}")
    backbone_id = str(spec.get("backbone_id") or backbone_storage_id(str(backbone_name)))
    return f"{backbone_id}_{spec['source_corpus']}_{method}"


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
        # Tuning directories use the canonical backbone ID.  This differs
        # from the shorter CamemBERT model IDs used in its final recipe.
        tuning_dir = tuning_root / _selection_stem(recipe, model_id, method)
        summary = tuning_dir / ("results_summary.csv" if method == "cross_entropy" else "grid_summary.csv")
        if not summary.is_file():
            raise FileNotFoundError(f"Selection summary missing for {model_id}: {summary}")
        row = _best_row(summary)
        overrides = model.setdefault("overrides", {})
        model_override = overrides.setdefault("model", {})
        scope = _scope(row)
        model_override["train_last_n_layers"] = scope
        yes = str(row.get("projector", "Yes")).strip().lower() in {"yes", "true", "1"}
        model_override["use_projector"] = yes
        if not yes:
            model_override["projection"] = None
        if method in {"supcon", "softtriple"} and pd.notna(row.get("best_lr_C")):
            model.setdefault("classifier", {})["C"] = float(row["best_lr_C"])
        _apply_scope_training_overrides(model, method=method, scope=scope)
        model["source_only_selection"] = {"summary": str(summary.relative_to(ROOT)).replace("\\", "/"), "score": float(row.get("selection_score", row.get("cv_ba_mean", row.get("mean_balanced_accuracy"))))}
        changes.append(model_id)
    if args.write:
        recipe_path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("\n".join(changes))


if __name__ == "__main__":
    main()
