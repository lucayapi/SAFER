#!/usr/bin/env python3
"""Compare unit, same-accident context, and concatenated text embeddings.

All hyperparameter selection is grouped source-only CV. Target labels are used
only for final transfer metrics and paired accident-level bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

TEXT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for import_root in (TEXT_ROOT, SCRIPTS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from test_narrative_residualization import (  # noqa: E402
    BACKBONES,
    CORPORA,
    accident_bootstrap,
    load_corpus,
    make_model,
    make_splits,
    score,
)


REPRESENTATIONS = ("unit_only", "context_only", "unit_plus_context")


def leave_one_out_context(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return each unit's mean sibling embedding; singletons receive zeros."""
    X = np.asarray(X, dtype=np.float32)
    groups = np.asarray(groups).astype(str)
    if X.ndim != 2 or len(X) != len(groups):
        raise ValueError("X doit être une matrice alignée sur groups")

    context = np.zeros_like(X, dtype=np.float32)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) > 1:
            total = X[idx].sum(axis=0, dtype=np.float64)
            context[idx] = ((total[None, :] - X[idx]) / (len(idx) - 1)).astype(np.float32)
    return context


def make_representation(X: np.ndarray, groups: np.ndarray, name: str) -> np.ndarray:
    """Build a unit-only, context-only, or concatenated feature matrix."""
    X = np.asarray(X, dtype=np.float32)
    if name == "unit_only":
        return X
    context = leave_one_out_context(X, groups)
    if name == "context_only":
        return context
    if name == "unit_plus_context":
        return np.concatenate((X, context), axis=1)
    raise ValueError(f"Représentation inconnue: {name}")


def load_existing_raw_selection(baseline_dir: Path) -> tuple[float, float]:
    """Read the already-selected source-only raw C and CV balanced accuracy."""
    summary_path = baseline_dir / "selected_source_parameters.csv"
    grid_path = baseline_dir / "source_selection_raw.csv"
    if not summary_path.is_file() or not grid_path.is_file():
        raise FileNotFoundError(
            "Sélection du baseline absente; attendus: "
            f"{summary_path} et {grid_path}"
        )
    summary = pd.read_csv(summary_path)
    selected = summary.loc[summary["mode"].eq("raw")]
    if len(selected) != 1:
        raise ValueError(f"Une seule sélection raw attendue dans {summary_path}")
    C = float(selected.iloc[0]["selected_C"])

    grid = pd.read_csv(grid_path)
    grid = grid.loc[np.isclose(pd.to_numeric(grid["alpha"]), 0.0)]
    row = grid.loc[np.isclose(pd.to_numeric(grid["C"]), C)]
    if len(row) != 1:
        raise ValueError(f"C={C} absent ou dupliqué dans {grid_path}")
    return C, float(row.iloc[0]["cv_balanced_accuracy_mean"])


def source_cv_grid(
    X_by_representation: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    c_values: list[float],
    n_folds: int,
    seed: int,
    fixed_unit_C: float,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Select C on grouped source CV, preserving the preselected unit baseline."""
    splits = make_splits(groups, n_folds, seed)
    rows: list[dict[str, float | str | int | bool]] = []
    selected: dict[str, tuple[float, float]] = {}

    for representation, X in X_by_representation.items():
        # Keep the unit-only condition identical to the already-reported
        # baseline instead of re-tuning it in this follow-up experiment.
        candidates = [fixed_unit_C] if representation == "unit_only" else c_values
        scores_by_C: dict[float, list[float]] = {}
        for C in candidates:
            fold_scores = []
            for fold, (train_idx, valid_idx) in enumerate(splits):
                model = make_model(C, seed + fold)
                model.fit(X[train_idx], y[train_idx])
                fold_scores.append(float(balanced_accuracy_score(
                    y[valid_idx], model.predict(X[valid_idx])
                )))
            scores_by_C[float(C)] = fold_scores

        if representation == "unit_only":
            best_C = float(fixed_unit_C)
        else:
            best_C = sorted(
                scores_by_C,
                key=lambda C: (-float(np.mean(scores_by_C[C])), C),
            )[0]
        best_scores = scores_by_C[best_C]
        selected[representation] = (best_C, float(np.mean(best_scores)))

        for C, fold_scores in scores_by_C.items():
            rows.append({
                "representation": representation,
                "C": C,
                "cv_balanced_accuracy_mean": float(np.mean(fold_scores)),
                "cv_balanced_accuracy_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                "n_folds": len(fold_scores),
                "selected": bool(np.isclose(C, best_C)),
                "selection_rule": "existing raw source-selected C" if representation == "unit_only" else "best grouped source-CV balanced accuracy; ties favour smaller C",
            })

    return pd.DataFrame(rows), selected


def verify_existing_raw_predictions(
    baseline_dir: Path,
    target: str,
    groups: np.ndarray,
    y: np.ndarray,
    raw_predictions: np.ndarray,
) -> bool:
    """Check exact row-aligned agreement with the recorded raw baseline."""
    path = baseline_dir / f"predictions_{target}_raw.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prédictions raw de référence absentes: {path}")
    old = pd.read_csv(path)
    if not {"accident_id", "true_role", "predicted_role"}.issubset(old.columns):
        raise ValueError(f"Colonnes inattendues dans {path}")
    return (
        len(old) == len(y)
        and np.array_equal(old["accident_id"].astype(str).to_numpy(), groups.astype(str))
        and np.array_equal(old["true_role"].astype(str).to_numpy(), y.astype(str))
        and np.array_equal(old["predicted_role"].astype(str).to_numpy(), raw_predictions.astype(str))
    )


def run(args: argparse.Namespace) -> None:
    started = time.time()
    baseline_dir = Path(args.baseline_input) / f"{args.backbone}_{args.source}"
    output_dir = Path(args.output) / f"{args.backbone}_{args.source}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Sortie non vide: {output_dir}; choisir --output ou --overwrite")

    source_meta, source_embeddings = load_corpus(args.source, args.backbone)
    source_y = source_meta["pred_label"].astype(str).to_numpy()
    source_groups = source_meta["accident_id"].astype(str).to_numpy()
    baseline_C, recorded_baseline_cv = load_existing_raw_selection(baseline_dir)

    targets = [corpus for corpus in CORPORA if corpus != args.source]
    metrics_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    baseline_matches: dict[str, bool] = {}

    with threadpool_limits(limits=args.threads):
        source_features = {
            name: make_representation(source_embeddings, source_groups, name)
            for name in REPRESENTATIONS
        }
        cv_grid, selected = source_cv_grid(
            source_features,
            source_y,
            source_groups,
            c_values=args.c_values,
            n_folds=args.selection_folds,
            seed=args.seed + 9000,
            fixed_unit_C=baseline_C,
        )
        cv_grid.to_csv(output_dir / "source_selection.csv", index=False)

        models = {}
        for name in REPRESENTATIONS:
            C, _ = selected[name]
            model = make_model(C, args.seed)
            model.fit(source_features[name], source_y)
            models[name] = model

        for target_index, target in enumerate(targets):
            meta, embeddings = load_corpus(target, args.backbone)
            y = meta["pred_label"].astype(str).to_numpy()
            groups = meta["accident_id"].astype(str).to_numpy()
            target_features = {
                name: make_representation(embeddings, groups, name)
                for name in REPRESENTATIONS
            }
            predictions: dict[str, np.ndarray] = {}
            for name in REPRESENTATIONS:
                pred = models[name].predict(target_features[name])
                # The existing bootstrap helper uses "raw" as the paired baseline key.
                key = "raw" if name == "unit_only" else name
                predictions[key] = pred
                C, source_cv_ba = selected[name]
                metrics_rows.append({
                    "backbone": args.backbone,
                    "source": args.source,
                    "target": target,
                    "representation": name,
                    "source_selected_C": C,
                    "source_cv_balanced_accuracy": source_cv_ba,
                    "n_units": len(y),
                    "n_accidents": int(pd.Series(groups).nunique()),
                    **score(y, pred),
                })

            baseline_matches[target] = verify_existing_raw_predictions(
                baseline_dir,
                target,
                groups,
                y,
                predictions["raw"],
            )
            if not baseline_matches[target]:
                raise AssertionError(
                    f"Le baseline unité seule ne reproduit pas les prédictions enregistrées pour {target}"
                )

            prediction_frame = pd.DataFrame({"accident_id": groups, "true_role": y})
            for name in REPRESENTATIONS:
                prediction_frame[f"predicted_{name}"] = predictions["raw" if name == "unit_only" else name]
            prediction_frame.to_csv(output_dir / f"predictions_{target}.csv", index=False)

            bootstrap = accident_bootstrap(
                y,
                predictions,
                groups,
                n_bootstrap=args.bootstrap,
                seed=args.seed + 30000 + target_index,
            )
            bootstrap.insert(0, "target", target)
            bootstrap.insert(0, "source", args.source)
            bootstrap.insert(0, "backbone", args.backbone)
            bootstrap_frames.append(bootstrap)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "target_metrics.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        output_dir / "target_accident_bootstrap.csv", index=False
    )
    manifest = vars(args).copy()
    manifest.update({
        "target_corpora": targets,
        "selection_uses_target_labels": False,
        "representations": list(REPRESENTATIONS),
        "context_definition": "mean embeddings of all other factual units in the same accident",
        "singleton_context": "zero vector",
        "unit_only_baseline_source_C": baseline_C,
        "unit_only_baseline_source_cv_balanced_accuracy": recorded_baseline_cv,
        "existing_raw_predictions_exactly_reproduced": baseline_matches,
        "bootstrap": {"resamples": args.bootstrap, "unit": "accident_id", "paired": True},
        "wall_time_seconds": time.time() - started,
    })
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {output_dir}; elapsed={time.time() - started:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--source", choices=("btp", "metallurgie"), required=True)
    parser.add_argument("--baseline-input", default="output/narrative_residualization")
    parser.add_argument("--output", default="output/narrative_context_concat")
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.selection_folds < 2:
        parser.error("--selection-folds doit être au moins 2")
    if args.bootstrap < 100:
        parser.error("--bootstrap doit être au moins 100")
    if args.threads < 1:
        parser.error("--threads doit être positif")
    if not args.c_values or any(c <= 0 for c in args.c_values):
        parser.error("--c-values doit contenir des valeurs strictement positives")
    return args


if __name__ == "__main__":
    run(parse_args())
