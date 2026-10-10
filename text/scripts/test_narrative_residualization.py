#!/usr/bin/env python3
"""Source-only selection and OOD test of narrative-level embedding residuals.

Each invocation analyzes one backbone/source pair. All model selection uses
grouped source folds; target labels are read only after the final source fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from safer_core.embedding_paths import embedding_export_path
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.data_metadata import LABEL2ID, load_filtered_metadata
from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings


CORPORA = ("btp", "metallurgie", "caou", "nicollin")
BACKBONES = {
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", "qwen3"),
    "multilingual_e5_large": ("intfloat/multilingual-e5-large", "multilingual_e5_large"),
}
ROLES = ("A0", "A1", "B", "C")


def load_corpus(corpus: str, backbone: str) -> tuple[pd.DataFrame, np.ndarray]:
    spec = resolve_test_corpus(corpus, require_files=True, require_emb_csv=False)
    backbone_name, backbone_id = BACKBONES[backbone]
    emb_path = embedding_export_path(corpus, backbone_name=backbone_name,
                                     backbone_id=backbone_id, output_root="embeddings")
    if not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings absents pour {backbone}/{corpus}: {emb_path}")
    meta = load_filtered_metadata(str(spec.data_csv), label_col="pred_label",
                                 pred_ok_col="pred_ok", group_col="accident_id")
    meta, dims = merge_metadata_with_embeddings(meta, str(emb_path), strict=True, corpus_id=corpus)
    X = meta[dims].to_numpy(dtype=np.float32, copy=True)
    labels = meta["pred_label"].astype(str)
    if not labels.isin(LABEL2ID).all():
        raise ValueError(f"Labels non reconnus dans {corpus}")
    meta = meta.reset_index(drop=True)
    return meta, X


def make_splits(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups.astype(str))
    if len(unique) < n_splits:
        raise ValueError(f"Seulement {len(unique)} accidents pour {n_splits} plis")
    # Shuffle group labels deterministically before GroupKFold for repeatable varied folds.
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {g: i for i, g in enumerate(shuffled)}
    encoded = np.asarray([mapping[g] for g in groups.astype(str)], dtype=np.int64)
    return [(a.astype(np.int64), b.astype(np.int64))
            for a, b in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=encoded)]


def accident_means(X: np.ndarray, groups: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    frame = pd.DataFrame({"group": groups.astype(str), "row": np.arange(len(groups))})
    means, sizes = {}, {}
    for group, rows in frame.groupby("group", sort=False)["row"]:
        idx = rows.to_numpy(dtype=np.int64)
        means[group] = X[idx].mean(axis=0, dtype=np.float64).astype(np.float32)
        sizes[group] = len(idx)
    return means, sizes


def transform_features(
    X: np.ndarray,
    groups: np.ndarray,
    mode: str,
    alpha: float,
    *,
    donor_X: np.ndarray | None = None,
    donor_groups: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Use same-accident leave-one-out means or matched unrelated-accident means."""
    X = np.asarray(X, dtype=np.float32)
    groups = groups.astype(str)
    if mode == "raw" or alpha == 0:
        return X
    if mode == "narrative":
        sums, counts = {}, {}
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group)
            sums[group] = X[idx].sum(axis=0, dtype=np.float64)
            counts[group] = len(idx)
        result = X.astype(np.float32, copy=True)
        for i, group in enumerate(groups):
            n = counts[group]
            if n > 1:
                sibling_mean = (sums[group] - X[i]) / (n - 1)
                result[i] = X[i] - alpha * sibling_mean.astype(np.float32)
        return result
    if mode != "random_control":
        raise ValueError(f"Mode inconnu: {mode}")
    if donor_X is None or donor_groups is None:
        raise ValueError("Le contrôle aléatoire requiert les accidents donneurs du pli d'entraînement")
    means, donor_sizes = accident_means(donor_X, donor_groups)
    donor_ids = np.asarray(sorted(means), dtype=object)
    if len(donor_ids) < 2:
        raise ValueError("Pas assez d'accidents donneurs pour le contrôle aléatoire")
    _, target_sizes = accident_means(X, groups)
    rng = np.random.default_rng(seed)
    donor_for_group: dict[str, str] = {}
    for group in np.unique(groups):
        eligible = [d for d in donor_ids if d != group]
        if not eligible:
            eligible = donor_ids.tolist()
        distance = np.asarray([abs(donor_sizes[d] - target_sizes[group]) for d in eligible])
        nearest = [eligible[j] for j in np.flatnonzero(distance == distance.min())]
        donor_for_group[group] = str(rng.choice(nearest))
    result = X.copy()
    for i, group in enumerate(groups):
        result[i] -= alpha * means[donor_for_group[group]]
    return result


def make_model(C: float, seed: int) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=float(C), class_weight="balanced", solver="lbfgs", max_iter=2000,
        random_state=int(seed), tol=1e-4,
    ))


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(ROLES), average="macro", zero_division=0)),
    }
    for role in ROLES:
        result[f"precision_{role}"] = float(precision_score(y, pred, labels=[role], average="micro", zero_division=0))
        result[f"recall_{role}"] = float(recall_score(y, pred, labels=[role], average="micro", zero_division=0))
    return result


def select_params(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, mode: str,
    alphas: Iterable[float], c_values: Iterable[float], n_folds: int, seed: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        for C in c_values:
            fold_scores = []
            for fold, (tr, va) in enumerate(make_splits(groups, n_folds, seed)):
                Xt = (transform_features(X[tr], groups[tr], mode, alpha,
                                         donor_X=X[tr] if mode == "random_control" else None,
                                         donor_groups=groups[tr] if mode == "random_control" else None,
                                         seed=seed + fold)
                      if mode == "random_control" else transform_features(X[tr], groups[tr], mode, alpha, seed=seed + fold))
                if mode == "random_control":
                    Xv = transform_features(X[va], groups[va], mode, alpha,
                                            donor_X=X[tr], donor_groups=groups[tr], seed=seed + 1000 + fold)
                else:
                    Xv = transform_features(X[va], groups[va], mode, alpha, seed=seed + 1000 + fold)
                model = make_model(C, seed + fold)
                model.fit(Xt, y[tr])
                fold_scores.append(float(balanced_accuracy_score(y[va], model.predict(Xv))))
            rows.append({"mode": mode, "alpha": float(alpha), "C": float(C),
                         "cv_balanced_accuracy_mean": float(np.mean(fold_scores)),
                         "cv_balanced_accuracy_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                         "n_folds": len(fold_scores)})
    table = pd.DataFrame(rows)
    # Joint grid selection uses mean grouped-CV BA; ties favor smaller |alpha| and then C.
    table["_abs_alpha"] = table["alpha"].abs()
    best = table.sort_values(["cv_balanced_accuracy_mean", "_abs_alpha", "C", "alpha"],
                             ascending=[False, True, True, True], kind="mergesort").iloc[0]
    return float(best["alpha"]), float(best["C"]), table.drop(columns="_abs_alpha")


def nested_source_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, modes: dict[str, list[float]],
                    c_values: list[float], outer_folds: int, inner_folds: int, seed: int) -> pd.DataFrame:
    rows = []
    for outer, (tr, va) in enumerate(make_splits(groups, outer_folds, seed)):
        for mode, alphas in modes.items():
            a, C, _ = select_params(X[tr], y[tr], groups[tr], mode=mode, alphas=alphas,
                                    c_values=c_values, n_folds=inner_folds, seed=seed + 17 * (outer + 1))
            Xt = (transform_features(X[tr], groups[tr], mode, a, donor_X=X[tr], donor_groups=groups[tr], seed=seed + outer)
                  if mode == "random_control" else transform_features(X[tr], groups[tr], mode, a, seed=seed + outer))
            Xv = (transform_features(X[va], groups[va], mode, a, donor_X=X[tr], donor_groups=groups[tr], seed=seed + 3000 + outer)
                  if mode == "random_control" else transform_features(X[va], groups[va], mode, a, seed=seed + 3000 + outer))
            model = make_model(C, seed + outer); model.fit(Xt, y[tr]); pred = model.predict(Xv)
            rows.append({"outer_fold": outer, "mode": mode, "selected_alpha": a, "selected_C": C,
                         "n_train_accidents": int(pd.Series(groups[tr]).nunique()),
                         "n_validation_accidents": int(pd.Series(groups[va]).nunique()), **score(y[va], pred)})
    return pd.DataFrame(rows)


def accident_bootstrap(y: np.ndarray, predictions: dict[str, np.ndarray], groups: np.ndarray,
                       *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups.astype(str))
    rows = []
    metric_functions = {
        "balanced_accuracy": lambda truth, pred: balanced_accuracy_score(truth, pred),
        "precision_A1": lambda truth, pred: precision_score(truth, pred, labels=["A1"], average="micro", zero_division=0),
        "recall_A1": lambda truth, pred: recall_score(truth, pred, labels=["A1"], average="micro", zero_division=0),
    }
    point = {(name, metric): float(fn(y, pred))
             for name, pred in predictions.items() for metric, fn in metric_functions.items()}
    boot = {(name, metric): np.empty(n_bootstrap)
            for name in predictions for metric in metric_functions}
    group_rows = {g: np.flatnonzero(groups.astype(str) == g) for g in unique}
    for b in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_rows[g] for g in sampled])
        for name, pred in predictions.items():
            for metric, fn in metric_functions.items():
                boot[(name, metric)][b] = float(fn(y[idx], pred[idx]))
    for (name, metric), values in boot.items():
        rows.append({"comparison": name, "metric": metric, "estimate": point[(name, metric)],
                     "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                     "n_accidents": len(unique), "n_bootstrap": n_bootstrap})
    for name in predictions:
        if name == "raw":
            continue
        for metric in metric_functions:
            delta = boot[(name, metric)] - boot[("raw", metric)]
            rows.append({"comparison": f"{name} minus raw", "metric": f"paired_{metric}_difference",
                         "estimate": point[(name, metric)] - point[("raw", metric)],
                         "ci_low": float(np.quantile(delta, .025)), "ci_high": float(np.quantile(delta, .975)),
                         "n_accidents": len(unique), "n_bootstrap": n_bootstrap,
                         "paired_resampling": "same sampled accidents"})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out = Path(args.output) / f"{args.backbone}_{args.source}"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Sortie non vide: {out}. Choisir un autre --output ou --overwrite.")

    source_meta, source_X = load_corpus(args.source, args.backbone)
    source_y = source_meta["pred_label"].astype(str).to_numpy()
    source_groups = source_meta["accident_id"].astype(str).to_numpy()
    modes = {"raw": [0.0], "narrative": args.alphas, "random_control": args.alphas}
    with threadpool_limits(limits=args.threads):
        cv = nested_source_cv(source_X, source_y, source_groups, modes=modes, c_values=args.c_values,
                              outer_folds=args.outer_folds, inner_folds=args.inner_folds, seed=args.seed)
        cv.to_csv(out / "nested_source_cv.csv", index=False)
        selected = {}
        selection_rows = []
        for mode, alphas in modes.items():
            alpha, C, table = select_params(source_X, source_y, source_groups, mode=mode, alphas=alphas,
                                            c_values=args.c_values, n_folds=args.selection_folds, seed=args.seed + 9000)
            table.to_csv(out / f"source_selection_{mode}.csv", index=False)
            selected[mode] = (alpha, C)
            selection_rows.append({"mode": mode, "selected_alpha": alpha, "selected_C": C,
                                   "selection_metric": "mean grouped source-validation balanced accuracy"})
        pd.DataFrame(selection_rows).to_csv(out / "selected_source_parameters.csv", index=False)

        raw_model = make_model(selected["raw"][1], args.seed)
        raw_model.fit(source_X, source_y)
        models = {"raw": raw_model}
        for mode in ("narrative", "random_control"):
            alpha, C = selected[mode]
            Xfit = transform_features(source_X, source_groups, mode, alpha,
                                       donor_X=source_X if mode == "random_control" else None,
                                       donor_groups=source_groups if mode == "random_control" else None,
                                       seed=args.seed + 12000)
            model = make_model(C, args.seed); model.fit(Xfit, source_y); models[mode] = model

        metric_rows, ci_rows = [], []
        target_corpora = [c for c in CORPORA if c != args.source]
        for corpus in target_corpora:
            meta, X = load_corpus(corpus, args.backbone)
            y = meta["pred_label"].astype(str).to_numpy()
            groups = meta["accident_id"].astype(str).to_numpy()
            preds = {}
            for mode, model in models.items():
                if mode == "raw":
                    Xt = X
                else:
                    alpha = selected[mode][0]
                    Xt = transform_features(X, groups, mode, alpha,
                                            donor_X=X if mode == "random_control" else None,
                                            donor_groups=groups if mode == "random_control" else None,
                                            seed=args.seed + 20000 + CORPORA.index(corpus))
                pred = model.predict(Xt); preds[mode] = pred
                metric_rows.append({"backbone": args.backbone, "source": args.source, "target": corpus,
                                    "mode": mode, "alpha": selected[mode][0], "C": selected[mode][1],
                                    "n_units": len(y), "n_accidents": int(pd.Series(groups).nunique()), **score(y, pred)})
                pd.DataFrame({"accident_id": groups, "true_role": y, "predicted_role": pred}).to_csv(
                    out / f"predictions_{corpus}_{mode}.csv", index=False)
            ci = accident_bootstrap(y, preds, groups, n_bootstrap=args.bootstrap, seed=args.seed + 30000 + CORPORA.index(corpus))
            ci.insert(0, "target", corpus); ci.insert(0, "source", args.source); ci.insert(0, "backbone", args.backbone)
            ci_rows.append(ci)
        pd.DataFrame(metric_rows).to_csv(out / "target_metrics.csv", index=False)
        pd.concat(ci_rows, ignore_index=True).to_csv(out / "target_accident_bootstrap.csv", index=False)
    manifest = vars(args).copy()
    manifest.update({"target_corpora": target_corpora, "selection_uses_target_labels": False,
                     "residual_definition": "h_i - alpha * mean(other units in same accident)",
                     "random_control": "mean embedding of unrelated accident matched to nearest accident unit count",
                     "wall_time_seconds": time.time() - started})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {out}; elapsed={time.time() - started:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--source", choices=("btp", "metallurgie"), required=True)
    parser.add_argument("--output", default="output/narrative_residualization")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, .25, .5, .75, 1.0, 1.25])
    parser.add_argument("--c-values", nargs="+", type=float, default=[.001, .01, .1, 1.0])
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap doit être >= 100")
    if 0.0 not in args.alphas or any(a < 0 or a > 2 for a in args.alphas):
        parser.error("La grille alpha doit contenir 0 et rester dans [0, 2]")
    return args


if __name__ == "__main__":
    run(parse_args())
