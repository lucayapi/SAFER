#!/usr/bin/env python3
"""Diagnostic OOD evaluation of positive narrative-residualization alphas.

Alpha is not selected on target data. For each alpha, C is selected from the
already-recorded grouped source CV grid, then the fitted pipeline is evaluated
on every target. Target results are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

TEXT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (TEXT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_narrative_residualization import (  # noqa: E402
    BACKBONES,
    CORPORA,
    accident_bootstrap,
    load_corpus,
    make_model,
    score,
    transform_features,
)


POSITIVE_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def source_c_for_alpha(selection_file: Path, alpha: float) -> tuple[float, float]:
    if not selection_file.is_file():
        raise FileNotFoundError(f"Sélection source manquante: {selection_file}")
    grid = pd.read_csv(selection_file)
    grid = grid[np.isclose(pd.to_numeric(grid["alpha"]), alpha)]
    if grid.empty:
        raise ValueError(f"Alpha {alpha} absent de {selection_file}")
    grid = grid.sort_values(
        ["cv_balanced_accuracy_mean", "C"], ascending=[False, True], kind="mergesort"
    )
    row = grid.iloc[0]
    return float(row["C"]), float(row["cv_balanced_accuracy_mean"])


def run(args: argparse.Namespace) -> None:
    started = time.time()
    source_dir = Path(args.input) / f"{args.backbone}_{args.source}"
    if not (source_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"Résultats de la campagne initiale absents: {source_dir}")
    out = Path(args.output) / f"{args.backbone}_{args.source}"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Sortie non vide: {out}. Choisir --output ou --overwrite.")

    source_meta, source_X = load_corpus(args.source, args.backbone)
    source_y = source_meta["pred_label"].astype(str).to_numpy()
    source_groups = source_meta["accident_id"].astype(str).to_numpy()
    target_corpora = [c for c in CORPORA if c != args.source]

    # Select C separately for each alpha using source CV only. Alpha values are
    # fixed in advance; their target scores are not used to choose a value.
    selected: dict[str, dict[float, tuple[float, float]]] = {}
    c_rows = []
    for mode in ("narrative", "random_control"):
        table_path = source_dir / f"source_selection_{mode}.csv"
        selected[mode] = {}
        for alpha in (0.0, *POSITIVE_ALPHAS):
            if alpha == 0.0:
                # A shared raw baseline, guaranteed to match the original model.
                c, cv_ba = source_c_for_alpha(source_dir / "source_selection_raw.csv", 0.0)
            else:
                c, cv_ba = source_c_for_alpha(table_path, alpha)
            selected[mode][alpha] = (c, cv_ba)
            c_rows.append({"backbone": args.backbone, "source": args.source, "mode": mode,
                           "alpha": alpha, "source_selected_C": c,
                           "source_cv_balanced_accuracy": cv_ba,
                           "selection_data": "grouped source CV only"})
    pd.DataFrame(c_rows).to_csv(out / "source_selected_C_by_alpha.csv", index=False)

    all_metric_rows, all_boot_rows = [], []
    with threadpool_limits(limits=args.threads):
        models: dict[tuple[str, float], object] = {}
        for mode in ("narrative", "random_control"):
            for alpha in (0.0, *POSITIVE_ALPHAS):
                C, _ = selected[mode][alpha]
                if alpha == 0.0:
                    X_fit = source_X
                else:
                    X_fit = transform_features(
                        source_X, source_groups, mode, alpha,
                        donor_X=source_X if mode == "random_control" else None,
                        donor_groups=source_groups if mode == "random_control" else None,
                        seed=args.seed + int(alpha * 1000),
                    )
                model = make_model(C, args.seed)
                model.fit(X_fit, source_y)
                models[(mode, alpha)] = model

        for corpus_index, corpus in enumerate(target_corpora):
            meta, X = load_corpus(corpus, args.backbone)
            y = meta["pred_label"].astype(str).to_numpy()
            groups = meta["accident_id"].astype(str).to_numpy()
            preds_by_mode: dict[str, dict[str, np.ndarray]] = {}
            for mode in ("narrative", "random_control"):
                preds_by_mode[mode] = {}
                for alpha in (0.0, *POSITIVE_ALPHAS):
                    # Alpha zero is the same raw baseline for both modes.
                    if alpha == 0.0:
                        pred = models[(mode, alpha)].predict(X)
                    else:
                        X_test = transform_features(
                            X, groups, mode, alpha,
                            donor_X=X if mode == "random_control" else None,
                            donor_groups=groups if mode == "random_control" else None,
                            seed=args.seed + 10000 + corpus_index * 100 + int(alpha * 1000),
                        )
                        pred = models[(mode, alpha)].predict(X_test)
                    key = "raw" if alpha == 0.0 else f"alpha_{alpha:g}"
                    preds_by_mode[mode][key] = pred
                    C, source_cv_ba = selected[mode][alpha]
                    all_metric_rows.append({
                        "backbone": args.backbone, "source": args.source, "target": corpus,
                        "mode": mode, "alpha": alpha, "source_selected_C": C,
                        "source_cv_balanced_accuracy": source_cv_ba,
                        "n_units": len(y), "n_accidents": int(pd.Series(groups).nunique()),
                        **score(y, pred),
                    })
                    pd.DataFrame({"accident_id": groups, "true_role": y,
                                  "predicted_role": pred}).to_csv(
                        out / f"predictions_{corpus}_{mode}_alpha_{alpha:g}.csv", index=False
                    )
                boot = accident_bootstrap(y, preds_by_mode[mode], groups,
                                          n_bootstrap=args.bootstrap,
                                          seed=args.seed + 20000 + corpus_index * 100 + (0 if mode == "narrative" else 1))
                boot.insert(0, "mode", mode)
                boot.insert(0, "target", corpus)
                boot.insert(0, "source", args.source)
                boot.insert(0, "backbone", args.backbone)
                all_boot_rows.append(boot)

    pd.DataFrame(all_metric_rows).to_csv(out / "target_diagnostic_metrics.csv", index=False)
    pd.concat(all_boot_rows, ignore_index=True).to_csv(out / "target_diagnostic_paired_bootstrap.csv", index=False)
    manifest = {
        "backbone": args.backbone, "source": args.source, "target_corpora": target_corpora,
        "positive_alphas_diagnostic_only": list(POSITIVE_ALPHAS),
        "target_scores_used_for_selection": False,
        "classifier_C_selected_from": "source_selection_{mode}.csv grouped source CV",
        "baseline": "alpha=0; same raw embeddings and source-selected logistic regression",
        "bootstrap": {"resamples": args.bootstrap, "unit": "accident_id", "paired": True},
        "wall_time_seconds": time.time() - started,
    }
    (out / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Diagnostic terminé: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--source", choices=("btp", "metallurgie"), required=True)
    parser.add_argument("--input", default="output/narrative_residualization")
    parser.add_argument("--output", default="output/narrative_residualization_target_diagnostic")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap doit être >= 100")
    return args


if __name__ == "__main__":
    run(parse_args())
