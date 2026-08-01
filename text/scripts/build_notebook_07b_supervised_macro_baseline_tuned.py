"""Génère notebooks/07b_supervised_macro_baseline_tuned.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "07b_supervised_macro_baseline_tuned.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def py(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


INTRO = r"""
# 07b — Baseline supervisée tunée (embedding brut Qwen)

## Objectif

Même protocole que le notebook **07**, avec une **recherche d'hyperparamètres**
par famille de classifieur (LR, Random Forest, XGBoost) sur **BTP** (GroupKFold),
puis évaluation OOD avec les **meilleurs params**.

Comparaison explicite **défaut (07)** vs **tuné (07b)** sur la balanced accuracy CV
et la généralisation hors domaine.

## Protocole

1. Grille compacte par modèle (`configs/tuning/supervised_macro_baseline_grid.yaml`)
2. Sélection : `balanced_accuracy` moyenne en CV GroupKFold (`accident_id`)
3. Réentraînement 100 % BTP avec les meilleurs params → test OOD
4. Table **default vs tuned** + synthèse cross-domain

## Lancer le tuning (hors notebook, recommandé)

```bash
python scripts/tune_supervised_macro_baseline.py \
  --config configs/tuning/supervised_macro_baseline_grid.yaml
```

## Sorties

```
output_test/metallurgie/supervised_baseline/tuning/
├── grid_summary.csv
└── best_combo.json

output_test/<corpus>/supervised_baseline_tuned/
├── cv/ …
├── transfer/ …
└── cross_domain_generalization.csv   # sous cv_corpus
```

Le notebook **07** (`supervised_baseline/`) n'est **pas** écrasé.
"""


def main() -> None:
    cells = [
        md(INTRO),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.notebook_viz import plot_fsp_confusion_heatmap
from macro_transfer.supervised_baseline import (
    load_cached_cv_results,
    load_cached_all_models_test_results,
    load_ood_balanced_accuracy_by_corpus,
    load_supervised_run_manifest,
    supervised_baseline_output_dir,
    summarize_cross_domain_generalization,
)
from macro_transfer.supervised_baseline_tuning import (
    build_tuned_registry_from_best_rows,
    compare_default_vs_tuned_cv,
    load_tuning_artifacts,
    run_supervised_baseline_tuning,
    select_best_row_per_model,
    supervised_baseline_tuned_output_dir,
    supervised_baseline_tuning_dir,
)
from safer_core.io import load_yaml
from safer_core.test_corpus import resolve_test_corpus

METHOD_NAME = "supervised_macro_baseline_tuned"
N_FOLDS = 7
SEED = 42
SELECTION_METRIC = "balanced_accuracy"
DISPLAY_METRICS = ("accuracy", "balanced_accuracy")
TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
CV_CORPUS = "metallurgie"

# True → relance toute la grille (+ OOD) même si le cache existe
RESTIMATE = False

TUNE_CONFIG = "configs/tuning/supervised_macro_baseline_grid.yaml"
TUNING_DIR = supervised_baseline_tuning_dir(cv_corpus=CV_CORPUS, anchor=TEXT_ROOT)
DEFAULT_CV_DIR = supervised_baseline_output_dir(CV_CORPUS, anchor=TEXT_ROOT)
TUNED_CV_DIR = supervised_baseline_tuned_output_dir(CV_CORPUS, anchor=TEXT_ROOT)

print("Tuning dir :", TUNING_DIR)
print("Baseline 07 :", DEFAULT_CV_DIR)
print("Tuned 07b  :", TUNED_CV_DIR)
print("RESTIMATE :", RESTIMATE)
sns.set_theme(style="whitegrid")
"""
        ),
        md(
            """
## 1 — Tuning (ou rechargement du cache)

Si `RESTIMATE=True` ou si `grid_summary.csv` est absent, lance la grille YAML.
Sinon recharge `grid_summary.csv` + `best_combo.json`.
"""
        ),
        py(
            r"""
cache_ready = (TUNING_DIR / "grid_summary.csv").is_file() and (TUNING_DIR / "best_combo.json").is_file()
tuned_ready = (TUNED_CV_DIR / "cv" / "cv_summary.csv").is_file()

if RESTIMATE or not cache_ready or not tuned_ready:
    print("Lancement du tuning (peut être long : ~24–27 combos × 7 folds)…")
    result = run_supervised_baseline_tuning(TUNE_CONFIG, anchor=TEXT_ROOT)
    grid_summary = result["grid_summary"]
    best_payload = result["best_payload"]
    cv_summary_tuned = result["cv_summary"]
    best_model = result["best_model"]
    tuned_registry = result["tuned_registry"]
    cross_tuned = result["cross_domain"]
else:
    print("Cache tuning + résultats tunés trouvés — pas de réentraînement.")
    grid_summary, best_payload = load_tuning_artifacts(TUNING_DIR)
    cv_summary_tuned = pd.read_csv(TUNED_CV_DIR / "cv" / "cv_summary.csv")
    best_model = str(best_payload.get("best_model") or load_supervised_run_manifest(TUNED_CV_DIR).get("best_model"))
    best_by_model = {
        mk: {
            "best_params": json.dumps(info.get("params") or {}),
            "combo_id": info.get("combo_id"),
            "selection_score": info.get("selection_score"),
            "model": mk,
        }
        for mk, info in (best_payload.get("best_by_model") or {}).items()
    }
    tuned_registry = build_tuned_registry_from_best_rows(best_by_model)
    cross_path = TUNED_CV_DIR / "cross_domain_generalization.csv"
    cross_tuned = pd.read_csv(cross_path) if cross_path.is_file() else pd.DataFrame()

print("Meilleur modèle (CV tunée) :", best_model)
print("Score :", best_payload.get("best_selection_score"))
display(pd.DataFrame(best_payload.get("best_by_model") or {}).T)
"""
        ),
        md("## 2 — Grille : top combos par `selection_score`"),
        py(
            r"""
ordered = grid_summary.sort_values("selection_score", ascending=False, kind="mergesort")
show_cols = [
    c
    for c in (
        "model",
        "combo_id",
        "selection_score",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_accuracy",
        "best_params",
    )
    if c in ordered.columns
]
display(ordered[show_cols].head(20))

fig, ax = plt.subplots(figsize=(10, 4))
top = ordered.head(12).copy()
top["label"] = top["model"].astype(str) + "\\n" + top["combo_id"].astype(str).str.slice(0, 28)
sns.barplot(data=top, x="label", y="selection_score", hue="model", dodge=False, ax=ax)
ax.set_ylabel("CV balanced accuracy")
ax.set_xlabel("")
ax.set_title("Top hyperparameter combos (BTP GroupKFold)")
plt.xticks(rotation=25, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(TUNING_DIR / "grid_top_combos.png", dpi=150)
plt.show()
"""
        ),
        md(
            """
## 3 — Défaut (07) vs tuné (07b) — CV BTP

Charge `cv_summary.csv` du notebook **07** si disponible, et le compare aux résultats tunés.
"""
        ),
        py(
            r"""
default_cv_path = DEFAULT_CV_DIR / "cv" / "cv_summary.csv"
if default_cv_path.is_file():
    cv_summary_default = pd.read_csv(default_cv_path)
    print("Baseline 07 chargée :", default_cv_path)
else:
    cv_summary_default = pd.DataFrame(columns=["model", "mean_balanced_accuracy", "std_balanced_accuracy"])
    print("Baseline 07 absente — lancez d'abord le notebook 07 pour la comparaison.")

cmp = compare_default_vs_tuned_cv(
    cv_summary_default, cv_summary_tuned, metric=SELECTION_METRIC
)
cmp_disp = cmp.copy()
for col in ("cv_ba_default", "cv_ba_tuned", "delta_ba"):
    if col in cmp_disp.columns:
        cmp_disp[col] = cmp_disp[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
display(
    cmp_disp.rename(
        columns={
            "model": "Model",
            "cv_ba_default": "CV BA default (07)",
            "cv_ba_tuned": "CV BA tuned (07b)",
            "delta_ba": "Δ BA",
        }
    )[["Model", "CV BA default (07)", "CV BA tuned (07b)", "Δ BA"]]
)

plot_cmp = cmp.dropna(subset=["cv_ba_default", "cv_ba_tuned"], how="any")
if not plot_cmp.empty:
    melt = plot_cmp.melt(
        id_vars=["model"],
        value_vars=["cv_ba_default", "cv_ba_tuned"],
        var_name="setting",
        value_name="ba",
    )
    melt["setting"] = melt["setting"].map(
        {"cv_ba_default": "Default (07)", "cv_ba_tuned": "Tuned (07b)"}
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melt, x="model", y="ba", hue="setting", ax=ax)
    ax.set_ylabel("CV balanced accuracy")
    ax.set_xlabel("Classifier")
    ax.set_ylim(0, 1)
    ax.set_title("Default vs tuned hyperparameters (BTP CV)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    (TUNED_CV_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(TUNED_CV_DIR / "figures" / "default_vs_tuned_cv.png", dpi=150)
    plt.show()
"""
        ),
        md("## 4 — Cross-domain (params tunés)"),
        py(
            r"""
if cross_tuned.empty:
    # Reconstruit depuis les métriques OOD tunées si besoin
    from macro_transfer.supervised_baseline_tuning import _load_ood_ba_tuned

    ood_ba = _load_ood_ba_tuned(TEST_CORPORA, list(tuned_registry.keys()), anchor=TEXT_ROOT)
    cross_tuned = summarize_cross_domain_generalization(
        cv_summary_tuned, ood_ba, model_keys=list(tuned_registry.keys())
    )
    TUNED_CV_DIR.mkdir(parents=True, exist_ok=True)
    cross_tuned.to_csv(TUNED_CV_DIR / "cross_domain_generalization.csv", index=False)

display(
    cross_tuned.rename(
        columns={
            "model": "Model",
            "cv_ba": "CV ± std (BTP)",
            "ba_ood_avg": "BA OOD Avg",
            "ba_ood_worst": "BA OOD Worst",
        }
    )[["Model", "CV ± std (BTP)", "BA OOD Avg", "BA OOD Worst"]]
)

# Comparaison OOD default vs tuned (si 07 disponible)
rows = []
for corpus_id in TEST_CORPORA:
    def_path = (
        supervised_baseline_output_dir(corpus_id, anchor=TEXT_ROOT)
        / "transfer"
        / "all_models_test_metrics.csv"
    )
    tun_path = (
        supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
        / "transfer"
        / "all_models_test_metrics.csv"
    )
    if not (def_path.is_file() and tun_path.is_file()):
        continue
    d = pd.read_csv(def_path).set_index("model")
    t = pd.read_csv(tun_path).set_index("model")
    for model in sorted(set(d.index) & set(t.index)):
        rows.append(
            {
                "corpus": resolve_test_corpus(corpus_id, anchor=TEXT_ROOT).display_name,
                "model": model,
                "ba_default": float(d.loc[model, "balanced_accuracy"]),
                "ba_tuned": float(t.loc[model, "balanced_accuracy"]),
                "delta": float(t.loc[model, "balanced_accuracy"]) - float(d.loc[model, "balanced_accuracy"]),
            }
        )

ood_cmp = pd.DataFrame(rows)
if ood_cmp.empty:
    print("Comparaison OOD incomplete — lancez le notebook 07 puis le tuning 07b.")
else:
    display(ood_cmp.sort_values(["corpus", "model"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    melt = ood_cmp.melt(
        id_vars=["corpus", "model"],
        value_vars=["ba_default", "ba_tuned"],
        var_name="setting",
        value_name="ba",
    )
    melt["setting"] = melt["setting"].map(
        {"ba_default": "Default (07)", "ba_tuned": "Tuned (07b)"}
    )
    melt["x"] = melt["corpus"].astype(str) + " | " + melt["model"].astype(str)
    sns.barplot(data=melt, x="x", y="ba", hue="setting", ax=ax)
    ax.set_ylabel("OOD balanced accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Default vs tuned — OOD test corpora")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    (TUNED_CV_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(TUNED_CV_DIR / "figures" / "default_vs_tuned_ood.png", dpi=150)
    plt.show()
"""
        ),
        md("## 5 — Matrices de confusion (meilleur modèle tuné, par corpus OOD)"),
        py(
            r"""
for corpus_id in TEST_CORPORA:
    out_dir = supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        preds_by_model, metrics_by_model, summary = load_cached_all_models_test_results(
            out_dir, list(tuned_registry.keys()), macros=list(MACRO_NAMES)
        )
    except FileNotFoundError as exc:
        print(f"{corpus_id}: pas de cache tuné ({exc})")
        continue
    spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT)
    print(f"\\n=== {spec.display_name} ({corpus_id}) — best={best_model} ===")
    display(summary[[c for c in summary.columns if c in ("model", *DISPLAY_METRICS)]])
    m = metrics_by_model.get(best_model, {})
    if "_confusion_matrix" not in m:
        continue
    cm = np.asarray(m["_confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=MACRO_NAMES, columns=MACRO_NAMES)
    plot_fsp_confusion_heatmap(
        cm_df,
        fig_dir=fig_dir,
        title=f"Confusion OOD (tuned) — {corpus_id} — {best_model}",
        filename=f"confusion_test_tuned_{best_model}.png",
    )
    display(cm_df)
"""
        ),
        md("## Artefacts"),
        py(
            r"""
expected = [
    TUNING_DIR / "grid_summary.csv",
    TUNING_DIR / "best_combo.json",
    TUNED_CV_DIR / "cv" / "cv_summary.csv",
    TUNED_CV_DIR / "cross_domain_generalization.csv",
]
for corpus_id in TEST_CORPORA:
    out_dir = supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
    expected.append(out_dir / "transfer" / "all_models_test_metrics.csv")
print("Artefacts :")
for p in expected:
    print(" ", p, "→", "OK" if p.is_file() else "absent")
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {NB_PATH}")


if __name__ == "__main__":
    main()
