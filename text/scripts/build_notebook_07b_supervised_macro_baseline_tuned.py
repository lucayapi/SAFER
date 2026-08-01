"""Génère notebooks/07b_supervised_macro_baseline_tuned.ipynb (lecture résultats du job)."""

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
# 07b — Résultats du tuning baseline supervisée

Notebook **lecture seule** : affiche les sorties du job Slurm de grille HP
(LR / Random Forest / XGBoost) sur embeddings Qwen bruts.

> Ce notebook **n'entraîne pas**. Lance d'abord le job, puis exécute les cellules.

## Lancer le job (cluster)

```bash
cd ~/SAFER/text
sbatch jobs/tune_supervised_macro_baseline.sh
```

Grille : `configs/tuning/supervised_macro_baseline_grid.yaml`

## Contenu

1. Tableau récap **meilleurs hyperparamètres** + BA CV / OOD  
2. Top combos de la grille  
3. Comparaison **défaut (07)** vs **tuné (07b)**  
4. Synthèse cross-domain  
5. **Matrices de confusion** train (BTP) + OOD (tous modèles / corpus)  
6. Checklist des artefacts

## Chemins attendus

```
output_test/metallurgie/supervised_baseline/tuning/
  results_summary.csv, grid_summary.csv, best_*.json

output_test/btp/supervised_baseline_tuned/transfer/
  source_macro_predictions.csv, models/<model>/…

output_test/<corpus>/supervised_baseline_tuned/transfer/
  target_macro_predictions.csv, all_models_test_metrics.csv, models/<model>/…
```
"""


def main() -> None:
    cells = [
        md(INTRO),
        md("### Paramètres"),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.notebook_viz import plot_fsp_confusion_heatmap
from macro_transfer.supervised_baseline import (
    load_cached_all_models_test_results,
    load_supervised_run_manifest,
    supervised_baseline_output_dir,
)
from macro_transfer.supervised_baseline_tuning import (
    compare_default_vs_tuned_cv,
    load_tuning_artifacts,
    supervised_baseline_tuned_output_dir,
    supervised_baseline_tuning_dir,
)
from safer_core.test_corpus import resolve_test_corpus

SELECTION_METRIC = "balanced_accuracy"
DISPLAY_METRICS = ("accuracy", "balanced_accuracy")
TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
CV_CORPUS = "metallurgie"
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]
MACROS = list(MACRO_NAMES)

TUNING_DIR = supervised_baseline_tuning_dir(cv_corpus=CV_CORPUS, anchor=TEXT_ROOT)
DEFAULT_CV_DIR = supervised_baseline_output_dir(CV_CORPUS, anchor=TEXT_ROOT)
TUNED_CV_DIR = supervised_baseline_tuned_output_dir(CV_CORPUS, anchor=TEXT_ROOT)
TRAIN_OUT_DIR = supervised_baseline_tuned_output_dir("btp", anchor=TEXT_ROOT)
FIG_DIR = TUNED_CV_DIR / "figures_notebook"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Tuning dir :", TUNING_DIR)
print("Baseline 07 :", DEFAULT_CV_DIR)
print("Tuned 07b  :", TUNED_CV_DIR)
print("Train BTP  :", TRAIN_OUT_DIR)
print("Figures    :", FIG_DIR)
sns.set_theme(style="whitegrid")
"""
        ),
        md(
            """
## 0 — Vérification des artefacts du job

Si des fichiers manquent, relance / attends la fin de :
`sbatch jobs/tune_supervised_macro_baseline.sh`
"""
        ),
        py(
            r"""
required = {
    "grid_summary": TUNING_DIR / "grid_summary.csv",
    "best_combo": TUNING_DIR / "best_combo.json",
    "results_summary": TUNING_DIR / "results_summary.csv",
    "cv_summary_tuned": TUNED_CV_DIR / "cv" / "cv_summary.csv",
    "cross_domain": TUNED_CV_DIR / "cross_domain_generalization.csv",
    "train_preds_best": TRAIN_OUT_DIR / "transfer" / "source_macro_predictions.csv",
}
for corpus_id in TEST_CORPORA:
    out = supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
    required[f"ood_metrics_{corpus_id}"] = out / "transfer" / "all_models_test_metrics.csv"
    required[f"ood_preds_{corpus_id}"] = out / "transfer" / "target_macro_predictions.csv"

missing = [k for k, p in required.items() if not p.is_file()]
print("Artefacts :")
for k, p in required.items():
    print(f"  [{'OK' if p.is_file() else 'ABSENT'}] {k}: {p}")

if missing:
    raise FileNotFoundError(
        "Job de tuning incomplet — fichiers manquants : "
        + ", ".join(missing)
        + "\\nLancez : sbatch jobs/tune_supervised_macro_baseline.sh"
    )
print("\\nTous les artefacts principaux sont présents.")
"""
        ),
        md(
            """
## 1 — Tableau récapitulatif (meilleurs HP)

`results_summary.csv` : une ligne par modèle avec params retenus, BA CV et BA OOD.
"""
        ),
        py(
            r"""
grid_summary, best_payload = load_tuning_artifacts(TUNING_DIR)
best_model = str(best_payload.get("best_model") or "")
results_summary = pd.read_csv(TUNING_DIR / "results_summary.csv")

# Ordre modèles stable
if "model" in results_summary.columns:
    cat = pd.Categorical(
        results_summary["model"], categories=MODEL_ORDER, ordered=True
    )
    results_summary = results_summary.assign(model=cat).sort_values("model")

print("Meilleur modèle global (CV) :", best_model)
print("Score CV :", best_payload.get("best_selection_score"))
print()

# Hyperparams lisibles
hp_path = TUNING_DIR / "best_hyperparams.json"
if hp_path.is_file():
    hp_payload = json.loads(hp_path.read_text(encoding="utf-8"))
    hp_by_model = hp_payload.get("best_hyperparams_by_model") or {}
else:
    hp_by_model = {
        mk: (info.get("params") or {})
        for mk, info in (best_payload.get("best_by_model") or {}).items()
    }

hp_rows = []
for mk in MODEL_ORDER:
    if mk not in hp_by_model:
        continue
    hp_rows.append(
        {
            "model": mk,
            "is_best_overall": mk == best_model,
            "best_params": json.dumps(hp_by_model[mk], ensure_ascii=False, default=str),
        }
    )
display(pd.DataFrame(hp_rows))

show_cols = [
    c
    for c in (
        "model",
        "is_best_overall",
        "cv_balanced_accuracy",
        "cv_ba_std",
        "cv_accuracy",
        "ba_ood_avg",
        "ba_ood_worst",
        *[c for c in results_summary.columns if c.startswith("ba_ood_") and c not in ("ba_ood_avg", "ba_ood_worst")],
        "best_params",
    )
    if c in results_summary.columns
]
display(results_summary[show_cols])

# Barplot CV vs OOD avg
plot_df = results_summary.copy()
if {"cv_balanced_accuracy", "ba_ood_avg", "model"}.issubset(plot_df.columns):
    melt = plot_df.melt(
        id_vars=["model"],
        value_vars=["cv_balanced_accuracy", "ba_ood_avg"],
        var_name="metric",
        value_name="score",
    )
    melt["metric"] = melt["metric"].map(
        {
            "cv_balanced_accuracy": "CV BA (BTP)",
            "ba_ood_avg": "BA OOD avg",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melt, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Best HP per model — CV vs OOD")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "results_cv_vs_ood.png", dpi=150)
    plt.show()
"""
        ),
        md("## 2 — Top combos de la grille (toutes les tentatives)"),
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
print(f"{len(ordered)} combos au total")
display(ordered[show_cols].head(25))

# Best score par modèle (boxplot / strip de la grille)
if {"model", "selection_score"}.issubset(grid_summary.columns):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=grid_summary, x="model", y="selection_score", order=MODEL_ORDER, ax=ax)
    sns.stripplot(
        data=grid_summary,
        x="model",
        y="selection_score",
        order=MODEL_ORDER,
        color="0.3",
        size=3,
        ax=ax,
    )
    ax.set_ylabel("CV balanced accuracy")
    ax.set_title("Distribution des scores de grille par modèle")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "grid_score_distribution.png", dpi=150)
    plt.show()

fig, ax = plt.subplots(figsize=(11, 4))
top = ordered.head(15).copy()
top["label"] = (
    top["model"].astype(str) + "\\n" + top["combo_id"].astype(str).str.slice(0, 32)
)
sns.barplot(data=top, x="label", y="selection_score", hue="model", dodge=False, ax=ax)
ax.set_ylabel("CV balanced accuracy")
ax.set_xlabel("")
ax.set_title("Top 15 hyperparameter combos")
plt.xticks(rotation=30, ha="right", fontsize=7)
plt.tight_layout()
fig.savefig(FIG_DIR / "grid_top_combos.png", dpi=150)
plt.show()
"""
        ),
        md(
            """
## 3 — Défaut (notebook 07) vs tuné (job 07b) — CV BTP

Compare `cv_summary` du baseline non tuné si disponible.
"""
        ),
        py(
            r"""
cv_summary_tuned = pd.read_csv(TUNED_CV_DIR / "cv" / "cv_summary.csv")
default_cv_path = DEFAULT_CV_DIR / "cv" / "cv_summary.csv"
if default_cv_path.is_file():
    cv_summary_default = pd.read_csv(default_cv_path)
    print("Baseline 07 chargée :", default_cv_path)
else:
    cv_summary_default = pd.DataFrame(
        columns=["model", "mean_balanced_accuracy", "std_balanced_accuracy"]
    )
    print("Baseline 07 absente — comparaison CV partielle.")

cmp = compare_default_vs_tuned_cv(
    cv_summary_default, cv_summary_tuned, metric=SELECTION_METRIC
)
cmp_disp = cmp.copy()
for col in ("cv_ba_default", "cv_ba_tuned", "delta_ba"):
    if col in cmp_disp.columns:
        cmp_disp[col] = cmp_disp[col].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        )
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
    ax.set_ylim(0, 1)
    ax.set_title("Default vs tuned hyperparameters (BTP CV)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "default_vs_tuned_cv.png", dpi=150)
    plt.show()
"""
        ),
        md("## 4 — Synthèse cross-domain (params tunés)"),
        py(
            r"""
cross_path = TUNED_CV_DIR / "cross_domain_generalization.csv"
cross_tuned = pd.read_csv(cross_path)
cross_disp = cross_tuned.rename(
    columns={
        "model": "Model",
        "cv_ba": "CV ± std (BTP)",
        "ba_ood_avg": "BA OOD Avg",
        "ba_ood_worst": "BA OOD Worst",
    }
)
keep = [c for c in ("Model", "CV ± std (BTP)", "BA OOD Avg", "BA OOD Worst") if c in cross_disp.columns]
display(cross_disp[keep])

# Métriques OOD détaillées par corpus
ood_tables = []
for corpus_id in TEST_CORPORA:
    path = (
        supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
        / "transfer"
        / "all_models_test_metrics.csv"
    )
    df = pd.read_csv(path)
    df.insert(0, "corpus", resolve_test_corpus(corpus_id, anchor=TEXT_ROOT).display_name)
    ood_tables.append(df)
ood_all = pd.concat(ood_tables, ignore_index=True)
disp_cols = ["corpus", "model"] + [c for c in DISPLAY_METRICS if c in ood_all.columns]
display(ood_all[disp_cols].sort_values(["corpus", "model"]))

# Default vs tuned OOD (si 07 dispo)
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
    for model in sorted(set(d.index.astype(str)) & set(t.index.astype(str))):
        rows.append(
            {
                "corpus": resolve_test_corpus(corpus_id, anchor=TEXT_ROOT).display_name,
                "model": model,
                "ba_default": float(d.loc[model, "balanced_accuracy"]),
                "ba_tuned": float(t.loc[model, "balanced_accuracy"]),
                "delta": float(t.loc[model, "balanced_accuracy"])
                - float(d.loc[model, "balanced_accuracy"]),
            }
        )

ood_cmp = pd.DataFrame(rows)
if ood_cmp.empty:
    print("Comparaison OOD 07 vs 07b incomplete (lancez le notebook 07 si besoin).")
else:
    display(ood_cmp.sort_values(["corpus", "model"]))
    fig, ax = plt.subplots(figsize=(11, 4))
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
    fig.savefig(FIG_DIR / "default_vs_tuned_ood.png", dpi=150)
    plt.show()
"""
        ),
        md(
            """
## 5 — Matrices de confusion — train BTP (fit final tuné)

Predictions : `output_test/btp/supervised_baseline_tuned/transfer/models/<model>/`
"""
        ),
        py(
            r"""
def _load_source_cm(model_key: str):
    model_dir = TRAIN_OUT_DIR / "transfer" / "models" / model_key
    cm_path = model_dir / "source_confusion_matrix.csv"
    metrics_path = model_dir / "source_metrics.json"
    preds_path = model_dir / "source_macro_predictions.csv"
    if cm_path.is_file():
        return pd.read_csv(cm_path, index_col=0), metrics_path
    if preds_path.is_file():
        preds = pd.read_csv(preds_path)
        if {"true_macro", "pred_macro"}.issubset(preds.columns):
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(
                preds["true_macro"].astype(str),
                preds["pred_macro"].astype(str),
                labels=MACROS,
            )
            return pd.DataFrame(cm, index=MACROS, columns=MACROS), metrics_path
    return None, metrics_path

print("Best model (référence) :", best_model)
for model_key in MODEL_ORDER:
    cm_df, metrics_path = _load_source_cm(model_key)
    if cm_df is None:
        print(f"[skip] train confusion absente pour {model_key}")
        continue
    ba = None
    if metrics_path.is_file():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        ba = meta.get("balanced_accuracy")
    title = f"Train BTP (tuned) — {model_key}"
    if ba is not None:
        title += f" | BA={float(ba):.3f}"
    print(f"\\n### {title}")
    plot_fsp_confusion_heatmap(
        cm_df,
        fig_dir=FIG_DIR,
        title=title,
        filename=f"confusion_train_{model_key}.png",
    )
    display(cm_df)
"""
        ),
        md(
            """
## 6 — Matrices de confusion — OOD (tous modèles × corpus)

Pour chaque corpus test : tableau de métriques + heatmap par modèle.
"""
        ),
        py(
            r"""
for corpus_id in TEST_CORPORA:
    out_dir = supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
    fig_dir = out_dir / "figures_notebook"
    fig_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_test_corpus(corpus_id, anchor=TEXT_ROOT)
    try:
        preds_by_model, metrics_by_model, summary = load_cached_all_models_test_results(
            out_dir, MODEL_ORDER, macros=MACROS
        )
    except FileNotFoundError as exc:
        print(f"{corpus_id}: cache tuné incomplet ({exc})")
        continue

    print(f"\\n{'=' * 60}")
    print(f"=== {spec.display_name} ({corpus_id}) ===")
    print(f"{'=' * 60}")
    cols = ["model"] + [c for c in DISPLAY_METRICS if c in summary.columns]
    display(summary[cols])

    for model_key in MODEL_ORDER:
        m = metrics_by_model.get(model_key, {})
        if "_confusion_matrix" not in m:
            # fallback fichier
            cm_path = out_dir / "transfer" / "models" / model_key / "confusion_matrix.csv"
            if not cm_path.is_file():
                print(f"  [skip] pas de confusion pour {model_key}")
                continue
            cm_df = pd.read_csv(cm_path, index_col=0)
        else:
            cm_df = pd.DataFrame(
                np.asarray(m["_confusion_matrix"]), index=MACROS, columns=MACROS
            )
        ba = m.get("balanced_accuracy", float("nan"))
        star = " ★" if model_key == best_model else ""
        title = (
            f"OOD {spec.display_name} — {model_key}{star} | BA={float(ba):.3f}"
            if ba == ba
            else f"OOD {spec.display_name} — {model_key}{star}"
        )
        print(f"\\n### {title}")
        plot_fsp_confusion_heatmap(
            cm_df,
            fig_dir=FIG_DIR,
            title=title,
            filename=f"confusion_ood_{corpus_id}_{model_key}.png",
        )
        # Copie sous le dossier corpus
        src = FIG_DIR / f"confusion_ood_{corpus_id}_{model_key}.png"
        if src.is_file():
            import shutil

            shutil.copy(src, fig_dir / src.name)
        display(cm_df)
"""
        ),
        md("## 7 — Checklist artefacts"),
        py(
            r"""
expected = [
    TUNING_DIR / "grid_summary.csv",
    TUNING_DIR / "results_summary.csv",
    TUNING_DIR / "best_combo.json",
    TUNING_DIR / "best_hyperparams.json",
    TUNED_CV_DIR / "cv" / "cv_summary.csv",
    TUNED_CV_DIR / "cross_domain_generalization.csv",
    TRAIN_OUT_DIR / "transfer" / "source_macro_predictions.csv",
]
for model_key in MODEL_ORDER:
    expected.append(
        TRAIN_OUT_DIR / "transfer" / "models" / model_key / "source_macro_predictions.csv"
    )
for corpus_id in TEST_CORPORA:
    out_dir = supervised_baseline_tuned_output_dir(corpus_id, anchor=TEXT_ROOT)
    expected.append(out_dir / "transfer" / "all_models_test_metrics.csv")
    expected.append(out_dir / "transfer" / "target_macro_predictions.csv")
    for model_key in MODEL_ORDER:
        expected.append(
            out_dir / "transfer" / "models" / model_key / "target_macro_predictions.csv"
        )

print("Artefacts :")
n_ok = 0
for p in expected:
    ok = p.is_file()
    n_ok += int(ok)
    print(" ", "OK    " if ok else "ABSENT", p)
print(f"\\n{n_ok}/{len(expected)} fichiers présents")
print("Figures notebook :", FIG_DIR)
"""
        ),
    ]

    # Fix buggy display columns line in section 4 - I'll write clean version
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
