"""Génère le notebook de lecture des réplications multi-seeds et bootstrap."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "10_replication_uncertainty_results.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    source = text.strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": "md-" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:12],
        "source": [line + "\n" for line in source.splitlines()],
    }


def py(text: str, cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in text.strip().splitlines()],
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


TITLE = """# Réplications multi-seeds et incertitude sur les corpus cibles

Notebook **lecture seule** pour comparer SoftTriple, supervised contrastive et
cross-entropy après les réplications BTP. Il ne relance jamais un entraînement.

Prérequis : les 15 runs sont terminés puis l'analyse locale a été exécutée :

```bash
cd text
python scripts/analyze_replications.py
```

L'analyse rééchantillonne des **accidents complets** (`accident_id`) et produit
des intervalles bootstrap séparés pour métallurgie, caou et Nicollin.

## Lecture des incertitudes

- **Écart-type entre seeds** : variabilité liée à l'entraînement.
- **IC bootstrap à 95 %** : incertitude liée à l'échantillonnage du corpus cible.
- **Différence appariée** : les deux méthodes sont évaluées sur les mêmes
  accidents rééchantillonnés. Un IC qui contient zéro ne permet pas de conclure
  à un avantage stable.
"""

PARAMETERS = """# --- Paramètres ---
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = TEXT_ROOT
ANALYSIS_DIR = ROOT / "output" / "replication_analysis"
BASELINE_ANALYSIS_DIR = ROOT / "output" / "baseline_paired_bootstrap"
BASELINE_PAIRED_PATH = BASELINE_ANALYSIS_DIR / "paired_bootstrap_differences.csv"
FIGURES_DIR = ANALYSIS_DIR / "figures_notebook"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRIC = "balanced_accuracy"  # "balanced_accuracy" ou "macro_f1"
CORPORA_ORDER = ["metallurgie", "caou", "nicollin"]
MODEL_LABELS = {
    "softtriple_full_yes": "SoftTriple",
    "supcon_full_yes": "Supervised contrastive",
    "cross_entropy_full_yes": "Cross-entropy",
}
MODEL_ORDER = list(MODEL_LABELS)
BASELINE_MODEL_LABELS = {
    "frozen_embeddings_lr": "Frozen embeddings + LR",
    "tfidf_lr": "TF-IDF + LR",
}
PALETTE = {
    "SoftTriple": "#0072B2",
    "Supervised contrastive": "#009E73",
    "Cross-entropy": "#D55E00",
}

# Labels and order used only for the two paper-ready outputs below.
PAPER_CORPUS_LABELS = {
    "metallurgie": "Metallurgy",
    "caou": "Chemicals and plastics",
    "nicollin": "Company corpus",
}
PAPER_COMPARISON_ORDER = [
    ("softtriple_full_yes", "cross_entropy_full_yes"),
    ("softtriple_full_yes", "supcon_full_yes"),
    ("supcon_full_yes", "cross_entropy_full_yes"),
]
PAPER_FIGURE_DPI = 300

sns.set_theme(style="whitegrid")
print("Dossier analyse :", ANALYSIS_DIR)
print("Comparaison des représentations :", BASELINE_PAIRED_PATH)
print("Métrique affichée :", METRIC)
"""

CHECK = """# --- Vérification des artefacts ---
NEURAL_REQUIRED = {
    "scores par seed": ANALYSIS_DIR / "seed_scores.csv",
    "IC bootstrap": ANALYSIS_DIR / "bootstrap_ci.csv",
    "différences appariées": ANALYSIS_DIR / "paired_bootstrap_differences.csv",
    "manifest": ANALYSIS_DIR / "analysis_manifest.json",
}
missing = []
for label, path in NEURAL_REQUIRED.items():
    exists = path.is_file()
    print(f"[{'OK' if exists else 'ABSENT'}] {label}: {path}")
    if not exists:
        missing.append(path)
if missing:
    print("\\nExécuter d'abord : python scripts/analyze_replications.py")
else:
    print("\\nTous les artefacts sont disponibles.")

baseline_available = BASELINE_PAIRED_PATH.is_file()
print(
    f"[{'OK' if baseline_available else 'ABSENT'}] comparaison représentations : "
    f"{BASELINE_PAIRED_PATH}"
)
if not baseline_available:
    print("Exécuter pour le panneau (a) : python scripts/analyze_baseline_paired_bootstrap.py")
"""

LOAD = """# --- Chargement ---
if missing:
    raise FileNotFoundError("Analyse des réplications absente; voir la cellule précédente.")

seed_scores = pd.read_csv(ANALYSIS_DIR / "seed_scores.csv")
bootstrap_ci = pd.read_csv(ANALYSIS_DIR / "bootstrap_ci.csv")
paired = pd.read_csv(ANALYSIS_DIR / "paired_bootstrap_differences.csv")

for frame in (seed_scores, bootstrap_ci, paired):
    for column in ("model", "model_a", "model_b"):
        if column in frame.columns:
            frame[column + "_label"] = frame[column].map(MODEL_LABELS).fillna(frame[column])

seed_view = seed_scores.query("metric == @METRIC").copy()
ci_view = bootstrap_ci.query("metric == @METRIC").copy()
paired_view = paired.query("metric == @METRIC").copy()
for frame in (seed_view, ci_view, paired_view):
    if "corpus" in frame.columns:
        frame["corpus"] = pd.Categorical(frame["corpus"], CORPORA_ORDER, ordered=True)

print(f"Scores individuels : {len(seed_view)} lignes")
print(f"IC bootstrap : {len(ci_view)} lignes")
print(f"Comparaisons appariées : {len(paired_view)} lignes")
"""

SEED_TABLE = """## 1 — Scores de chaque seed

Chaque ligne est l'évaluation complète d'un modèle entraîné avec une seed sur
un corpus cible. La dernière colonne donne la moyenne et l'écart-type entre
seeds; cet écart-type ne remplace pas l'intervalle bootstrap.
"""

SEED_TABLE_CODE = """seed_table = (
    seed_view.assign(model=lambda x: x["model_label"])
    .pivot_table(index=["corpus", "model"], columns="training_seed", values="score", aggfunc="first")
    .reindex(columns=sorted(seed_view["training_seed"].unique()))
)
seed_table["mean_across_seeds"] = seed_table.mean(axis=1)
seed_table["std_across_seeds"] = seed_table.iloc[:, :-1].std(axis=1, ddof=1)
display(seed_table.style.format("{:.3f}", na_rep="—"))
"""

SEED_PLOT = """## 2 — Dispersion entre seeds

Les points correspondent aux runs individuels. Leur dispersion reflète la
variabilité de l'entraînement avec des partitions BTP inchangées.
"""

SEED_PLOT_CODE = """plot_data = seed_view.copy()
plot_data["model"] = plot_data["model_label"]
fig, axes = plt.subplots(1, len(CORPORA_ORDER), figsize=(5 * len(CORPORA_ORDER), 4), sharey=True)
for axis, corpus in zip(np.atleast_1d(axes), CORPORA_ORDER):
    subset = plot_data[plot_data["corpus"] == corpus]
    sns.stripplot(data=subset, x="model", y="score", order=list(MODEL_LABELS.values()),
                  palette=PALETTE, size=8, ax=axis)
    axis.set_title(corpus.capitalize())
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylim(0, 1)
axes[0].set_ylabel(METRIC.replace("_", " "))
fig.tight_layout()
path = FIGURES_DIR / f"seed_dispersion_{METRIC}.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print("Figure :", path)
plt.show()
"""

CI_TABLE = """## 3 — Moyenne entre seeds et IC bootstrap à 95 %

Les bornes d'IC sont calculées en rééchantillonnant les `accident_id` avec
remise. Elles répondent à une question différente de l'écart-type entre seeds :
elles indiquent à quel point le score peut varier si le corpus cible changeait.
"""

CI_TABLE_CODE = """ci_table = ci_view[[
    "corpus", "model_label", "mean_across_seeds", "std_across_seeds", "ci_low", "ci_high",
    "n_accidents", "n_units", "n_resamples",
]].rename(columns={
    "model_label": "model", "mean_across_seeds": "mean", "std_across_seeds": "seed_std",
})
display(ci_table.sort_values(["corpus", "mean"], ascending=[True, False]).style.format({
    "mean": "{:.3f}", "seed_std": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}",
}, na_rep="—"))
"""

CI_PLOT = """## 4 — Intervalles de confiance par corpus
"""

CI_PLOT_CODE = """fig, axes = plt.subplots(1, len(CORPORA_ORDER), figsize=(5 * len(CORPORA_ORDER), 4), sharey=True)
for axis, corpus in zip(np.atleast_1d(axes), CORPORA_ORDER):
    subset = ci_view[ci_view["corpus"] == corpus].copy()
    subset["model_label"] = pd.Categorical(subset["model_label"], list(MODEL_LABELS.values()), ordered=True)
    subset = subset.sort_values("model_label")
    x = np.arange(len(subset))
    means = subset["mean_across_seeds"].to_numpy()
    low = subset["ci_low"].to_numpy()
    high = subset["ci_high"].to_numpy()
    colors = [PALETTE[label] for label in subset["model_label"]]
    axis.errorbar(x, means, yerr=[means - low, high - means], fmt="none", ecolor="#333333", capsize=5)
    axis.scatter(x, means, s=70, c=colors, zorder=3)
    axis.set_xticks(x, subset["model_label"], rotation=35, ha="right")
    axis.set_title(corpus.capitalize())
    axis.set_ylim(0, 1)
axes[0].set_ylabel(METRIC.replace("_", " "))
fig.tight_layout()
path = FIGURES_DIR / f"bootstrap_intervals_{METRIC}.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print("Figure :", path)
plt.show()
"""

PAIRED_TABLE = """## 5 — Comparaisons appariées entre méthodes

`difference_a_minus_b` est la différence de score moyenne entre les seeds. La
comparaison est **non concluante** lorsque l'IC bootstrap de la différence
contient zéro.
"""

PAIRED_CODE = """paired_table = paired_view[[
    "corpus", "model_a_label", "model_b_label", "difference_a_minus_b",
    "ci_low", "ci_high", "ci_excludes_zero", "n_resamples",
]].copy()
paired_table["interpretation"] = np.where(
    paired_table["ci_excludes_zero"],
    "IC exclut 0", "IC contient 0 : non concluant",
)
paired_table = paired_table.rename(columns={"model_a_label": "model_a", "model_b_label": "model_b"})
display(paired_table.sort_values(["corpus", "model_a", "model_b"]).style.format({
    "difference_a_minus_b": "{:+.3f}", "ci_low": "{:+.3f}", "ci_high": "{:+.3f}",
}, na_rep="—"))
"""

PAPER_SECTION = """## For the paper

This section creates the results intended for the manuscript. The table uses
only strategies with complete multi-seed replications. The figure combines a
paired comparison of the two fixed representations with the paired comparisons
of the three adapted strategies.

### Table — stability across training seeds

Each entry is **mean ± standard deviation across the five training seeds**,
expressed in percentage points. `OOD average` is computed for each seed as the
mean over the three target corpora, then summarized across seeds. `OOD worst`
is the minimum target score for each seed, then summarized across seeds.
"""

PAPER_TABLE_CODE = """PAPER_METRIC = "balanced_accuracy"
paper_scores = seed_scores.loc[seed_scores["metric"].eq(PAPER_METRIC)].copy()
paper_models = [model for model in MODEL_ORDER if model in set(paper_scores["model"])]
if not paper_models:
    raise ValueError("No replicated model is available for the paper table.")

paper_by_seed = (
    paper_scores.loc[paper_scores["model"].isin(paper_models)]
    .pivot(index=["model", "training_seed"], columns="corpus", values="score")
    .reindex(columns=CORPORA_ORDER)
)
if paper_by_seed.isna().any().any():
    raise ValueError("Incomplete seed × corpus coverage: cannot build the paper table.")
paper_by_seed["ood_average"] = paper_by_seed.mean(axis=1)
paper_by_seed["ood_worst"] = paper_by_seed.min(axis=1)

paper_numeric = paper_by_seed.groupby(level="model").agg(["mean", "std"])
paper_numeric = paper_numeric.reindex(paper_models)
paper_columns = CORPORA_ORDER + ["ood_average", "ood_worst"]
paper_output_labels = {
    **PAPER_CORPUS_LABELS,
    "ood_average": "OOD average",
    "ood_worst": "OOD worst",
}
paper_table = pd.DataFrame(index=[MODEL_LABELS[model] for model in paper_models])
paper_table.index.name = "Strategy"
for column in paper_columns:
    mean = paper_numeric[(column, "mean")].to_numpy() * 100
    std = paper_numeric[(column, "std")].to_numpy() * 100
    label = paper_output_labels[column]
    paper_table[label] = [f"{value:.1f} ± {spread:.1f}" for value, spread in zip(mean, std)]

display(paper_table)
print("Metric: balanced accuracy. Values are percentage points; ± is the SD across training seeds.")
"""

PAPER_FOREST = """### Figure - paired bootstrap comparisons

Panel (a) compares frozen embeddings + LR with TF-IDF + LR. Panel (b) compares
SoftTriple, supervised contrastive and cross-entropy. Each point is the
estimated difference in balanced accuracy, in percentage points, and each
horizontal bar is its 95% paired bootstrap confidence interval.
"""

PAPER_FOREST_CODE = """if not BASELINE_PAIRED_PATH.is_file():
    raise FileNotFoundError(
        "Baseline paired bootstrap missing. Run: "
        "python scripts/analyze_baseline_paired_bootstrap.py"
    )

baseline_paired = pd.read_csv(BASELINE_PAIRED_PATH)
required_baseline_columns = {
    "corpus", "model_a", "model_b", "metric", "difference_a_minus_b", "ci_low", "ci_high",
}
missing_baseline_columns = required_baseline_columns - set(baseline_paired.columns)
if missing_baseline_columns:
    raise ValueError(f"Baseline paired bootstrap is incomplete: {sorted(missing_baseline_columns)}")

representation_pair = ("frozen_embeddings_lr", "tfidf_lr")
representation_paired = baseline_paired.loc[
    baseline_paired["metric"].eq("balanced_accuracy")
].copy()
representation_paired = representation_paired.loc[
    representation_paired.apply(
        lambda row: (row["model_a"], row["model_b"]) == representation_pair,
        axis=1,
    )
].copy()
representation_paired["comparison"] = [
    f"{BASELINE_MODEL_LABELS[model_a]} - {BASELINE_MODEL_LABELS[model_b]}"
    for model_a, model_b in zip(
        representation_paired["model_a"], representation_paired["model_b"]
    )
]
if len(representation_paired) != len(CORPORA_ORDER):
    raise ValueError("Incomplete baseline paired bootstrap results: cannot build panel (a).")

paper_paired = paired.loc[paired["metric"].eq("balanced_accuracy")].copy()
paper_paired = paper_paired.loc[
    paper_paired.apply(lambda row: (row["model_a"], row["model_b"]) in PAPER_COMPARISON_ORDER, axis=1)
].copy()
order_rank = {pair: rank for rank, pair in enumerate(PAPER_COMPARISON_ORDER)}
paper_paired["comparison_rank"] = [
    order_rank[(model_a, model_b)]
    for model_a, model_b in zip(paper_paired["model_a"], paper_paired["model_b"])
]
paper_paired["comparison"] = [
    f"{MODEL_LABELS[model_a]} - {MODEL_LABELS[model_b]}"
    for model_a, model_b in zip(paper_paired["model_a"], paper_paired["model_b"])
]
expected_rows = len(CORPORA_ORDER) * len(PAPER_COMPARISON_ORDER)
if len(paper_paired) != expected_rows:
    raise ValueError("Incomplete neural paired bootstrap results: cannot build panel (b).")

limit = 1.10 * max(
    representation_paired["ci_low"].abs().max(),
    representation_paired["ci_high"].abs().max(),
    paper_paired["ci_low"].abs().max(),
    paper_paired["ci_high"].abs().max(),
) * 100
from matplotlib.ticker import MaxNLocator

fig, axes = plt.subplots(
    2, len(CORPORA_ORDER), figsize=(17, 8.0), sharex=True, squeeze=False
)

def style_axis(axis):
    axis.axvline(0, color="#202020", lw=1.8, zorder=2)
    axis.set_xlim(-limit, limit)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="x", color="#bdbdbd", linewidth=0.8, alpha=0.8)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    axis.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.8)

for column, corpus in enumerate(CORPORA_ORDER):
    representation_axis = axes[0, column]
    representation_subset = representation_paired.loc[
        representation_paired["corpus"].eq(corpus)
    ]
    estimate = representation_subset["difference_a_minus_b"].to_numpy() * 100
    low = representation_subset["ci_low"].to_numpy() * 100
    high = representation_subset["ci_high"].to_numpy() * 100
    representation_axis.errorbar(
        estimate, [0], xerr=[estimate - low, high - estimate], fmt="o",
        color="#6A3D9A", ecolor="#6A3D9A", elinewidth=2.2, capsize=5,
        capthick=2.2, markersize=10, markeredgecolor="white", markeredgewidth=1.0,
        zorder=3,
    )
    representation_axis.set_yticks([0], representation_subset["comparison"])
    representation_axis.set_ylim(-0.7, 0.7)
    representation_axis.set_title(PAPER_CORPUS_LABELS[corpus], fontweight="bold")
    style_axis(representation_axis)

    strategy_axis = axes[1, column]
    strategy_subset = paper_paired.loc[
        paper_paired["corpus"].eq(corpus)
    ].sort_values("comparison_rank")
    y = np.arange(len(strategy_subset))
    estimate = strategy_subset["difference_a_minus_b"].to_numpy() * 100
    low = strategy_subset["ci_low"].to_numpy() * 100
    high = strategy_subset["ci_high"].to_numpy() * 100
    strategy_axis.errorbar(
        estimate, y, xerr=[estimate - low, high - estimate], fmt="o",
        color="#0072B2", ecolor="#0072B2", elinewidth=2.2, capsize=5,
        capthick=2.2, markersize=10, markeredgecolor="white", markeredgewidth=1.0,
        zorder=3,
    )
    strategy_axis.set_yticks(y, strategy_subset["comparison"])
    strategy_axis.invert_yaxis()
    style_axis(strategy_axis)

fig.text(0.015, 0.955, "(a) Representation comparison", fontweight="bold", fontsize=12)
fig.text(0.015, 0.495, "(b) Adapted-strategy comparison", fontweight="bold", fontsize=12)
fig.supxlabel("Difference in balanced accuracy (percentage points)", y=0.035)
fig.subplots_adjust(left=0.30, right=0.985, bottom=0.14, top=0.91, hspace=0.48, wspace=0.24)
paper_forest_png = FIGURES_DIR / "paper_paired_bootstrap_balanced_accuracy.png"
paper_forest_pdf = FIGURES_DIR / "paper_paired_bootstrap_balanced_accuracy.pdf"
fig.savefig(paper_forest_png, dpi=PAPER_FIGURE_DPI, bbox_inches="tight")
fig.savefig(paper_forest_pdf, bbox_inches="tight")
print("Paper figure (PNG):", paper_forest_png)
print("Paper figure (PDF):", paper_forest_pdf)
plt.show()
"""

PAPER_EXPORT = """paper_table.to_csv(FIGURES_DIR / "paper_seed_stability_balanced_accuracy.csv")
paper_numeric.to_csv(FIGURES_DIR / "paper_seed_stability_balanced_accuracy_numeric.csv")
paper_paired.to_csv(FIGURES_DIR / "paper_paired_bootstrap_balanced_accuracy.csv", index=False)
representation_paired.to_csv(
    FIGURES_DIR / "paper_representation_paired_bootstrap_balanced_accuracy.csv", index=False
)
print("Paper-ready table and figure data exported to:", FIGURES_DIR)
"""
EXPORT = """## Exports for notebook tables
"""

EXPORT_CODE = """seed_table.reset_index().to_csv(FIGURES_DIR / f"seed_table_{METRIC}.csv", index=False)
ci_table.to_csv(FIGURES_DIR / f"bootstrap_ci_table_{METRIC}.csv", index=False)
paired_table.to_csv(FIGURES_DIR / f"paired_differences_{METRIC}.csv", index=False)
print("Exports notebook :", FIGURES_DIR)
"""


def build_notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            md(TITLE), py(NOTEBOOK_PATH_SETUP, "bootstrap"), py(PARAMETERS, "parameters"),
            py(CHECK, "check-artifacts"), py(LOAD, "load-results"), md(SEED_TABLE),
            py(SEED_TABLE_CODE, "seed-table"), md(SEED_PLOT), py(SEED_PLOT_CODE, "seed-plot"),
            md(CI_TABLE), py(CI_TABLE_CODE, "bootstrap-table"), md(CI_PLOT),
            py(CI_PLOT_CODE, "bootstrap-plot"), md(PAIRED_TABLE), py(PAIRED_CODE, "paired-table"),
            md(PAPER_SECTION), py(PAPER_TABLE_CODE, "paper-stability-table"), md(PAPER_FOREST),
            py(PAPER_FOREST_CODE, "paper-forest-plot"), py(PAPER_EXPORT, "paper-exports"),
            md(EXPORT), py(EXPORT_CODE, "exports"),
        ],
    }


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Écrit :", NB_PATH)


if __name__ == "__main__":
    main()
