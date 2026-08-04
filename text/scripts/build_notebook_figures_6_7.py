"""Génère le notebook des Figures 6 et 7 pour l'article."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "11_article_figures_6_7.ipynb"

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


TITLE_MD = """# Article figures 6 and 7 — transfer performance

This notebook reproduces the two figures describing the effect of encoder-update depth,
projector inclusion and learning strategy on cross-corpus balanced accuracy.

The values below are the results supplied for the article tables. Edit the data cells if
the final training campaign produces updated values. Figures are exported as both PNG
and PDF under `output/article_figures_6_7/figures/`.

The y-axes are intentionally truncated to make differences visible. The limits are
reported in each figure caption and should be retained in the article caption.
"""


PARAMETERS_CODE = """from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = TEXT_ROOT
OUTPUT_DIR = ROOT / "output/article_figures_6_7"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.frameon": False,
})

DEPTH_ORDER = ["Last 1 layer", "Last 2 layers", "Last 3 layers", "Full encoder"]
PROJECTOR_ORDER = ["Without projector", "With projector"]
PROJECTOR_COLORS = {
    "Without projector": "#D55E00",
    "With projector": "#0072B2",
}
METHOD_ORDER = ["Cross-entropy", "Batch-hard Triplet", "Supervised contrastive", "SoftTriple"]
METHOD_COLORS = {
    "Frozen logistic regression": "#7F7F7F",
    "Cross-entropy": "#D55E00",
    "Batch-hard Triplet": "#0072B2",
    "Supervised contrastive": "#009E73",
    "SoftTriple": "#CC79A7",
}
"""


FIG6_DATA_CODE = """FIG6_ROWS = [
    {"method": "Cross-entropy", "depth": "Last 1 layer", "projector": "With projector", "ood_avg": 75.5, "ood_worst": 66.5},
    {"method": "Cross-entropy", "depth": "Last 1 layer", "projector": "Without projector", "ood_avg": 74.4, "ood_worst": 64.9},
    {"method": "Cross-entropy", "depth": "Last 2 layers", "projector": "With projector", "ood_avg": 75.6, "ood_worst": 66.2},
    {"method": "Cross-entropy", "depth": "Last 2 layers", "projector": "Without projector", "ood_avg": 68.1, "ood_worst": 61.2},
    {"method": "Cross-entropy", "depth": "Last 3 layers", "projector": "With projector", "ood_avg": 75.7, "ood_worst": 68.1},
    {"method": "Cross-entropy", "depth": "Last 3 layers", "projector": "Without projector", "ood_avg": 75.7, "ood_worst": 67.2},
    {"method": "Cross-entropy", "depth": "Full encoder", "projector": "With projector", "ood_avg": 80.4, "ood_worst": 73.7},
    {"method": "Cross-entropy", "depth": "Full encoder", "projector": "Without projector", "ood_avg": 85.5, "ood_worst": 84.4},
    {"method": "Batch-hard Triplet", "depth": "Last 1 layer", "projector": "With projector", "ood_avg": 80.4, "ood_worst": 74.7},
    {"method": "Batch-hard Triplet", "depth": "Last 1 layer", "projector": "Without projector", "ood_avg": 80.2, "ood_worst": 74.6},
    {"method": "Batch-hard Triplet", "depth": "Last 2 layers", "projector": "With projector", "ood_avg": 77.6, "ood_worst": 70.8},
    {"method": "Batch-hard Triplet", "depth": "Last 2 layers", "projector": "Without projector", "ood_avg": 78.1, "ood_worst": 71.9},
    {"method": "Batch-hard Triplet", "depth": "Last 3 layers", "projector": "With projector", "ood_avg": 54.1, "ood_worst": 44.3},
    {"method": "Batch-hard Triplet", "depth": "Last 3 layers", "projector": "Without projector", "ood_avg": 76.6, "ood_worst": 68.6},
    {"method": "Batch-hard Triplet", "depth": "Full encoder", "projector": "With projector", "ood_avg": 79.2, "ood_worst": 71.9},
    {"method": "Batch-hard Triplet", "depth": "Full encoder", "projector": "Without projector", "ood_avg": 83.0, "ood_worst": 76.4},
    {"method": "Supervised contrastive", "depth": "Last 1 layer", "projector": "With projector", "ood_avg": 79.3, "ood_worst": 72.8},
    {"method": "Supervised contrastive", "depth": "Last 1 layer", "projector": "Without projector", "ood_avg": 80.0, "ood_worst": 72.8},
    {"method": "Supervised contrastive", "depth": "Last 2 layers", "projector": "With projector", "ood_avg": 81.6, "ood_worst": 74.8},
    {"method": "Supervised contrastive", "depth": "Last 2 layers", "projector": "Without projector", "ood_avg": 80.9, "ood_worst": 74.2},
    {"method": "Supervised contrastive", "depth": "Last 3 layers", "projector": "With projector", "ood_avg": 80.2, "ood_worst": 73.3},
    {"method": "Supervised contrastive", "depth": "Last 3 layers", "projector": "Without projector", "ood_avg": 81.6, "ood_worst": 74.6},
    {"method": "Supervised contrastive", "depth": "Full encoder", "projector": "With projector", "ood_avg": 85.2, "ood_worst": 79.3},
    {"method": "Supervised contrastive", "depth": "Full encoder", "projector": "Without projector", "ood_avg": 84.5, "ood_worst": 78.5},
    {"method": "SoftTriple", "depth": "Last 1 layer", "projector": "With projector", "ood_avg": 80.0, "ood_worst": 73.3},
    {"method": "SoftTriple", "depth": "Last 1 layer", "projector": "Without projector", "ood_avg": 79.0, "ood_worst": 71.8},
    {"method": "SoftTriple", "depth": "Last 2 layers", "projector": "With projector", "ood_avg": 80.9, "ood_worst": 75.2},
    {"method": "SoftTriple", "depth": "Last 2 layers", "projector": "Without projector", "ood_avg": 79.5, "ood_worst": 70.9},
    {"method": "SoftTriple", "depth": "Last 3 layers", "projector": "With projector", "ood_avg": 80.9, "ood_worst": 74.5},
    {"method": "SoftTriple", "depth": "Last 3 layers", "projector": "Without projector", "ood_avg": 80.9, "ood_worst": 75.5},
    {"method": "SoftTriple", "depth": "Full encoder", "projector": "With projector", "ood_avg": 84.2, "ood_worst": 78.1},
    {"method": "SoftTriple", "depth": "Full encoder", "projector": "Without projector", "ood_avg": 84.5, "ood_worst": 78.7},
]

fig6_df = pd.DataFrame(FIG6_ROWS)
fig6_df["depth"] = pd.Categorical(fig6_df["depth"], categories=DEPTH_ORDER, ordered=True)
fig6_df["method"] = pd.Categorical(fig6_df["method"], categories=METHOD_ORDER, ordered=True)
fig6_df = fig6_df.sort_values(["method", "depth", "projector"])
display(fig6_df)
"""


FIG6_PLOT_CODE = """from matplotlib.lines import Line2D

fig6, axes = plt.subplots(
    2,
    4,
    figsize=(14, 7.4),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.18, "wspace": 0.16},
)

metric_specs = [
    ("ood_avg", 76.9, "(a) OOD average BA"),
    ("ood_worst", 68.9, "(b) Worst-corpus BA"),
]
x = range(len(DEPTH_ORDER))
for column, method in enumerate(METHOD_ORDER):
    method_df = fig6_df[fig6_df["method"] == method]
    for row, (metric, baseline, row_label) in enumerate(metric_specs):
        ax = axes[row, column]
        for projector in PROJECTOR_ORDER:
            series = method_df[method_df["projector"] == projector].set_index("depth").loc[DEPTH_ORDER]
            ax.plot(
                list(x),
                series[metric].to_numpy(),
                marker="o",
                linewidth=2,
                markersize=5,
                color=PROJECTOR_COLORS[projector],
            )
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2)
        ax.set_ylim(40, 90)
        ax.set_xticks(list(x), ["Last 1", "Last 2", "Last 3", "Full"])
        if row == 0:
            ax.set_title(method)
        if row == 1:
            ax.set_xlabel("Encoder-update depth")
        ax.grid(axis="y", alpha=0.3)

axes[0, 0].text(
    0.03,
    76.9 + 1.0,
    "Frozen baseline: 76.9%",
    color="#555555",
    fontsize=8,
)
axes[1, 0].text(
    0.03,
    68.9 + 1.0,
    "Frozen baseline: 68.9%",
    color="#555555",
    fontsize=8,
)
fig6.text(0.012, 0.72, "(a) OOD average BA", rotation=90, va="center", fontweight="bold")
fig6.text(0.012, 0.285, "(b) Worst-corpus BA", rotation=90, va="center", fontweight="bold")
legend_handles = [
    Line2D([0], [0], color=PROJECTOR_COLORS["Without projector"], marker="o", linewidth=2, label="No projector"),
    Line2D([0], [0], color=PROJECTOR_COLORS["With projector"], marker="o", linewidth=2, label="With projector"),
    Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2, label="Frozen logistic regression baseline"),
]
fig6.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3)
fig6.tight_layout(rect=[0.035, 0.02, 1, 0.91])
fig6.savefig(FIGURES_DIR / "figure_6_encoder_depth_projector.png", bbox_inches="tight")
fig6.savefig(FIGURES_DIR / "figure_6_encoder_depth_projector.pdf", bbox_inches="tight")
display(fig6)
"""


FIG7_DATA_CODE = """FIG7_ROWS = [
    {"strategy": "Frozen logistic regression", "Metallurgy": 81.7, "Chemistry-plastics": 80.2, "Company": 68.9},
    {"strategy": "Cross-entropy", "Metallurgy": 86.9, "Chemistry-plastics": 85.1, "Company": 84.4},
    {"strategy": "Batch-hard Triplet", "Metallurgy": 86.9, "Chemistry-plastics": 85.6, "Company": 76.4},
    {"strategy": "Supervised contrastive", "Metallurgy": 89.1, "Chemistry-plastics": 87.1, "Company": 79.3},
    {"strategy": "SoftTriple", "Metallurgy": 88.1, "Chemistry-plastics": 86.6, "Company": 78.7},
]

fig7_df = pd.DataFrame(FIG7_ROWS).set_index("strategy")
display(fig7_df)
"""


FIG7_PLOT_CODE = """fig7, ax = plt.subplots(figsize=(11.5, 6.8))
fig7_df.T.plot(
    kind="bar",
    ax=ax,
    color=[METHOD_COLORS[name] for name in fig7_df.index],
    width=0.82,
)

ax.set_xlabel("Target corpus")
ax.set_ylabel("Balanced accuracy (%)")
ax.set_ylim(60, 92)
ax.set_xticklabels(["Metallurgy", "Chemistry-plastics", "Company"], rotation=0)
ax.grid(axis="y", alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
fig7.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5)
ax.get_legend().remove()

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)

fig7.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98)
fig7.savefig(FIGURES_DIR / "figure_7_best_strategy_by_corpus.png", bbox_inches="tight")
fig7.savefig(FIGURES_DIR / "figure_7_best_strategy_by_corpus.pdf", bbox_inches="tight")
display(fig7)
"""


CONFUSION_MD = """## Confusion matrix — Cross-entropy, full encoder, no projector

This is the selected global model evaluated on the **Company** corpus.
The matrix is normalized by true class (row-wise), so each diagonal entry is the
recall of the corresponding accident-process role. Rows are true roles and columns
are predicted roles, in the order `A0, A1, B, C`.
"""


CONFUSION_CODE = """import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

VARIANTS_DIR = ROOT / "output/supervised_macro_ft/variants"
SUMMARY_PATH = VARIANTS_DIR / "results_summary.csv"
CLASS_ORDER = ["A0", "A1", "B", "C"]

if SUMMARY_PATH.is_file():
    variants_summary = pd.read_csv(SUMMARY_PATH)
    selected = variants_summary[
        (variants_summary["encoder_scope"].astype(str) == "Full encoder")
        & (variants_summary["projector"].astype(str).str.lower() == "no")
    ]
    if selected.empty:
        raise FileNotFoundError("Full encoder / No projector not found in results_summary.csv")
    confusion_combo_id = str(selected.iloc[0]["combo_id"])
else:
    confusion_combo_id = (
        "backbone_trainableTrue_cache_backbone_embeddingsFalse_"
        "class_weightbalanced_oversamplingFalse_project_703bf26a"
    )

PREDICTIONS_PATH = (
    VARIANTS_DIR
    / "combos"
    / confusion_combo_id
    / "predictions"
    / "predictions_nicollin.csv"
)
if not PREDICTIONS_PATH.is_file():
    raise FileNotFoundError(f"Company predictions not found: {PREDICTIONS_PATH}")

company_predictions = pd.read_csv(PREDICTIONS_PATH)
cm = confusion_matrix(
    company_predictions["true_macro"].astype(str),
    company_predictions["pred_macro"].astype(str),
    labels=CLASS_ORDER,
    normalize="true",
) * 100.0
cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
display(cm_df.round(1))

fig_cm, ax_cm = plt.subplots(figsize=(6.8, 5.8))
sns.heatmap(
    cm_df,
    ax=ax_cm,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    square=True,
    cbar_kws={"label": "Row-normalized percentage"},
    linewidths=0.6,
    linecolor="white",
)
ax_cm.set_xlabel("Predicted accident-process role")
ax_cm.set_ylabel("True accident-process role")
ax_cm.set_title("")
fig_cm.tight_layout()
fig_cm.savefig(FIGURES_DIR / "figure_company_confusion_cross_entropy_full_no.png", bbox_inches="tight")
fig_cm.savefig(FIGURES_DIR / "figure_company_confusion_cross_entropy_full_no.pdf", bbox_inches="tight")
display(fig_cm)
"""


INTERPRETATION_MD = """## Interpretation for the article

- Full-encoder adaptation is generally strongest, but the effect is not monotonic with depth.
- Projector impact depends on the loss: it strongly hurts Cross-entropy for the full encoder,
  slightly helps SupCon, and is mixed for Triplet and SoftTriple.
- Supervised contrastive learning is strongest on Metallurgy and Chemistry-plastics, while Cross-entropy is strongest
  on the Company corpus, indicating a different organizational and reporting shift.

Suggested Figure 6 caption:

> *Effect of encoder-update depth and projector inclusion on cross-corpus balanced accuracy.
> Panel (a) reports average performance across the three unseen target corpora, while panel
> (b) reports performance on the most difficult target corpus. All panels use the same
> truncated 40–90% y-axis scale.*

Suggested Figure 7 caption:

> *Balanced accuracy of the strongest configuration from each learning strategy on the three
> unseen target corpora. Metallurgy and chemistry–plastics represent cross-sector transfer
> within EPICEA, whereas the company corpus additionally introduces an organisational and
> reporting shift.*
"""


cells = [
    md(TITLE_MD),
    py(NOTEBOOK_PATH_SETUP),
    py(PARAMETERS_CODE),
    md("## Figure 6 data"),
    py(FIG6_DATA_CODE),
    py(FIG6_PLOT_CODE),
    md("## Figure 7 data"),
    py(FIG7_DATA_CODE),
    py(FIG7_PLOT_CODE),
    md(CONFUSION_MD),
    py(CONFUSION_CODE),
    md(INTERPRETATION_MD),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook écrit : {NB_PATH}")
