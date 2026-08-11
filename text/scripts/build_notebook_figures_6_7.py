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

The figures load the latest available metrics directly from each method's variant
directories. They do not depend exclusively on global summary CSV files. Figures are
exported as both PNG and PDF under `output/article_figures_6_7/figures/`.

The y-axes are intentionally truncated to make differences visible. The limits are
reported in each figure caption and should be retained in the article caption.
"""


PARAMETERS_CODE = """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
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
METHOD_DISPLAY_NAMES = {
    "Frozen logistic regression": "Frozen embeddings + LR",
    "Cross-entropy": "Cross-entropy fine-tuning",
    "Batch-hard Triplet": "Batch-hard Triplet",
    "Supervised contrastive": "Supervised contrastive",
    "SoftTriple": "SoftTriple",
}
METHOD_COLORS = {
    "Frozen embeddings + LR": "#7F7F7F",
    "Cross-entropy fine-tuning": "#D55E00",
    "Batch-hard Triplet": "#0072B2",
    "Supervised contrastive": "#009E73",
    "SoftTriple": "#CC79A7",
}
METHOD_RESULTS_DIRS = {
    "Cross-entropy": ROOT / "output" / "supervised_macro_ft" / "variants",
    "Batch-hard Triplet": ROOT / "output" / "batch_triplet" / "macro_ft_tuning",
    "Supervised contrastive": ROOT / "output" / "supcon" / "macro_ft_tuning",
    "SoftTriple": ROOT / "output" / "softtriple" / "macro_ft_tuning",
}
TARGET_CORPORA = ["metallurgie", "caou", "nicollin"]
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


def _read_metric_csv(path):
    if not path.is_file():
        return float("nan")
    frame = pd.read_csv(path)
    if frame.empty or "balanced_accuracy" not in frame.columns:
        return float("nan")
    return float(frame.iloc[0]["balanced_accuracy"])


def _variant_labels(config, combo_id):
    variant = str(config.get("architecture_variant") or combo_id)
    if variant in {"last_1_yes", "last_1_no", "last_2_yes", "last_2_no", "last_3_yes", "last_3_no", "full_yes", "full_no"}:
        scope, projector = variant.rsplit("_", 1)
        depth = "Full encoder" if scope == "full" else f"Last {scope.split('_')[1]} " + ("layer" if scope == "last_1" else "layers")
        return depth, "With projector" if projector == "yes" else "Without projector"
    model = config.get("model") or {}
    layers = model.get("train_last_n_layers")
    depth = "Full encoder" if layers is None else f"Last {int(layers)} " + ("layer" if int(layers) == 1 else "layers")
    return depth, "With projector" if model.get("use_projector", True) else "Without projector"


def _discover_contrastive_method_rows(method, results_dir):
    rows = []
    combos_dir = results_dir / "combos"
    if not combos_dir.is_dir():
        return pd.DataFrame()
    for combo_dir in sorted(path for path in combos_dir.iterdir() if path.is_dir()):
        cv_path = combo_dir / "cv" / "cv_summary.csv"
        config_path = combo_dir / "configs" / "config_resolved.yaml"
        if not cv_path.is_file():
            continue
        cv = pd.read_csv(cv_path)
        if cv.empty:
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        depth, projector = _variant_labels(config, combo_dir.name)
        cv_row = cv.iloc[0]
        row = {
            "method": method,
            "depth": depth,
            "projector": projector,
            "ood_avg": float(cv_row.get("mean_ood_balanced_accuracy", float("nan"))) * 100.0,
            "ood_worst": float(cv_row.get("min_ood_balanced_accuracy", float("nan"))) * 100.0,
        }
        values = []
        for corpus_id in TARGET_CORPORA:
            metric_path = results_dir / "combos" / combo_dir.name / "metrics" / f"metrics_classification_test_{corpus_id}.csv"
            value = _read_metric_csv(metric_path) * 100.0
            row[f"ba_ood_{corpus_id}"] = value
            if np.isfinite(value):
                values.append(value)
        if values:
            row["ood_avg"] = float(np.mean(values))
            row["ood_worst"] = float(np.min(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_article_method_rows(method, results_dir):
    if method == "Cross-entropy":
        path = results_dir / "results_summary.csv"
        if not path.is_file():
            return pd.DataFrame()
        source = pd.read_csv(path)
        rows = []
        for _, item in source.iterrows():
            rows.append({
                "method": method,
                "depth": item.get("encoder_scope"),
                "projector": "With projector" if str(item.get("projector")).lower() == "yes" else "Without projector",
                "ood_avg": float(item.get("ba_ood_avg", float("nan"))) * 100.0,
                "ood_worst": float(item.get("ba_ood_worst", float("nan"))) * 100.0,
                **{f"ba_ood_{corpus_id}": float(item.get(f"ba_ood_{corpus_id}", float("nan"))) * 100.0 for corpus_id in TARGET_CORPORA},
            })
        return pd.DataFrame(rows)
    return _discover_contrastive_method_rows(method, results_dir)


dynamic_rows = []
for method_name in METHOD_ORDER:
    method_rows = _load_article_method_rows(method_name, METHOD_RESULTS_DIRS[method_name])
    if method_rows.empty:
        print(f"No complete result rows found for {method_name}")
    dynamic_rows.append(method_rows)
dynamic_fig6_df = pd.concat(dynamic_rows, ignore_index=True) if dynamic_rows else pd.DataFrame()
if not dynamic_fig6_df.empty:
    fig6_df = dynamic_fig6_df
display(fig6_df)
"""


FIG6_PLOT_CODE = """from matplotlib.lines import Line2D

fig6, axes = plt.subplots(
    2,
    4,
    figsize=(14, 6.8),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.08, "wspace": 0.14},
)

metric_specs = [
    ("ood_avg", 76.9, "(a) OOD average BA"),
    ("ood_worst", 68.9, "(b) Worst-corpus BA"),
]
x = range(len(DEPTH_ORDER))
projector_styles = {
    "Without projector": {"linestyle": "-", "marker": "o"},
    "With projector": {"linestyle": "--", "marker": "s"},
}
for column, method in enumerate(METHOD_ORDER):
    method_df = fig6_df[fig6_df["method"] == method]
    for row, (metric, baseline, row_label) in enumerate(metric_specs):
        ax = axes[row, column]
        for projector in PROJECTOR_ORDER:
            series = method_df[method_df["projector"] == projector].set_index("depth").reindex(DEPTH_ORDER)
            ax.plot(
                list(x),
                series[metric].to_numpy(),
                marker=projector_styles[projector]["marker"],
                linestyle=projector_styles[projector]["linestyle"],
                linewidth=2,
                markersize=5,
                color=PROJECTOR_COLORS[projector],
            )
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2)
        ax.set_ylim(65, 90)
        ax.set_xticks(list(x), ["Last 1", "Last 2", "Last 3", "Full"])
        if row == 0:
            ax.set_title(METHOD_DISPLAY_NAMES[method])
        ax.grid(axis="y", alpha=0.3)

for ax in axes.flat:
    ax.tick_params(axis="y", pad=2)
axes[0, 0].set_ylabel("(a) OOD average BA", labelpad=6, fontweight="bold")
axes[1, 0].set_ylabel("(b) Worst-corpus BA", labelpad=6, fontweight="bold")
fig6.text(0.5, 0.015, "Encoder-update depth", ha="center", fontweight="bold")
legend_handles = [
    Line2D([0], [0], color=PROJECTOR_COLORS["Without projector"], marker="o", linestyle="-", linewidth=2, label="No projector"),
    Line2D([0], [0], color=PROJECTOR_COLORS["With projector"], marker="s", linestyle="--", linewidth=2, label="With projector"),
    Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2, label="Frozen embeddings + LR baseline"),
]
fig6.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3)
fig6.subplots_adjust(left=0.085, right=0.99, bottom=0.08, top=0.91, hspace=0.08, wspace=0.14)
fig6.savefig(FIGURES_DIR / "figure_6_encoder_depth_projector.png", bbox_inches="tight")
fig6.savefig(FIGURES_DIR / "figure_6_encoder_depth_projector.pdf", bbox_inches="tight")
display(fig6)
plt.close(fig6)
"""


FIG7_DATA_CODE = """FIG7_ROWS = [
    {"strategy": "Frozen embeddings + LR", "Metallurgy": 81.7, "Chemistry-plastics": 80.2, "Company": 68.9},
    {"strategy": "Cross-entropy fine-tuning", "Metallurgy": 86.9, "Chemistry-plastics": 85.1, "Company": 84.4},
    {"strategy": "Batch-hard Triplet", "Metallurgy": 86.9, "Chemistry-plastics": 85.6, "Company": 76.4},
    {"strategy": "Supervised contrastive", "Metallurgy": 89.1, "Chemistry-plastics": 87.1, "Company": 79.3},
    {"strategy": "SoftTriple", "Metallurgy": 88.1, "Chemistry-plastics": 86.6, "Company": 78.7},
]

fig7_df = pd.DataFrame(FIG7_ROWS).set_index("strategy")

FIG7_ROWS_DYNAMIC = [
    {"strategy": "Frozen embeddings + LR", "Metallurgy": 81.7, "Chemistry-plastics": 80.2, "Company": 68.9},
]
corpus_display = {
    "metallurgie": "Metallurgy",
    "caou": "Chemistry-plastics",
    "nicollin": "Company",
}
for method_name in METHOD_ORDER:
    method_rows = _load_article_method_rows(method_name, METHOD_RESULTS_DIRS[method_name])
    if method_rows.empty:
        continue
    method_rows = method_rows.dropna(subset=["ood_avg"])
    if method_rows.empty:
        continue
    best = method_rows.sort_values("ood_avg", ascending=False).iloc[0]
    FIG7_ROWS_DYNAMIC.append({
        "strategy": METHOD_DISPLAY_NAMES[method_name],
        **{
            display_name: float(best.get(f"ba_ood_{corpus_id}", float("nan")))
            for corpus_id, display_name in corpus_display.items()
        },
    })
fig7_df = pd.DataFrame(FIG7_ROWS_DYNAMIC).set_index("strategy")
display(fig7_df)
"""


FIG7_PLOT_CODE = """fig7, ax = plt.subplots(figsize=(11.5, 6.3))
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
fig7.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5)
ax.get_legend().remove()

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)

fig7.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.98)
fig7.savefig(FIGURES_DIR / "figure_7_best_strategy_by_corpus.png", bbox_inches="tight")
fig7.savefig(FIGURES_DIR / "figure_7_best_strategy_by_corpus.pdf", bbox_inches="tight")
display(fig7)
plt.close(fig7)
"""


CONFUSION_MD = """## Confusion matrix — Cross-entropy fine-tuning, full encoder, no projector

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
- Projector impact depends on the loss: it strongly hurts Cross-entropy fine-tuning for the full encoder,
  slightly helps SupCon, and is mixed for Triplet and SoftTriple.
- Supervised contrastive learning is strongest on Metallurgy and Chemistry-plastics, while Cross-entropy fine-tuning is strongest
  on the Company corpus, indicating a different organizational and reporting shift.

Suggested Figure 6 caption:

> *Effect of encoder-update depth and projector inclusion on cross-corpus balanced accuracy.
> Panel (a) reports average performance across the three unseen target corpora, while panel
> (b) reports performance on the most difficult target corpus. All panels use the same
> truncated 65–90% y-axis scale. The frozen logistic-regression baseline reaches 76.9%
> for average OOD BA and 68.9% for worst-corpus BA.*

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
