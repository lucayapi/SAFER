#!/usr/bin/env python3
"""Génère annotation/article_annotation_stats.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "annotation" / "article_annotation_stats.ipynb"

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


cells = [
    md(
        """
# Statistiques d'annotation — comparaison multi-corpus (article)

**Volumes & distributions** : CSV/XLSX `dataset/data_<corpus>.*` (labels pass-1 migrés).

**Accord inter-annotateurs** : apparie `annotation/outputs/run_all_*` (`gpt-5.4-nano`)
avec `run_partial_*` (`gpt-5-nano`, **full corpus IAA v2**) → Observed agreement,
Cohen's κ (IC 95 %), matrices de confusion, export LaTeX.

Palette unique (Okabe–Ito) pour le barplot de distribution et les heatmaps de confusion.

### Lancer la double annotation full v2 (avant la section accord)

```bash
python scripts/run_annotation_batch.py pipeline --config configs/annotation_batch_pass1_v13_partial_btp.yaml
python scripts/run_annotation_batch.py pipeline --config configs/annotation_batch_pass1_v13_partial_metallurgie.yaml
python scripts/run_annotation_batch.py pipeline --config configs/annotation_batch_pass1_v13_partial_caou_chimie_plas.yaml
python scripts/run_annotation_batch.py pipeline --config configs/annotation_batch_pass1_v13_partial_nicollin.yaml
```
"""
    ),
    py(NOTEBOOK_PATH_SETUP),
    py(
        """
# --- Paramètres ---
from pathlib import Path

DATASET_DIR = TEXT_ROOT / "dataset"
OUTPUTS_DIR = TEXT_ROOT / "annotation" / "outputs"
ONLY_CORPUS_IDS = None  # ex. ["btp", "metallurgie"] ou None = tous les data_*.csv
N_BOOT = 2000  # bootstrap IC 95 % sur κ (niveau récit)
BOOT_SEED = 42

print("DATASET_DIR =", DATASET_DIR)
print("OUTPUTS_DIR =", OUTPUTS_DIR)
"""
    ),
    py(
        """
# --- Découverte des corpus dataset/ ---
from IPython.display import display

from annotation.article_stats import (
    ARTICLE_PLOT_COLORS,
    article_confusion_cmap,
    article_plot_colors,
    build_label_distribution_long,
    build_summary_table,
    discover_dataset_corpora,
)

all_datasets = discover_dataset_corpora(DATASET_DIR)
if ONLY_CORPUS_IDS:
    allowed = set(ONLY_CORPUS_IDS)
    datasets = [d for d in all_datasets if d["corpus_id"] in allowed]
else:
    datasets = all_datasets

if not datasets:
    raise FileNotFoundError(
        f"Aucun dataset/data_*.csv/.xlsx dans {DATASET_DIR}.\\n"
        "Migrez d'abord : python scripts/migrate_annotation_to_dataset.py"
    )

print(f"{len(datasets)} corpus trouvé(s) :")
for d in datasets:
    print(f"  - {d['corpus']} ({d['corpus_id']})")
    print(f"    → {d['annotated_path'].name}")

# Alias pour compatibilité avec les cellules suivantes / exports.
runs = datasets
"""
    ),
    py(
        """
# --- Tableau récapitulatif volumes (article) ---
summary_table = build_summary_table(datasets, include_overall=True)

display_cols = [
    "corpus",
    "n_recits",
    "n_units",
    "n_annotated",
    "pct_A0",
    "pct_A1",
    "pct_B",
    "pct_C",
]
article_table = summary_table[display_cols].copy()
article_table = article_table.rename(
    columns={
        "corpus": "Corpus",
        "n_recits": "Narratives (≥1 ok)",
        "n_units": "Nb unités factuelles",
        "n_annotated": "Nb annotées (pred_ok)",
        "pct_A0": "% A0",
        "pct_A1": "% A1",
        "pct_B": "% B",
        "pct_C": "% C",
    }
)

print("Tableau synthèse (pour article) — source : dataset/")
display(article_table)

summary_table
"""
    ),
    py(
        """
# --- Distribution des classes (données longues, sans Overall) ---
dist_long = build_label_distribution_long(datasets)
dist_long
"""
    ),
    py(
        """
# --- Graphique : % par classe, comparaison des corpus (figure article) ---
import matplotlib.pyplot as plt
import numpy as np

if dist_long.empty:
    raise ValueError("Aucune distribution à tracer.")

dist_plot = dist_long.copy()
corpora = list(dict.fromkeys(dist_plot["corpus"]))
labels = ["A0", "A1", "B", "C"]
x = np.arange(len(labels))
width = 0.8 / max(len(corpora), 1)
colors = article_plot_colors(len(corpora))

fig, ax = plt.subplots(figsize=(10, 5.5))

for i, corpus in enumerate(corpora):
    subset = (
        dist_plot[dist_plot["corpus"] == corpus]
        .set_index("label")
        .reindex(labels)
    )
    offset = (i - (len(corpora) - 1) / 2) * width
    bars = ax.bar(
        x + offset,
        subset["pct"].values,
        width=width,
        label=corpus,
        color=colors[i],
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, pct in zip(bars, subset["pct"].values):
        if pct >= 3:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Factual units (%)")
ax.set_xlabel("Accident-process role")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_ylim(0, max(dist_plot["pct"].max() * 1.15, 10))
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(bottom=0.22)
fig.text(
    0.5,
    0.02,
    "Distribution of accident-process role annotations across corpora "
    "(dataset/ CSV, pass-1 labels).",
    ha="center",
    va="bottom",
    fontsize=9,
)
plt.show()
"""
    ),
    md(
        """
## Inter-annotator agreement

Jointure `(accident_id, fact_id)` entre `run_all_*` et `run_partial_*` (même corpus).
Métriques : observed agreement, Cohen's κ multiclasse (IC 95 % bootstrap au niveau récit),
effectifs par rôle (`n_A0`…`n_C` selon labels `run_all` sur l'échantillon apparié),
et matrices de confusion (`run_all` × `run_partial`).
"""
    ),
    py(
        """
# --- Accord run_all (gpt-5.4-nano) vs run_partial full v2 (gpt-5-nano) ---
import pandas as pd
from annotation.agreement_stats import (
    build_agreement_artifacts,
    disagreement_subset,
    discover_agreement_pairs,
    export_agreement_latex,
    export_confusion_latex,
)

pairs = discover_agreement_pairs(OUTPUTS_DIR)
disagreements_by_corpus = {}
all_disagreements = pd.DataFrame()
if not pairs:
    print(
        "Aucune paire run_all_* / run_partial_* trouvée.\\n"
        "Lancez d'abord les pipelines full v2 (voir intro)."
    )
    agreement_table = None
    confusion_matrices = {}
    paired_by_corpus = {}
    latex_confusions = {}
    latex_agreement = None
else:
    print(f"{len(pairs)} paire(s) :")
    for p in pairs:
        print(f"  - {p['corpus']}: {p['run_all_id']} ↔ {p['run_partial_id']}")

    agreement_table, confusion_matrices, paired_by_corpus = build_agreement_artifacts(
        pairs, n_boot=N_BOOT, seed=BOOT_SEED, include_overall=True
    )
    latex_confusions = {}
    display_agree = agreement_table[
        [
            "corpus",
            "n_narratives",
            "n_factual_units",
            "n_A0",
            "n_A1",
            "n_B",
            "n_C",
            "observed_agreement_pct",
            "kappa_display",
        ]
    ].rename(
        columns={
            "corpus": "Corpus",
            "n_narratives": "Narratives",
            "n_factual_units": "Factual units",
            "n_A0": "n_A0",
            "n_A1": "n_A1",
            "n_B": "n_B",
            "n_C": "n_C",
            "observed_agreement_pct": "Observed agreement (%)",
            "kappa_display": "Cohen's κ (95% CI)",
        }
    )
    display(display_agree)
    latex_agreement = export_agreement_latex(agreement_table)
    print(latex_agreement)

    # Tous les désaccords — corpus appariés.
    # ``paired_df`` a déjà une colonne ``corpus`` (build_agreement_artifacts).
    for corpus_name, paired_df in paired_by_corpus.items():
        dd = disagreement_subset(paired_df)  # tous les rôles
        if not dd.empty:
            dd = dd.copy()
            if "corpus" not in dd.columns:
                dd.insert(0, "corpus", corpus_name)
            else:
                dd["corpus"] = corpus_name
        disagreements_by_corpus[corpus_name] = dd
    if disagreements_by_corpus:
        all_disagreements = pd.concat(
            list(disagreements_by_corpus.values()), ignore_index=True
        )
    else:
        all_disagreements = pd.DataFrame()
"""
    ),
    py(
        """
# --- Matrices de confusion (run_all × run_partial) ---
import matplotlib.pyplot as plt
import numpy as np

# Même palette Okabe–Ito que le barplot de distribution.
_iaa_cmap = article_confusion_cmap()

if not confusion_matrices:
    print("Pas de matrice de confusion (aucune paire).")
else:
    latex_confusions = {}
    for corpus_name, cm in confusion_matrices.items():
        print(f"\\n=== {corpus_name} ===")
        display(cm)
        latex_confusions[corpus_name] = export_confusion_latex(cm, corpus=corpus_name)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        data = cm.to_numpy(dtype=float)
        im = ax.imshow(data, cmap=_iaa_cmap, vmin=0)
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns)
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("Annotator 2")
        ax.set_ylabel("Annotator 1")
        thresh = data.max() * 0.55 if data.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm.iloc[i, j])
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if data[i, j] > thresh else "#333333",
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Number of factual units", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.5,
            0.02,
            f"Confusion matrix — {corpus_name}: Annotator 1 (rows) vs "
            "Annotator 2 (columns), before adjudication.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        plt.show()
"""
    ),
    md(
        """
### Annotator disagreements (all corpora)

Unités où Annotator 1 (`run_all`) et Annotator 2 (`run_partial`) divergent
sur le rôle (`A0`/`A1`/`B`/`C`), pour **chaque** corpus apparié.
"""
    ),
    py(
        """
# --- Tous les désaccords : tous les corpus ---
show_cols = [
    "corpus",
    "accident_id",
    "fact_id",
    "sentence",
    "accident_summary",
    "label_all",
    "label_partial",
    "disagreement",
    "justification_all",
    "justification_partial",
]

if not disagreements_by_corpus:
    print("Aucun désaccord (aucune paire ou accord parfait).")
    summary_disagreements_all = pd.DataFrame()
else:
    summary_rows = []
    for corpus_name, dd_df in disagreements_by_corpus.items():
        print(f"\\n=== {corpus_name} — désaccords ({len(dd_df)} units) ===")
        if dd_df.empty:
            print("  (aucun désaccord)")
            continue
        summary_c = (
            dd_df.groupby("disagreement", dropna=False)
            .size()
            .reset_index(name="n_units")
            .sort_values("n_units", ascending=False)
        )
        summary_c.insert(0, "corpus", corpus_name)
        summary_rows.append(summary_c)
        display(summary_c)
        cols = [c for c in show_cols if c in dd_df.columns]
        display(dd_df[cols])
    summary_disagreements_all = (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    )
    if not all_disagreements.empty:
        print(f"\\nTotal désaccords (tous corpus) : {len(all_disagreements)}")
        if not summary_disagreements_all.empty:
            print("Résumé agrégé :")
            display(summary_disagreements_all)
"""
    ),
    py(
        """
# --- Export optionnel (CSV / XLSX / LaTeX) ---
export_dir = OUTPUTS_DIR / "_article_exports"
export_dir.mkdir(parents=True, exist_ok=True)

article_xlsx = export_dir / "annotation_corpus_summary.xlsx"
article_csv = export_dir / "annotation_corpus_summary.csv"
article_table.to_excel(article_xlsx, index=False)
article_table.to_csv(article_csv, index=False, encoding="utf-8-sig")

print("Exporté :")
print(" ", article_xlsx)
print(" ", article_csv)

if agreement_table is not None:
    agree_csv = export_dir / "annotation_agreement.csv"
    agree_xlsx = export_dir / "annotation_agreement.xlsx"
    agree_tex = export_dir / "annotation_agreement.tex"
    agreement_table.to_csv(agree_csv, index=False, encoding="utf-8-sig")
    agreement_table.to_excel(agree_xlsx, index=False)
    agree_tex.write_text(latex_agreement or "", encoding="utf-8")
    print(" ", agree_csv)
    print(" ", agree_xlsx)
    print(" ", agree_tex)

if confusion_matrices:
    for corpus_name, cm in confusion_matrices.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in corpus_name.lower())
        cm_csv = export_dir / f"annotation_confusion_{slug}.csv"
        cm_xlsx = export_dir / f"annotation_confusion_{slug}.xlsx"
        cm_tex = export_dir / f"annotation_confusion_{slug}.tex"
        cm.to_csv(cm_csv, encoding="utf-8-sig")
        cm.to_excel(cm_xlsx)
        cm_tex.write_text(
            latex_confusions.get(corpus_name, export_confusion_latex(cm, corpus=corpus_name)),
            encoding="utf-8",
        )
        print(" ", cm_csv)
        print(" ", cm_xlsx)
        print(" ", cm_tex)

if disagreements_by_corpus:
    all_units = []
    all_summaries = []
    for corpus_name, dd_df in disagreements_by_corpus.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in corpus_name.lower())
        dd_xlsx = export_dir / f"annotation_disagreements_{slug}.xlsx"
        dd_csv = export_dir / f"annotation_disagreements_{slug}.csv"
        summary_c = (
            dd_df.groupby("disagreement", dropna=False)
            .size()
            .reset_index(name="n_units")
            .sort_values("n_units", ascending=False)
            if not dd_df.empty
            else pd.DataFrame(columns=["disagreement", "n_units"])
        )
        with pd.ExcelWriter(dd_xlsx, engine="openpyxl") as writer:
            summary_c.to_excel(writer, sheet_name="summary", index=False)
            dd_df.to_excel(writer, sheet_name="units", index=False)
        dd_df.to_csv(dd_csv, index=False, encoding="utf-8-sig")
        print(" ", dd_xlsx)
        print(" ", dd_csv)
        if not dd_df.empty:
            all_units.append(dd_df)
            s = summary_c.copy()
            s.insert(0, "corpus", corpus_name)
            all_summaries.append(s)

    if all_units:
        dd_all_xlsx = export_dir / "annotation_disagreements_all.xlsx"
        dd_all_csv = export_dir / "annotation_disagreements_all.csv"
        units_all = pd.concat(all_units, ignore_index=True)
        summary_all = pd.concat(all_summaries, ignore_index=True)
        with pd.ExcelWriter(dd_all_xlsx, engine="openpyxl") as writer:
            summary_all.to_excel(writer, sheet_name="summary", index=False)
            units_all.to_excel(writer, sheet_name="units", index=False)
        units_all.to_csv(dd_all_csv, index=False, encoding="utf-8-sig")
        print(" ", dd_all_xlsx)
        print(" ", dd_all_csv)
"""
    ),
]
NB_PATH.write_text(
    json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "cells": cells,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print(f"Écrit : {NB_PATH}")
