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

Ce notebook parcourt automatiquement les dossiers `annotation/outputs/run_*`
qui contiennent un fichier `*__annotated.xlsx` (résultats passe 1 v13).

**Tableau** : nombre de récits, unités factuelles, % A0 / A1 / B / C.

**Graphique** : distributions des classes par corpus (comparaison visuelle).
"""
    ),
    py(NOTEBOOK_PATH_SETUP),
    py(
        """
# --- Paramètres ---
from pathlib import Path

OUTPUTS_DIR = TEXT_ROOT / "annotation" / "outputs"
ONLY_RUN_IDS = None  # ex. ["run_all_btp", "run_all_metallurgie"] ou None = auto

print("OUTPUTS_DIR =", OUTPUTS_DIR)
"""
    ),
    py(
        """
# --- Découverte des runs annotées ---
from IPython.display import display

from annotation.article_stats import (
    build_label_distribution_long,
    build_summary_table,
    discover_annotation_runs,
)

all_runs = discover_annotation_runs(OUTPUTS_DIR)
if ONLY_RUN_IDS:
    allowed = set(ONLY_RUN_IDS)
    runs = [run for run in all_runs if run["run_id"] in allowed]
else:
    runs = all_runs

if not runs:
    raise FileNotFoundError(
        f"Aucune run avec *__annotated.xlsx dans {OUTPUTS_DIR}.\\n"
        "Lancez d'abord ingest sur au moins un corpus."
    )

print(f"{len(runs)} corpus trouvé(s) :")
for run in runs:
    print(f"  - {run['corpus']} ({run['run_id']})")
    print(f"    → {run['annotated_path'].name}")
"""
    ),
    py(
        """
# --- Tableau récapitulatif (article) ---
summary_table = build_summary_table(runs)

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
        "n_recits": "Nb récits",
        "n_units": "Nb unités factuelles",
        "n_annotated": "Nb annotées (pred_ok)",
        "pct_A0": "% A0",
        "pct_A1": "% A1",
        "pct_B": "% B",
        "pct_C": "% C",
    }
)

print("Tableau synthèse (pour article)")
display(article_table)

summary_table
"""
    ),
    py(
        """
# --- Distribution des classes (données longues) ---
dist_long = build_label_distribution_long(runs)
dist_long
"""
    ),
    py(
        """
# --- Graphique : % par classe, comparaison des corpus ---
import matplotlib.pyplot as plt
import numpy as np

if dist_long.empty:
    raise ValueError("Aucune distribution à tracer.")

corpora = list(dist_long["corpus"].unique())
labels = ["A0", "A1", "B", "C"]
x = np.arange(len(labels))
width = 0.8 / max(len(corpora), 1)

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = plt.cm.Set2(np.linspace(0, 1, len(corpora)))

for i, corpus in enumerate(corpora):
    subset = dist_long[dist_long["corpus"] == corpus].set_index("label").reindex(labels)
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
ax.set_ylabel("Pourcentage des unités annotées (%)")
ax.set_xlabel("Classe prédite (passe 1 v13)")
ax.set_title("Distribution des classes par corpus")
ax.legend(title="Corpus", loc="upper right")
ax.set_ylim(0, max(dist_long["pct"].max() * 1.15, 10))
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    py(
        """
# --- Graphique alternatif : effectifs absolus ---
fig, ax = plt.subplots(figsize=(10, 5.5))

for i, corpus in enumerate(corpora):
    subset = dist_long[dist_long["corpus"] == corpus].set_index("label").reindex(labels)
    offset = (i - (len(corpora) - 1) / 2) * width
    ax.bar(
        x + offset,
        subset["n_units"].values,
        width=width,
        label=corpus,
        color=colors[i],
        edgecolor="white",
        linewidth=0.6,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Nombre d'unités")
ax.set_xlabel("Classe prédite (passe 1 v13)")
ax.set_title("Effectifs par classe et par corpus")
ax.legend(title="Corpus", loc="upper right")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    py(
        """
# --- Export optionnel du tableau (CSV / XLSX) ---
export_dir = OUTPUTS_DIR / "_article_exports"
export_dir.mkdir(parents=True, exist_ok=True)

article_xlsx = export_dir / "annotation_corpus_summary.xlsx"
article_csv = export_dir / "annotation_corpus_summary.csv"

article_table.to_excel(article_xlsx, index=False)
article_table.to_csv(article_csv, index=False, encoding="utf-8-sig")

print("Exporté :")
print(" ", article_xlsx)
print(" ", article_csv)
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
