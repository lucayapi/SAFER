"""Génère notebooks/08_fsp_macro_transfer_results.ipynb (diagnostics FSP)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "08_fsp_macro_transfer_results.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP
from macro_transfer.notebook_viz import (
    RAW_TEST_EMBEDDING_SECTION_MD,
    notebook_raw_test_embedding_source,
    notebook_topic_judge_section_md,
    notebook_topic_judge_source,
)


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


def main() -> None:
    cells = [
        md(
            r"""
# 08 — Diagnostics Frozen Source Prototypes (toutes méthodes)

Synthèse comparative de **toutes** les méthodes FSP (`scgm_text`, `softtriple`, `supcon`, `batch_triplet`, `raw_embedding`).

Artefacts sous `output_test/<TEST_CORPUS>/macro_transfer/frozen_source_prototypes/<méthode>/`.

**Prérequis** : lancer un run par encodeur —
`BASE_METHOD=<encodeur> CORPUS=<id> bash jobs/run_frozen_source_prototypes.sh`

La section détail utilise `FSP_BASE_METHOD` (variable d'environnement ou valeur par défaut).
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from macro_transfer.fsp_config import FSP_ENCODER_METHODS, resolve_fsp_output_dir
from macro_transfer.notebook_viz import (
    export_fsp_metrics_latex_table,
    load_fsp_metrics_comparison_table,
    load_fsp_run_artifacts,
    plot_fsp_methods_metrics_comparison,
    plot_fsp_distribution_histograms,
    plot_fsp_pred_macro_distribution,
    plot_fsp_distance_boxplot,
    plot_fsp_confusion_heatmap,
    compute_fsp_confidence_calibration,
    get_fsp_top_confident_errors,
)

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
FSP_TRAINED_METHODS = [m for m in FSP_ENCODER_METHODS if m != "raw_embedding"]
FSP_BASE_METHOD = os.environ.get("FSP_BASE_METHOD", "scgm_text")

def fsp_out_dir(method: str) -> Path:
    return resolve_fsp_output_dir(TEST_CORPUS, method, anchor=TEXT_ROOT)

OUT_DIR = fsp_out_dir(FSP_BASE_METHOD)
ROOT_FSP = OUT_DIR.parent
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"08_fsp_{TEST_CORPUS}"
FIG_DETAIL_DIR = FIG_DIR / FSP_BASE_METHOD
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DETAIL_DIR.mkdir(parents=True, exist_ok=True)

print("Corpus test :", TEST_CORPUS)
print("Méthodes FSP :", ", ".join(FSP_ENCODER_METHODS))
print("Détail (FSP_BASE_METHOD) :", FSP_BASE_METHOD)
print("OUT_DIR :", OUT_DIR)
sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Synthèse — toutes les méthodes"),
        py(
            r"""
all_methods_df = load_fsp_metrics_comparison_table(TEST_CORPUS, anchor=TEXT_ROOT)
display_cols = ["Méthode", "Bal. Acc.", "F1 (étapes)", "Confiance moy.", "Entropie moy.", "metrics_available"]
display(all_methods_df[display_cols])

missing = all_methods_df.loc[~all_methods_df["metrics_available"], "method_key"].tolist()
if missing:
    print("Runs manquants (metrics.json absent) :", ", ".join(missing))
    print("Relancer : BASE_METHOD=<encodeur> CORPUS=" + TEST_CORPUS + " bash jobs/run_frozen_source_prototypes.sh")

export_df = all_methods_df[["Méthode", "Bal. Acc.", "F1 (étapes)", "Confiance moy.", "Entropie moy."]].copy()
TABLE_CSV = ROOT_FSP / "table_transfer_direct_all_methods.csv"
TABLE_TEX = ROOT_FSP / "table_transfer_direct_all_methods.tex"
ROOT_FSP.mkdir(parents=True, exist_ok=True)
export_df.to_csv(TABLE_CSV, index=False)
latex_table = export_fsp_metrics_latex_table(export_df)
print(latex_table)
TABLE_TEX.write_text(latex_table + "\n", encoding="utf-8")
print("CSV :", TABLE_CSV)
print("TEX :", TABLE_TEX)

plot_fsp_methods_metrics_comparison(all_methods_df, fig_dir=FIG_DIR)
"""
        ),
        md(
            r"""
## § Détail méthode sélectionnée (`FSP_BASE_METHOD`)

Changer `FSP_BASE_METHOD` (ou `os.environ["FSP_BASE_METHOD"]`) puis réexécuter à partir d'ici
pour explorer une autre méthode : `scgm_text`, `softtriple`, `supcon`, `batch_triplet`, `raw_embedding`.
"""
        ),
        md("### Chargement et résumé"),
        py(
            r"""
ART = load_fsp_run_artifacts(OUT_DIR)
pred = ART.predictions.copy()
protos = ART.prototypes.copy()
metrics = ART.metrics or {}
display(protos.head())
if metrics:
    display(pd.DataFrame([metrics]))
else:
    print("metrics.json absent.")
"""
        ),
        md("### Distribution étapes et scores"),
        py(
            r"""
plot_fsp_pred_macro_distribution(pred, fig_dir=FIG_DETAIL_DIR)
plot_fsp_distribution_histograms(pred, fig_dir=FIG_DETAIL_DIR)
plot_fsp_distance_boxplot(pred, fig_dir=FIG_DETAIL_DIR)
"""
        ),
        md("### Probabilités et distances"),
        py(
            r"""
prob_cols = [c for c in pred.columns if c.startswith("prob_")]
dist_cols = [c for c in pred.columns if c.startswith("dist_")]
print("Colonnes prob_* :", prob_cols)
print("Colonnes dist_* :", dist_cols)
display(pred[[c for c in ["pred_macro", "confidence", "margin", "entropy"] + prob_cols[:4] + dist_cols[:4] if c in pred.columns]].head())
"""
        ),
        md("### Confusion, report, calibration"),
        py(
            r"""
if ART.confusion is not None and not ART.confusion.empty:
    display(ART.confusion)
    plot_fsp_confusion_heatmap(ART.confusion, fig_dir=FIG_DETAIL_DIR)
else:
    print("confusion_matrix.csv absent.")

if ART.classification_report is not None and not ART.classification_report.empty:
    display(ART.classification_report)
else:
    print("classification_report.csv absent.")

cal = compute_fsp_confidence_calibration(pred)
if cal is not None and not cal.empty:
    display(cal)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cal["mean_confidence"], cal["accuracy"], "o-")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Calibration confiance")
    ax.set_xlabel("confidence moyenne")
    ax.set_ylabel("accuracy")
    fig.savefig(FIG_DETAIL_DIR / "calibration.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("Calibration non disponible.")
"""
        ),
        md("### Erreurs à forte confiance"),
        py(
            r"""
top_err = get_fsp_top_confident_errors(pred, top_k=30)
if top_err.empty:
    print("Aucune erreur exploitable (ou true_macro absent).")
else:
    cols = [c for c in ["sentence", "true_macro", "pred_macro", "confidence", "margin", "entropy"] if c in top_err.columns]
    display(top_err[cols])
"""
        ),
        md("### Inputs BERTopic"),
        py(
            r"""
all_path = OUT_DIR / "transfer" / "bertopic_input_all.csv"
if all_path.is_file():
    all_df = pd.read_csv(all_path)
    print("bertopic_input_all.csv:", len(all_df))
    display(all_df.head())
else:
    print("bertopic_input_all.csv absent.")

for macro in ("A0", "A1", "B", "C"):
    p = OUT_DIR / "transfer" / f"bertopic_input_{macro}.csv"
    if p.is_file():
        print(f"{macro}: {len(pd.read_csv(p))} lignes")
    else:
        print(f"{macro}: fichier absent")
"""
        ),
        md("### BERTopic thèmes / labels"),
        py(
            r"""
themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"
assign_path = OUT_DIR / "topics_bertopic" / "assignments.csv"
summary_path = OUT_DIR / "summary" / "topics_summary.csv"

if summary_path.is_file():
    display(pd.read_csv(summary_path))
else:
    print("topics_summary.csv absent.")

if themes_path.is_file():
    themes = pd.read_csv(themes_path)
    cols = [c for c in ["macro", "topic_id", "n_units", "theme_label", "top_words"] if c in themes.columns]
    display(themes[cols].head(20))
    if "theme_label" not in themes.columns:
        print("theme_label absent (fallback top_words).")
else:
    print("themes_by_macro.csv absent.")

if assign_path.is_file():
    display(pd.read_csv(assign_path).head())
else:
    print("assignments.csv absent.")
"""
        ),
        md(notebook_topic_judge_section_md()),
        py(
            notebook_topic_judge_source(
                "OUT_DIR",
                "FIG_DETAIL_DIR",
                restimate_var=None,
                topic_judge_cfg_var=None,
            )
        ),
        md("### Tableau topics par étape (format article / LaTeX)"),
        py(
            r"""
stats_path = OUT_DIR / "summary" / "macro_topic_stats.csv"
if not stats_path.is_file():
    print("macro_topic_stats.csv absent.")
else:
    df_stats = pd.read_csv(stats_path)
    required = {"macro", "n_units", "n_topics", "bruit_pct", "plus_gros_topic_pct"}
    missing = sorted(required.difference(df_stats.columns))
    if missing:
        print(f"Colonnes manquantes pour tableau topics par étape: {missing}")
    else:
        table_macro = pd.DataFrame(
            {
                "Étape": df_stats["macro"].astype(str),
                "Unités": pd.to_numeric(df_stats["n_units"], errors="coerce").fillna(0).astype(int),
                "Topics": pd.to_numeric(df_stats["n_topics"], errors="coerce").fillna(0).astype(int),
                "Bruit": pd.to_numeric(df_stats["bruit_pct"], errors="coerce").map(
                    lambda x: f"{x:.1f}\\%" if pd.notna(x) else "--"
                ),
                "Plus gros topic": pd.to_numeric(df_stats["plus_gros_topic_pct"], errors="coerce").map(
                    lambda x: f"{x:.1f}\\%" if pd.notna(x) else "--"
                ),
            }
        )
        display(table_macro)
        latex_macro = table_macro.to_latex(
            index=False,
            escape=False,
            column_format="lcccc",
        )
        print(latex_macro)
        tex_out = OUT_DIR / "summary" / "macro_topic_stats_table.tex"
        tex_out.write_text(latex_macro, encoding="utf-8")
        print("LaTeX écrit :", tex_out)
"""
        ),
        md(RAW_TEST_EMBEDDING_SECTION_MD),
        py(notebook_raw_test_embedding_source("FIG_DIR")),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Wrote", NB_PATH)


if __name__ == "__main__":
    main()
