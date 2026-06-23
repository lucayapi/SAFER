"""Génère notebooks/09_softtriple_native_fsp_diagnostics.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "09_softtriple_native_fsp_diagnostics.ipynb"

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
# 09 — Diagnostics FSP SoftTriple (centres natifs)

Transfert macro via les **centres appris** `W_{r,k}` du checkpoint SoftTriple
(agrégation γ, softmax inter-macro / T), job dédié — **pas** la moyenne BTP.

**Prérequis** :
```bash
CORPUS=<id> bash jobs/run_softtriple_native_fsp.sh
python scripts/build_notebook_09_softtriple_native_fsp.py
```

Artefacts natifs : `output_test/<TEST_CORPUS>/macro_transfer/frozen_source_prototypes/softtriple_native/`

Comparaison optionnelle avec l'ancien run `softtriple/` (prototype moyen, job FSP générique).
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

from macro_transfer.fsp_config import FSP_SOFTTRIPLE_NATIVE_METHOD, resolve_fsp_output_dir
from macro_transfer.notebook_viz import (
    compute_fsp_confidence_calibration,
    export_fsp_metrics_latex_table,
    get_fsp_top_confident_errors,
    load_fsp_run_artifacts,
    load_softtriple_native_vs_legacy_metrics,
    plot_fsp_confusion_heatmap,
    plot_fsp_distribution_histograms,
    plot_fsp_distance_boxplot,
    plot_fsp_pred_macro_distribution,
    plot_softtriple_center_similarity_heatmap,
    plot_softtriple_center_weight_bars,
    plot_softtriple_relaxed_score_boxplot,
)

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
OUT_NATIVE = resolve_fsp_output_dir(TEST_CORPUS, FSP_SOFTTRIPLE_NATIVE_METHOD, anchor=TEXT_ROOT)
OUT_LEGACY = resolve_fsp_output_dir(TEST_CORPUS, "softtriple", anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"09_softtriple_native_{TEST_CORPUS}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Corpus test :", TEST_CORPUS)
print("OUT_NATIVE :", OUT_NATIVE)
print("OUT_LEGACY :", OUT_LEGACY)
sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Comparaison transfert — natif vs prototype moyen"),
        py(
            r"""
cmp_df = load_softtriple_native_vs_legacy_metrics(TEST_CORPUS, anchor=TEXT_ROOT)
display_cols = [
    "Méthode", "Bal. Acc.", "F1 (étapes)", "Confiance moy.", "Entropie moy.",
    "assignment_mode", "gamma", "temperature", "distance_metric", "centers_per_class",
    "metrics_available",
]
display(cmp_df[[c for c in display_cols if c in cmp_df.columns]])

if not cmp_df.loc[cmp_df["method_key"] == FSP_SOFTTRIPLE_NATIVE_METHOD, "metrics_available"].iloc[0]:
    print("Run natif manquant — lancer : CORPUS=" + TEST_CORPUS + " bash jobs/run_softtriple_native_fsp.sh")

export_cmp = cmp_df.loc[cmp_df["method_key"].isin([FSP_SOFTTRIPLE_NATIVE_METHOD, "softtriple", "delta_native_minus_legacy"])]
TABLE_CSV = OUT_NATIVE.parent / "table_softtriple_native_vs_legacy.csv"
TABLE_TEX = OUT_NATIVE.parent / "table_softtriple_native_vs_legacy.tex"
export_display = export_cmp[["Méthode", "Bal. Acc.", "F1 (étapes)", "Confiance moy.", "Entropie moy."]].copy()
export_display.to_csv(TABLE_CSV, index=False)
latex_cmp = export_fsp_metrics_latex_table(export_display)
print(latex_cmp)
TABLE_TEX.write_text(latex_cmp + "\n", encoding="utf-8")
print("CSV :", TABLE_CSV)
print("TEX :", TABLE_TEX)
"""
        ),
        md("## § Hyperparamètres et chargement run natif"),
        py(
            r"""
ART = load_fsp_run_artifacts(OUT_NATIVE)
pred = ART.predictions.copy()
protos = ART.prototypes.copy()
metrics = ART.metrics or {}
display(pd.DataFrame([metrics]))
print("assignment_mode :", metrics.get("assignment_mode"))
print("gamma :", metrics.get("gamma"), "| temperature :", metrics.get("temperature"))
print("centers_per_class :", metrics.get("centers_per_class"))
print("distance_metric :", metrics.get("distance_metric"))
"""
        ),
        md("### Centres appris W_{r,k}"),
        py(
            r"""
display(protos.head(20))
print("Lignes source_prototypes.csv :", len(protos), "(attendu K × 4 macros)")
plot_softtriple_center_similarity_heatmap(protos, fig_dir=FIG_DIR)
"""
        ),
        md("### Poids centres α (agrégation intra-macro)"),
        py(
            r"""
weights_path = ART.transfer_dir / "softtriple_center_weights_summary.csv"
if weights_path.is_file():
    weights_df = pd.read_csv(weights_path)
    display(weights_df)
    plot_softtriple_center_weight_bars(weights_df, fig_dir=FIG_DIR)
else:
    print("softtriple_center_weights_summary.csv absent.")
"""
        ),
        md("### Scores relaxés S (dist_* = -S)"),
        py(
            r"""
relaxed_path = ART.transfer_dir / "softtriple_relaxed_scores.csv"
if relaxed_path.is_file():
    display(pd.read_csv(relaxed_path).head())
plot_softtriple_relaxed_score_boxplot(pred, fig_dir=FIG_DIR)
plot_fsp_distance_boxplot(pred, fig_dir=FIG_DIR)
"""
        ),
        md("## § Distribution étapes et scores"),
        py(
            r"""
plot_fsp_pred_macro_distribution(pred, fig_dir=FIG_DIR)
plot_fsp_distribution_histograms(pred, fig_dir=FIG_DIR)
"""
        ),
        md("### Probabilités et distances"),
        py(
            r"""
prob_cols = [c for c in pred.columns if c.startswith("prob_")]
dist_cols = [c for c in pred.columns if c.startswith("dist_")]
print("Colonnes prob_* :", prob_cols)
print("Colonnes dist_* :", dist_cols)
display(pred[[c for c in ["pred_macro", "confidence", "margin", "entropy"] + prob_cols + dist_cols if c in pred.columns]].head())
"""
        ),
        md("## § Confusion, report, calibration"),
        py(
            r"""
if ART.confusion is not None and not ART.confusion.empty:
    display(ART.confusion)
    plot_fsp_confusion_heatmap(ART.confusion, fig_dir=FIG_DIR)
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
    ax.set_title("Calibration confiance (centres natifs)")
    ax.set_xlabel("confidence moyenne")
    ax.set_ylabel("accuracy")
    fig.savefig(FIG_DIR / "calibration.png", dpi=120, bbox_inches="tight")
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
        md("## § BERTopic — inputs et thèmes"),
        py(
            r"""
all_path = OUT_NATIVE / "transfer" / "bertopic_input_all.csv"
if all_path.is_file():
    all_df = pd.read_csv(all_path)
    print("bertopic_input_all.csv:", len(all_df))
    display(all_df.head())
else:
    print("bertopic_input_all.csv absent.")

for macro in ("A0", "A1", "B", "C"):
    p = OUT_NATIVE / "transfer" / f"bertopic_input_{macro}.csv"
    if p.is_file():
        print(f"{macro}: {len(pd.read_csv(p))} lignes")
    else:
        print(f"{macro}: fichier absent")
"""
        ),
        py(
            r"""
themes_path = OUT_NATIVE / "topics_bertopic" / "themes_by_macro.csv"
assign_path = OUT_NATIVE / "topics_bertopic" / "assignments.csv"
summary_path = OUT_NATIVE / "summary" / "topics_summary.csv"

if summary_path.is_file():
    display(pd.read_csv(summary_path))
else:
    print("topics_summary.csv absent.")

if themes_path.is_file():
    themes = pd.read_csv(themes_path)
    cols = [c for c in ["macro", "topic_id", "n_units", "theme_label", "top_words"] if c in themes.columns]
    display(themes[cols].head(20))
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
                "OUT_NATIVE",
                "FIG_DIR",
                restimate_var=None,
                topic_judge_cfg_var=None,
            )
        ),
        md("### Tableau topics par étape (format article / LaTeX)"),
        py(
            r"""
stats_path = OUT_NATIVE / "summary" / "macro_topic_stats.csv"
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
        latex_macro = table_macro.to_latex(index=False, escape=False, column_format="lcccc")
        print(latex_macro)
        tex_out = OUT_NATIVE / "summary" / "macro_topic_stats_table.tex"
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
