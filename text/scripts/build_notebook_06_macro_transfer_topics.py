"""Génère notebooks/06_macro_transfer_topics.ipynb (baseline Frozen Source Prototypes)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "06_macro_transfer_topics.ipynb"

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
# 06 — Frozen Source Prototypes + topics par étape (chaîne accidentelle)

Notebook orienté baseline **Frozen Source Prototypes** :
- prédictions étape (chaîne accidentelle) : `transfer/target_macro_predictions.csv`
- prototypes source : `transfer/source_prototypes.csv`
- métriques : `transfer/metrics.json` (si labels cible)
- BERTopic inputs : `transfer/bertopic_input_all.csv` et `transfer/bertopic_input_<macro>.csv`
- BERTopic sorties : `topics_bertopic/assignments.csv`, `topics_bertopic/themes_by_macro.csv`, `summary/topics_summary.csv`

**Prérequis** : `BASE_METHOD=scgm_text CORPUS=<id> bash jobs/run_frozen_source_prototypes.sh`.
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from safer_core.test_corpus import resolve_test_corpus, macro_transfer_output_dir
from macro_transfer.notebook_viz import (
    load_fsp_run_artifacts,
    plot_fsp_distribution_histograms,
    plot_fsp_pred_macro_distribution,
    plot_fsp_distance_boxplot,
    plot_fsp_confusion_heatmap,
    compute_fsp_confidence_calibration,
    get_fsp_top_confident_errors,
)

# --- Parameters (modifier ici ou via papermill) ---
TEST_CORPUS = "metallurgie"
FSP_BASE_METHOD = "scgm_text"
TOP_ERRORS_K = 20

_spec = resolve_test_corpus(TEST_CORPUS, anchor=TEXT_ROOT)
OUT_DIR = macro_transfer_output_dir(f"frozen_source_prototypes/{FSP_BASE_METHOD}", _spec.id, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"06_fsp_macro_{_spec.id}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Corpus test : {_spec.display_name} ({_spec.id})")
print("Frozen Source Prototypes :", OUT_DIR)

sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Chargement artefacts"),
        py(
            r"""
ART = load_fsp_run_artifacts(OUT_DIR)
pred = ART.predictions.copy()
protos = ART.prototypes.copy()
metrics = ART.metrics or {}
print("n target:", len(pred))
print("n prototypes:", len(protos))
display(protos.head())
if metrics:
    display(pd.DataFrame([metrics]))
else:
    print("metrics.json absent (normal si labels cible indisponibles).")
"""
        ),
        py(
            r"""
required = {"pred_macro", "confidence"}
missing = sorted(required.difference(pred.columns))
if missing:
    raise KeyError(f"Colonnes manquantes dans target_macro_predictions.csv: {missing}")
display(pred.head(5))
"""
        ),
        md("## § Graphes FSP principaux"),
        py(
            r"""
plot_fsp_pred_macro_distribution(pred, fig_dir=FIG_DIR)
plot_fsp_distribution_histograms(pred, fig_dir=FIG_DIR)
plot_fsp_distance_boxplot(pred, fig_dir=FIG_DIR)
"""
        ),
        md("## § Calibration confiance et erreurs"),
        py(
            r"""
calib = compute_fsp_confidence_calibration(pred)
if calib is not None and not calib.empty:
    display(calib)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(calib["mean_confidence"], calib["accuracy"], "o-")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("confidence moyenne (bin)")
    ax.set_ylabel("accuracy (bin)")
    ax.set_title("Calibration confiance (quantiles)")
    fig.savefig(FIG_DIR / "calibration_confidence_vs_accuracy.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("Calibration non disponible (true_macro absent).")

err = get_fsp_top_confident_errors(pred, top_k=TOP_ERRORS_K)
if err.empty:
    print("Pas d'erreurs confiantes (ou true_macro absent).")
else:
    cols = [c for c in ["sentence", "true_macro", "pred_macro", "confidence", "margin", "entropy"] if c in err.columns]
    display(err[cols])
"""
        ),
        md("## § Matrice de confusion / report classification"),
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
"""
        ),
        md("## § Inputs BERTopic (fichiers transfer/)"),
        py(
            r"""
bertopic_all = OUT_DIR / "transfer" / "bertopic_input_all.csv"
if bertopic_all.is_file():
    df_all = pd.read_csv(bertopic_all)
    print("bertopic_input_all.csv:", len(df_all), "lignes")
    display(df_all.head())
else:
    print("bertopic_input_all.csv absent.")

for macro in ("A0", "A1", "B", "C"):
    p = OUT_DIR / "transfer" / f"bertopic_input_{macro}.csv"
    if p.is_file():
        d = pd.read_csv(p)
        print(f"{p.name}: {len(d)} lignes")
    else:
        print(f"{p.name}: absent")
"""
        ),
        md("## § Sorties BERTopic (topics + labels)"),
        py(
            r"""
topics_dir = OUT_DIR / "topics_bertopic"
assign_path = topics_dir / "assignments.csv"
themes_path = topics_dir / "themes_by_macro.csv"
summary_path = OUT_DIR / "summary" / "topics_summary.csv"

if summary_path.is_file():
    print("topics_summary.csv")
    display(pd.read_csv(summary_path))
else:
    print("topics_summary.csv absent.")

if themes_path.is_file():
    th = pd.read_csv(themes_path)
    print("themes_by_macro.csv :", len(th), "lignes")
    show_cols = [c for c in ["macro", "topic_id", "n_units", "theme_label", "top_words"] if c in th.columns]
    display(th[show_cols].head(20))
    if "theme_label" not in th.columns:
        print("theme_label absent (fallback top_words).")
else:
    print("themes_by_macro.csv absent.")

if assign_path.is_file():
    ass = pd.read_csv(assign_path)
    print("assignments.csv :", len(ass), "lignes")
    display(ass.head())
else:
    print("assignments.csv absent.")
"""
        ),
        md(notebook_topic_judge_section_md()),
        py(
            notebook_topic_judge_source(
                "OUT_DIR",
                "FIG_DIR",
                restimate_var=None,
                topic_judge_cfg_var=None,
            )
        ),
        md("## § Fichiers attendus / robustesse"),
        py(
            r"""
expected = [
    OUT_DIR / "transfer" / "target_macro_predictions.csv",
    OUT_DIR / "transfer" / "source_prototypes.csv",
    OUT_DIR / "transfer" / "metrics.json",
    OUT_DIR / "transfer" / "bertopic_input_all.csv",
    OUT_DIR / "topics_bertopic" / "assignments.csv",
    OUT_DIR / "topics_bertopic" / "themes_by_macro.csv",
]
for p in expected:
    print(("OK " if p.is_file() else "MISSING "), p)
print("Figures :", FIG_DIR)
"""
        ),
        md("## § Tableau topics par étape (format article / LaTeX)"),
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
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("OK:", NB_PATH)


if __name__ == "__main__":
    main()
