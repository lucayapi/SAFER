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
# 06 — Frozen Source Prototypes + topics intra-macro

Notebook orienté baseline **Frozen Source Prototypes** :
- prédictions macro : `transfer/target_macro_predictions.csv`
- prototypes source : `transfer/source_prototypes.csv`
- métriques : `transfer/metrics.json` (si labels cible)
- BERTopic inputs : `transfer/bertopic_input_all.csv` et `transfer/bertopic_input_<macro>.csv`

**Prérequis** : exécuter `python scripts/run_frozen_source_prototypes.py --config configs/frozen_source_prototypes.yaml`.
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
TOP_ERRORS_K = 20

_spec = resolve_test_corpus(TEST_CORPUS, anchor=TEXT_ROOT)
OUT_DIR = macro_transfer_output_dir("frozen_source_prototypes", _spec.id, anchor=TEXT_ROOT)
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
        md("## § Fichiers attendus / robustesse"),
        py(
            r"""
expected = [
    OUT_DIR / "transfer" / "target_macro_predictions.csv",
    OUT_DIR / "transfer" / "source_prototypes.csv",
    OUT_DIR / "transfer" / "metrics.json",
    OUT_DIR / "transfer" / "bertopic_input_all.csv",
]
for p in expected:
    print(("OK " if p.is_file() else "MISSING "), p)
print("Figures :", FIG_DIR)
"""
        ),
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
