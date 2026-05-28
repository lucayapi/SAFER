"""Génère notebooks/08_tpn_macro_transfer_results.ipynb (diagnostics FSP)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "08_tpn_macro_transfer_results.ipynb"

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
# 08 — Diagnostics Frozen Source Prototypes

Artefacts sous `output_test/<TEST_CORPUS>/macro_transfer/frozen_source_prototypes/transfer/`.

**Prérequis** : `python scripts/run_frozen_source_prototypes.py --config configs/frozen_source_prototypes.yaml`
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from safer_core.test_corpus import macro_transfer_output_dir
from macro_transfer.notebook_viz import (
    load_fsp_run_artifacts,
    plot_fsp_distribution_histograms,
    plot_fsp_pred_macro_distribution,
    plot_fsp_distance_boxplot,
    plot_fsp_confusion_heatmap,
    compute_fsp_confidence_calibration,
    get_fsp_top_confident_errors,
)

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
OUT_DIR = macro_transfer_output_dir("frozen_source_prototypes", TEST_CORPUS, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"08_fsp_{TEST_CORPUS}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Corpus test :", TEST_CORPUS)
print("OUT_DIR :", OUT_DIR)
print("Predictions file :", OUT_DIR / "transfer" / "target_macro_predictions.csv")
sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Chargement et résumé"),
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
        md("## § Comparaison transfert direct (raw vs méthode source)"),
        py(
            r"""
from pathlib import Path
import math

ROOT_FSP = OUT_DIR.parent
RAW_METRICS = ROOT_FSP / "raw" / "transfer" / "metrics.json"
SCGM_METRICS = ROOT_FSP / "scgm" / "transfer" / "metrics.json"
TABLE_CSV = ROOT_FSP / "table_transfer_direct.csv"
TABLE_TEX = ROOT_FSP / "table_transfer_direct.tex"
ROOT_FSP.mkdir(parents=True, exist_ok=True)

def _load_metrics(path: Path):
    if not path.is_file():
        print(f"[WARN] metrics.json absent: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def _to_float(v):
    try:
        x = float(v)
        if math.isnan(x):
            return np.nan
        return x
    except Exception:
        return np.nan

raw_m = _load_metrics(RAW_METRICS) or {}
scgm_m = _load_metrics(SCGM_METRICS) or {}

row_raw = {
    "Méthode": "Embedding brut + prototypes source",
    "Bal. Acc.": _to_float(raw_m.get("balanced_accuracy", np.nan)),
    "Macro-F1": _to_float(raw_m.get("macro_f1", np.nan)),
    "Confiance moy.": _to_float(raw_m.get("mean_confidence", np.nan)),
    "Entropie moy.": _to_float(raw_m.get("mean_entropy", np.nan)),
}
scgm_method = scgm_m.get("method", "SCGM + prototypes source gelés")
row_scgm = {
    "Méthode": str(scgm_method),
    "Bal. Acc.": _to_float(scgm_m.get("balanced_accuracy", np.nan)),
    "Macro-F1": _to_float(scgm_m.get("macro_f1", np.nan)),
    "Confiance moy.": _to_float(scgm_m.get("mean_confidence", np.nan)),
    "Entropie moy.": _to_float(scgm_m.get("mean_entropy", np.nan)),
}

table_df = pd.DataFrame([row_raw, row_scgm])
display(table_df)
table_df.to_csv(TABLE_CSV, index=False)

def _winner_indices(values, mode="max"):
    s = pd.Series(values, dtype="float64")
    if s.notna().sum() == 0:
        return set()
    best = s.max() if mode == "max" else s.min()
    return set(s.index[s == best].tolist())

def _fmt(v, bold=False):
    if pd.isna(v):
        return "--"
    txt = f"{float(v):.4f}"
    return f"\\textbf{{{txt}}}" if bold else txt

best_bal = _winner_indices(table_df["Bal. Acc."], mode="max")
best_f1 = _winner_indices(table_df["Macro-F1"], mode="max")
best_conf = _winner_indices(table_df["Confiance moy."], mode="max")
best_ent = _winner_indices(table_df["Entropie moy."], mode="min")

lines = []
lines.append("\\begin{tabular}{lcccc}")
lines.append("\\toprule")
lines.append("\\textbf{Méthode} & \\textbf{Bal. Acc.} & \\textbf{Macro-F1} & \\textbf{Confiance moy.} & \\textbf{Entropie moy.} \\\\")
lines.append("\\midrule")
for i, row in table_df.iterrows():
    lines.append(
        f"{row['Méthode']} & "
        f"{_fmt(row['Bal. Acc.'], i in best_bal)} & "
        f"{_fmt(row['Macro-F1'], i in best_f1)} & "
        f"{_fmt(row['Confiance moy.'], i in best_conf)} & "
        f"{_fmt(row['Entropie moy.'], i in best_ent)} \\\\"
    )
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
latex_table = "\n".join(lines)
print(latex_table)
TABLE_TEX.write_text(latex_table + "\n", encoding="utf-8")
print("CSV :", TABLE_CSV)
print("TEX :", TABLE_TEX)
"""
        ),
        md("## § Distribution macro et scores"),
        py(
            r"""
plot_fsp_pred_macro_distribution(pred, fig_dir=FIG_DIR)
plot_fsp_distribution_histograms(pred, fig_dir=FIG_DIR)
plot_fsp_distance_boxplot(pred, fig_dir=FIG_DIR)
"""
        ),
        md("## § Probabilités et distances"),
        py(
            r"""
prob_cols = [c for c in pred.columns if c.startswith("prob_")]
dist_cols = [c for c in pred.columns if c.startswith("dist_")]
print("Colonnes prob_* :", prob_cols)
print("Colonnes dist_* :", dist_cols)
display(pred[[c for c in ["pred_macro", "confidence", "margin", "entropy"] + prob_cols[:4] + dist_cols[:4] if c in pred.columns]].head())
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
    ax.set_title("Calibration confiance")
    ax.set_xlabel("confidence moyenne")
    ax.set_ylabel("accuracy")
    fig.savefig(FIG_DIR / "calibration.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("Calibration non disponible.")
"""
        ),
        md("## § Erreurs à forte confiance"),
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
        md("## § Inputs BERTopic"),
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
