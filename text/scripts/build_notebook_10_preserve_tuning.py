"""Génère notebooks/10_preserve_tuning.ipynb (courbes λ_pres vs métriques)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "10_preserve_tuning.ipynb"

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
# 10 — Tuning λ_pres (preserve) — macro_transfer TPN

Compare l'effet de `loss_weights.preserve` (λ_pres) sur :
- **Macro** : MacroF1, BalancedAccuracy, Entropy (phases initial et adapté)
- **Topics** : R_m (part du plus gros topic), K_m (nombre de topics), r_noise (bruit BERTopic)

**Prérequis** :
```bash
python scripts/run_tpn_macro_transfer_preserve_tuning.py --corpus metallurgie
```

CSV global : `output_test/<TEST_CORPUS>/macro_transfer/preserve_tuning_metrics.csv`
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
METRICS_CSV = TEXT_ROOT / "output_test" / TEST_CORPUS / "macro_transfer" / "preserve_tuning_metrics.csv"
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"10_preserve_tuning_{TEST_CORPUS}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if not METRICS_CSV.is_file():
    raise FileNotFoundError(
        f"Métriques absentes : {METRICS_CSV}\n"
        "Lancer : python scripts/run_tpn_macro_transfer_preserve_tuning.py"
    )

df = pd.read_csv(METRICS_CSV)
df = df.sort_values(["base_method", "lambda_pres"]).reset_index(drop=True)
print("Lignes :", len(df), "| encodeurs :", df["base_method"].unique().tolist())
display(df.head(20))
sns.set_theme(style="whitegrid")
"""
        ),
        md("## § MacroF1 et BalancedAccuracy (initial vs adapté)"),
        py(
            r"""
def _plot_metric_vs_lambda(data, y_col, title, fname):
    if y_col not in data.columns:
        print("Colonne absente :", y_col)
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for enc, sub in data.groupby("base_method"):
        ax.plot(
            sub["lambda_pres"],
            sub[y_col],
            marker="o",
            label=str(enc),
        )
    ax.set_xlabel("lambda_pres (loss_weights.preserve)")
    ax.set_ylabel(y_col)
    ax.set_title(f"{title} — {TEST_CORPUS}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.show()

for col, title, fname in [
    ("macro_f1_initial", "Macro-F1 initial", "macro_f1_initial.png"),
    ("macro_f1_adapted", "Macro-F1 adapté", "macro_f1_adapted.png"),
    ("balanced_accuracy_initial", "Balanced accuracy initial", "bal_acc_initial.png"),
    ("balanced_accuracy_adapted", "Balanced accuracy adapté", "bal_acc_adapted.png"),
]:
    _plot_metric_vs_lambda(df, col, title, fname)
"""
        ),
        md("## § Entropie moyenne (gating)"),
        py(
            r"""
for col, title, fname in [
    ("entropy_initial", "Entropie moyenne initial", "entropy_initial.png"),
    ("entropy_adapted", "Entropie moyenne adapté", "entropy_adapted.png"),
]:
    _plot_metric_vs_lambda(df, col, title, fname)
"""
        ),
        md("## § Topics : R_m, K_m, r_noise (agrégés pondérés)"),
        py(
            r"""
for col, title, fname in [
    ("R_m", "R_m (part plus gros topic, pondéré)", "R_m.png"),
    ("K_m", "K_m (nombre de topics, pondéré)", "K_m.png"),
    ("r_noise", "r_noise (bruit BERTopic, pondéré)", "r_noise.png"),
]:
    _plot_metric_vs_lambda(df, col, title, fname)
"""
        ),
        md("## § R_m par macro (A0–C)"),
        py(
            r"""
macro_cols = [c for c in df.columns if c.startswith("R_m_") and c != "R_m"]
if macro_cols:
    fig, ax = plt.subplots(figsize=(9, 4))
    for enc, sub in df.groupby("base_method"):
        for mc in macro_cols:
            ax.plot(
                sub["lambda_pres"],
                sub[mc],
                marker="o",
                alpha=0.7,
                label=f"{enc} {mc.replace('R_m_', '')}",
            )
    ax.set_xlabel("lambda_pres")
    ax.set_ylabel("R_m")
    ax.set_title(f"R_m par macro — {TEST_CORPUS}")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG_DIR / "R_m_by_macro.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("Pas de colonnes R_m_<macro> (relancer le tuning avec BERTopic)")
"""
        ),
        md("## § Compromis : gain MacroF1 vs dégradation R_m / topics"),
        py(
            r"""
if {"macro_f1_adapted", "R_m", "K_m"}.issubset(df.columns):
    fig, ax = plt.subplots(figsize=(7, 5))
    for enc, sub in df.groupby("base_method"):
        ax.scatter(
            sub["macro_f1_adapted"],
            sub["R_m"],
            s=80,
            label=enc,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['lambda_pres']:.2g}",
                (row["macro_f1_adapted"], row["R_m"]),
                fontsize=7,
                alpha=0.8,
            )
    ax.set_xlabel("Macro-F1 adapté")
    ax.set_ylabel("R_m (pondéré)")
    ax.set_title(f"Compromis classification / structure topics — {TEST_CORPUS}")
    ax.legend()
    fig.savefig(FIG_DIR / "tradeoff_macro_f1_vs_Rm.png", dpi=120, bbox_inches="tight")
    plt.show()

    trade = df[["base_method", "lambda_pres", "macro_f1_adapted", "R_m", "K_m", "r_noise"]].copy()
    display(trade)
else:
    print("Colonnes manquantes pour le graphique compromis")
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
