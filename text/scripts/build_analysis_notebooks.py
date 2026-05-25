"""Génère les notebooks d'analyse (lecture seule, sans entraînement)."""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"


def _cell(code: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in code.strip().split("\n")],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


COMPARE_01_SETUP = """
import os
import sys
from pathlib import Path

import pandas as pd


def _find_text_root(start: Path) -> Path:
    \"\"\"Racine text/ (contient safer_core/ ou metrics/).\"\"\"
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "safer_core" / "paths.py").is_file():
            return candidate
        nested = candidate / "text"
        if (nested / "safer_core" / "paths.py").is_file():
            return nested
    return here


ROOT = _find_text_root(Path.cwd())
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.compare_display import (
    EMBEDDING_COMPARE_METHODS,
    METHOD_DISPLAY,
    kfold_barplot_frame,
    kfold_geometry_display_table,
    slim_geometry_table,
)
from safer_core.paths import ensure_comparisons_dirs

TABLES = ensure_comparisons_dirs() / "tables"
# Rétrocompat si anciens runs sous resultats/
_legacy_tables = ROOT / "resultats/comparisons/tables"
if not (TABLES / "embedding_geometry_comparison_btp.csv").is_file() and _legacy_tables.is_dir():
    TABLES = _legacy_tables

PATH_BTP = TABLES / "embedding_geometry_comparison_btp.csv"
PATH_BTP_KFOLD = TABLES / "embedding_geometry_comparison_btp_kfold.csv"
PATH_TEST = TABLES / "embedding_geometry_comparison_test.csv"
EXPECTED = [METHOD_DISPLAY[k] for k in EMBEDDING_COMPARE_METHODS]

if not PATH_BTP.is_file():
    raise FileNotFoundError(
        "Lancez d'abord : python scripts/collect_results.py\\n"
        f"  (attendu : {TABLES / 'embedding_geometry_comparison_btp.csv'})"
    )

df_btp = pd.read_csv(PATH_BTP)
df_btp_kfold = pd.read_csv(PATH_BTP_KFOLD) if PATH_BTP_KFOLD.is_file() else pd.DataFrame()
df_test = pd.read_csv(PATH_TEST) if PATH_TEST.is_file() else pd.DataFrame()

for frame, corpus_name in [(df_btp, "BTP"), (df_test, "test métallurgie")]:
    present = (
        set(frame["method"].astype(str))
        if not frame.empty and "method" in frame.columns
        else set()
    )
    for name in EXPECTED:
        if name not in present:
            print(f"[absent] {name} — pas de métriques {corpus_name} (relancer fit final + éval)")
"""

COMPARE_01_DISPLAY = """
import matplotlib.pyplot as plt

print("## Corpus BTP — validation K-fold (μ ± σ sur les folds)")
if df_btp_kfold.empty:
    print(
        "(absent) Relancer collect_results.py après entraînement K-fold "
        "(metrics/kfold_summary.csv par méthode). Embedding brut : pas de K-fold."
    )
else:
    display(kfold_geometry_display_table(df_btp_kfold))

print("\\n## Corpus BTP — fit final 100 % (métriques sur tout le BTP)")
display(slim_geometry_table(df_btp))

print("\\n## Corpus test — métallurgie")
if df_test.empty:
    print("(tableau test absent — python scripts/collect_results.py après éval test)")
else:
    display(slim_geometry_table(df_test))

BAR_METRICS = [
    ("eta2_macro_balanced_perc", "η² macro balancé (%)"),
    ("rankme_over_d", "RankMe / d"),
]

def _barplot_corpus(df, title_prefix):
    if df.empty:
        return
    for col, ylab in BAR_METRICS:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["method"].astype(str), df[col].astype(float))
        ax.set_ylabel(ylab)
        ax.set_title(f"{title_prefix} — {ylab}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

def _barplot_kfold(df_kfold, title_prefix):
    if df_kfold.empty:
        return
    for col, ylab in BAR_METRICS:
        frame = kfold_barplot_frame(df_kfold, col)
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            frame["method"].astype(str),
            frame["mean"].astype(float),
            yerr=frame["std"].astype(float),
            capsize=4,
        )
        ax.set_ylabel(ylab)
        ax.set_title(f"{title_prefix} — {ylab}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

if not df_btp_kfold.empty:
    _barplot_kfold(df_btp_kfold, "BTP K-fold val")
_barplot_corpus(df_btp, "BTP fit final")
_barplot_corpus(df_test, "Test métallurgie")

# Lecture seule — aucun entraînement.
"""

CHECK_00 = """
import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / "text" / "dataset").is_dir():
    ROOT = ROOT / "text"
df = pd.read_csv(ROOT / "dataset/data_btp.csv", nrows=5)
print("Colonnes :", list(df.columns))
print("Lignes (aperçu) :", len(pd.read_csv(ROOT / "dataset/data_btp.csv")))
if "pred_label" in df.columns:
    print(pd.read_csv(ROOT / "dataset/data_btp.csv")["pred_label"].value_counts())
"""


def main() -> None:
    import sys

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from build_contrastive_view_notebooks import main as build_contrastive_views

    NB.mkdir(parents=True, exist_ok=True)
    (NB / "00_check_data.ipynb").write_text(
        json.dumps(_nb([_cell("# Vérification des données", "markdown"), _cell(CHECK_00)]), indent=1),
        encoding="utf-8",
    )
    (NB / "01_compare_embedding_methods.ipynb").write_text(
        json.dumps(
            _nb(
                [
                    _cell(
                        "# Comparaison des méthodes d'embedding\n\n"
                        "Cinq lignes : **Embedding brut**, **Batch Triplet**, **SupCon**, "
                        "**SoftTriple**, **SCGM** (BTP + test métallurgie).\n\n"
                        "Prérequis : `python scripts/collect_results.py` "
                        "(BTP fit final, **BTP K-fold μ±σ**, test). "
                        "**Pas d'entraînement** dans ce notebook.",
                        "markdown",
                    ),
                    _cell(COMPARE_01_SETUP),
                    _cell(COMPARE_01_DISPLAY),
                ]
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print("Écrit notebooks 00 et 01 dans", NB)
    build_contrastive_views()


if __name__ == "__main__":
    main()
