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


COMPARE_01 = """
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "resultats").is_dir() and (ROOT / "text" / "resultats").is_dir():
    ROOT = ROOT / "text"
elif (ROOT / "text").is_dir() and not (ROOT / "resultats").is_dir():
    ROOT = ROOT / "text"

table_path = ROOT / "resultats/comparisons/tables/embedding_geometry_comparison.csv"
if not table_path.is_file():
    raise FileNotFoundError(
        "Lancez d'abord : python scripts/collect_results.py && python scripts/compare_methods.py"
    )

df = pd.read_csv(table_path)
display(df)

for col in ["eta2_macro_balanced", "eta2_weighted", "rankme_global"]:
    if col in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(df["method"], df[col].astype(float))
        ax.set_title(col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

# Dans ce notebook : aucun entraînement — uniquement lecture de resultats/.
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
                        "Charge `resultats/comparisons/tables/embedding_geometry_comparison.csv`. "
                        "**Pas d'entraînement** dans ce notebook.",
                        "markdown",
                    ),
                    _cell(COMPARE_01),
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
