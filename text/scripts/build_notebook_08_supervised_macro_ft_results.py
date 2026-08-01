"""Génère notebooks/08_view_supervised_macro_ft_results.ipynb.

Viewer lecture seule de la campagne de 8 variantes FT (tableau article)
+ drill-down confusions / prédictions pour une variante sélectionnée.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "08_view_supervised_macro_ft_results.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str, cell_id: str | None = None) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    cell: dict = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


TITLE_MD = """# Résultats supervised_macro_ft — variantes article

Notebook **lecture seule** : tableau des **8 variantes** FT
(encoder scope × projector) + confusions / prédictions par combo.

> Ce notebook n'entraîne pas. Lancez d'abord le job, puis exécutez les cellules.

## Lancer la campagne (cluster)

```bash
cd ~/SAFER/text
sbatch jobs/tune_supervised_macro_ft.sh
```

Grille : `configs/tuning/supervised_macro_ft_grid.yaml`  
Sorties : `output/supervised_macro_ft/variants/`

## Contenu

1. Tableau article (`results_summary.csv`)
2. Drill-down d'une variante (CV, OOD, confusions)
3. Checklist artefacts
"""

CONFIG_CODE = """# --- Paramètres ---
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display

ROOT = TEXT_ROOT  # bootstrap

from safer_core.brand_style import apply_matplotlib_brand
from supervised_macro_ft.notebook_viz import load_saved_predictions, plot_confusion_matrix_brand

apply_matplotlib_brand()
sns.set_theme(style="whitegrid")

VARIANTS_DIR = ROOT / "output/supervised_macro_ft/variants"
# Run unique (optionnel) :
# RESULTS_DIR = ROOT / "output/supervised_macro_ft"
# ou une combo : VARIANTS_DIR / "combos" / "<combo_id>"

TEST_CORPORA = ["metallurgie", "caou", "nicollin"]
DEFAULT_CFG = ROOT / "configs/methods/supervised_macro_ft.yaml"
if DEFAULT_CFG.is_file():
    _cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    TEST_CORPORA = list(_cfg.get("test_corpora") or TEST_CORPORA)

# Index de la variante à inspecter (0 = première ligne du tableau article)
COMBO_ROW = 0

FIGURES_DIR = VARIANTS_DIR / "figures_notebook"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Variants :", VARIANTS_DIR)
print("Figures  :", FIGURES_DIR)
print("Corpora  :", TEST_CORPORA)
"""

CHECK_MD = """## 0 — Artefacts de la campagne

Si des fichiers manquent : `sbatch jobs/tune_supervised_macro_ft.sh`
"""

CHECK_CODE = """required = {
    "results_summary": VARIANTS_DIR / "results_summary.csv",
    "grid_summary": VARIANTS_DIR / "grid_summary.csv",
    "combos_dir": VARIANTS_DIR / "combos",
}
missing = []
print("Artefacts :")
for k, p in required.items():
    ok = p.is_file() if p.suffix else p.is_dir()
    print(f"  [{'OK' if ok else 'ABSENT'}] {k}: {p}")
    if not ok:
        missing.append(k)
if missing:
    raise FileNotFoundError(
        "Campagne incomplète — fichiers manquants : "
        + ", ".join(missing)
        + "\\nLancez : sbatch jobs/tune_supervised_macro_ft.sh"
    )
print("\\nOK — artefacts principaux présents.")
"""

TABLE_MD = """## 1 — Tableau article (8 variantes)

Colonnes : Encoder scope × Projector × Construction CV (μ±σ) × OOD par corpus × Target avg / worst.
"""

TABLE_CODE = """results = pd.read_csv(VARIANTS_DIR / "results_summary.csv")
grid = pd.read_csv(VARIANTS_DIR / "grid_summary.csv")

# Affichage type article
article_cols = [
    c
    for c in (
        "encoder_scope",
        "projector",
        "cv_ba_mean",
        "cv_ba_std",
        *[f"ba_ood_{c}" for c in TEST_CORPORA],
        "ba_ood_avg",
        "ba_ood_worst",
        "combo_id",
    )
    if c in results.columns
]
display(results[article_cols].round(4))

# CV format μ±σ
if {"cv_ba_mean", "cv_ba_std"}.issubset(results.columns):
    pretty = results.copy()
    pretty["Construction"] = pretty.apply(
        lambda r: f"{r['cv_ba_mean']:.3f} ± {r['cv_ba_std']:.3f}",
        axis=1,
    )
    show = ["encoder_scope", "projector", "Construction"]
    show += [c for c in article_cols if c.startswith("ba_ood_")]
    display(pretty[[c for c in show if c in pretty.columns]])

# Heatmap encoder × projector (CV BA)
if {"encoder_scope", "projector", "cv_ba_mean"}.issubset(results.columns):
    pivot = results.pivot_table(
        index="encoder_scope",
        columns="projector",
        values="cv_ba_mean",
        aggfunc="first",
    )
    # ordre article
    scope_order = ["Last 1 layer", "Last 2 layers", "Last 3 layers", "Full encoder"]
    pivot = pivot.reindex([s for s in scope_order if s in pivot.index])
    if "Yes" in pivot.columns and "No" in pivot.columns:
        pivot = pivot[["Yes", "No"]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_title("CV balanced accuracy (BTP)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_cv_ba_heatmap.png", dpi=120, bbox_inches="tight")
    plt.show()
"""

DRILL_MD = """## 2 — Drill-down d'une variante

Choisissez `COMBO_ROW` (ou `COMBO_ID`) pour charger CV / metrics / confusions.
"""

DRILL_CODE = """# COMBO_ID = "train_last_n_layers1_projectionmlp_sklearn_........"  # optionnel
COMBO_ID = globals().get("COMBO_ID", None)

if COMBO_ID:
    row = results.loc[results["combo_id"].astype(str) == str(COMBO_ID)].iloc[0]
else:
    row = results.iloc[int(COMBO_ROW)]

combo_id = str(row["combo_id"])
combo_dir = Path(str(row.get("combo_dir") or (VARIANTS_DIR / "combos" / combo_id)))
if not combo_dir.is_dir():
    combo_dir = VARIANTS_DIR / "combos" / combo_id

print("Variante :", row.get("encoder_scope"), "| projector =", row.get("projector"))
print("combo_id :", combo_id)
print("dir      :", combo_dir)

cv_summary = combo_dir / "cv" / "cv_summary.csv"
if cv_summary.is_file():
    display(pd.read_csv(cv_summary))
else:
    print("(absent) cv/cv_summary.csv")

metrics_dir = combo_dir / "metrics"
if metrics_dir.is_dir():
    files = sorted(metrics_dir.glob("metrics_classification*.csv"))
    for f in files:
        print(f"\\n=== {f.name} ===")
        display(pd.read_csv(f))
else:
    print("(absent) metrics/")
"""

CONF_MD = """## 3 — Matrices de confusion (train BTP + OOD)

Prédictions écrites par le job sous `combos/<id>/predictions/`.
"""

CONF_CODE = """corpora = ["btp", *TEST_CORPORA]
for corpus_id in corpora:
    pred_df = load_saved_predictions(combo_dir, corpus_id)
    if pred_df is None:
        print(f"(absent) predictions_{corpus_id}.csv")
        continue
    true_col = "true_macro" if "true_macro" in pred_df.columns else "pred_label"
    pred_col = "pred_macro" if "pred_macro" in pred_df.columns else None
    if pred_col is None:
        print(f"(skip) pas de pred_macro pour {corpus_id}")
        continue
    print(f"\\n=== Confusion — {corpus_id} (n={len(pred_df)}) ===")
    plot_confusion_matrix_brand(
        pred_df[true_col],
        pred_df[pred_col],
        title=f"{row.get('encoder_scope')} / {row.get('projector')} — {corpus_id}",
        fig_dir=FIGURES_DIR,
        filename=f"02_confusion_{combo_id[:40]}_{corpus_id}.png",
    )
"""

FOOTER_MD = """---

**Figures** : `{VARIANTS_DIR}/figures_notebook/`

Run unitaire (hors campagne) : `sbatch jobs/train_supervised_macro_ft.sh`  
Baseline sklearn Qwen : notebook **07** / **07b**.
"""


def build_notebook() -> dict:
    cells = [
        md(TITLE_MD),
        py(NOTEBOOK_PATH_SETUP, cell_id="bootstrap"),
        py(CONFIG_CODE, cell_id="config"),
        md(CHECK_MD),
        py(CHECK_CODE, cell_id="check"),
        md(TABLE_MD),
        py(TABLE_CODE, cell_id="table"),
        md(DRILL_MD),
        py(DRILL_CODE, cell_id="drill"),
        md(CONF_MD),
        py(CONF_CODE, cell_id="confusion"),
        md(FOOTER_MD),
    ]
    return {
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
    }


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Écrit : {NB_PATH}")


if __name__ == "__main__":
    main()
