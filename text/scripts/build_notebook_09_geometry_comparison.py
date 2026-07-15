"""Génère notebooks/09_geometry_comparison.ipynb (comparaison η² multi-méthodes)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "09_geometry_comparison.ipynb"

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


INTRO_MD = """# Comparaison géométrique (η² macro balanced)

**Exécutable** — compare la **séparation géométrique macro** entre plusieurs méthodes d'embedding
(Qwen brut, contrastives, fine-tuning supervisé, etc.) sur les corpus BTP, métallurgie et caou.

## Indicateur

Métrique principale : **`eta2_macro_balanced_perc`** (η² macro balanced, en %).

Pour chaque macro-classe *k* (A0, A1, B, C), on calcule l'inertie intra (*Wₖ*) et inter (*Bₖ*) sur les
embeddings (optionnellement L2-normalisés). Les macros avec trop peu d'échantillons sont ignorées.

- **η² macro balanced** = 1 − W/T sur les inerties **macro-balanced** (chaque macro compte autant,
  indépendamment de sa taille).
- **η² weighted** = 1 − W/T sur les inerties pondérées par effectif.

Implémentation : `metrics/embedding_geometry_separation.py` (`build_geometry_metrics_row`).
Pas de classifieur : purement géométrique.

## Configuration des dossiers

En tête de notebook, **`METHOD_SPECS`** liste les méthodes à comparer —
**un dossier `results_dir` par méthode** (run standard, combo tuning, chemins absolus, etc.) :

```python
METHOD_SPECS = [
    {"name": "Qwen brut", "kind": "raw"},  # CSV Qwen du registre
    {"name": "Batch Triplet", "kind": "projected", "results_dir": ROOT / "output/batch_triplet"},
    {"name": "SoftTriple", "kind": "projected", "results_dir": ROOT / "output/softtriple/tuning/combos/..."},
]
```

| `kind` | Rôle |
|--------|------|
| `raw` | Embeddings Qwen bruts (CSV registre `configs/test_corpora.yaml`) — pas de `results_dir` |
| `projected` | Run entraîné : `results_dir/embeddings/projected_<corpus>.npy` + `_metadata.csv` |

Exemples de chemins (`results_dir`, relatif à `ROOT` ou absolu) :

- `output/batch_triplet`
- `output/softtriple`
- `output/supcon`
- `output/supervised_macro_ft`
- Combo tuning : `output/softtriple/tuning/combos/...`

**Prérequis** : exports déjà produits par les scripts d'entraînement / évaluation.
Pour Qwen brut : CSV d'embeddings exportés (`emb_csv` du registre). Si des lignes sont ignorées
à la fusion (CSV obsolète), relancer :
`python scripts/export_corpus_embeddings.py --corpus <id> --force`

## Sorties

- Tableau récapitulatif **toutes méthodes × tous corpus**
- Trois barplots séparés (BTP, métallurgie, caou)
- CSV : `output/geometry_comparison/geometry_comparison_summary.csv`
"""

PARAMS_CODE = """# --- Paramètres (modifier ici) ---
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = TEXT_ROOT  # défini par la cellule bootstrap

from safer_core.brand_style import apply_matplotlib_brand

apply_matplotlib_brand()
sns.set_theme(style="whitegrid")

# Méthodes à comparer — adapter results_dir à vos runs
METHOD_SPECS = [
    {"name": "Qwen brut", "kind": "raw"},
    {"name": "Batch Triplet", "kind": "projected", "results_dir": ROOT / "output/batch_triplet"},
    {"name": "SoftTriple", "kind": "projected", "results_dir": ROOT / "output/softtriple"},
    {"name": "SupCon", "kind": "projected", "results_dir": ROOT / "output/supcon"},
    {
        "name": "supervised_macro_ft",
        "kind": "projected",
        "results_dir": ROOT / "output/supervised_macro_ft",
    },
]

CORPORA = ["btp", "metallurgie", "caou"]
LABEL_COL = "pred_label"
GEOMETRY_METRIC = "eta2_macro_balanced_perc"
L2_NORMALIZE = False  # True = normaliser L2 avant η² (comme certaines évaluations val)

OUTPUT_DIR = ROOT / "output/geometry_comparison"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SKIP_ERRORS = True  # False = lever l'erreur si un run/corpus manque

print("Méthodes :", [s["name"] for s in METHOD_SPECS])
print("Corpus   :", CORPORA)
print("Figures  :", FIGURES_DIR)
"""

TABLE_CODE = """from metrics.geometry_comparison import build_geometry_comparison_table

summary = build_geometry_comparison_table(
    METHOD_SPECS,
    CORPORA,
    label_col=LABEL_COL,
    l2_normalize=L2_NORMALIZE,
    anchor=ROOT,
    skip_errors=SKIP_ERRORS,
)

print(f"Lignes : {len(summary)}")
display(summary)
"""

BARPLOTS_CODE = """from metrics.geometry_comparison import plot_geometry_comparison_bars

for corpus_id in CORPORA:
    plot_geometry_comparison_bars(
        summary,
        corpus_id,
        metric=GEOMETRY_METRIC,
        fig_dir=FIGURES_DIR,
        show=True,
    )
"""

EXPORT_CODE = """csv_path = OUTPUT_DIR / "geometry_comparison_summary.csv"
summary.to_csv(csv_path, index=False)
print("Export :", csv_path)
"""

cells = [
    md(INTRO_MD),
    py(NOTEBOOK_PATH_SETUP),
    py(PARAMS_CODE),
    py(TABLE_CODE),
    py(BARPLOTS_CODE),
    py(EXPORT_CODE),
]

nb = {
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

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Écrit :", NB_PATH)
