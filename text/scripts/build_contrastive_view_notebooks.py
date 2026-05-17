"""Génère un notebook de visualisation des résultats par méthode contrastive."""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

METHODS = (
    ("batch_triplet", "Batch Triplet", "batch_triplet"),
    ("softtriple", "SoftTriple", "softtriple"),
    ("supcon", "SupCon", "supcon"),
)

VIEW_CODE_TEMPLATE = '''
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

METHOD_KEY = "{method_key}"
DISPLAY_NAME = "{display_name}"
CONFIG_PATH = ROOT / "configs/methods/{config_file}"

cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
DATASET_CSV = ROOT / cfg.get("dataset_path", "dataset/data_btp.csv")
OUTPUT_DIR = cfg.get("output_dir", f"resultats/{{METHOD_KEY}}")
RESULTS = ROOT / OUTPUT_DIR
LABEL_COL = cfg.get("label_col", "pred_label")
BACKBONE = cfg.get("backbone_name", "(non défini)")

print("Méthode :", DISPLAY_NAME)
print("Config   :", CONFIG_PATH.relative_to(ROOT))
print("Dataset  :", DATASET_CSV)
print("Sorties  :", RESULTS)
print("Label    :", LABEL_COL)
print("Backbone :", BACKBONE)

if not RESULTS.is_dir():
    raise FileNotFoundError(
        f"Dossier absent : {{RESULTS}}\\n"
        f"Lancez : python scripts/train_{{METHOD_KEY}}.py"
    )

# Arborescence (2 niveaux)
for p in sorted(RESULTS.rglob("*"))[:80]:
    if p.is_file() and p.stat().st_size < 5_000_000:
        print(p.relative_to(RESULTS))
if len(list(RESULTS.rglob("*"))) > 80:
    print("…")

summary_csv = RESULTS / "grid_search_summary.csv"
if summary_csv.is_file():
    grid = pd.read_csv(summary_csv)
    display(grid.head(10))
    if "selection_score_mean_test_delta_ratio" in grid.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        top = grid.head(min(10, len(grid)))
        ax.barh(top["combo_id"].astype(str), top["selection_score_mean_test_delta_ratio"].astype(float))
        ax.set_xlabel("score sélection (delta_ratio test)")
        ax.set_title(f"{{DISPLAY_NAME}} — top combinaisons grid search")
        plt.tight_layout()
        plt.show()
else:
    print("Pas de grid_search_summary.csv (entraînement non lancé ou ancien format).")

for name in ("best_config.json", "final_full_data_training.json", "grid_search_summary.json"):
    path = RESULTS / name
    if path.is_file():
        print(f"\\n=== {{name}} ===")
        display(json.loads(path.read_text(encoding="utf-8")))

# Métriques agrégées k-fold (meilleure combo ou modèle final)
agg_candidates = sorted(RESULTS.rglob("*aggregate_metrics*.json"))
for path in agg_candidates[:3]:
    print(f"\\n=== {{path.relative_to(RESULTS)}} ===")
    display(json.loads(path.read_text(encoding="utf-8")))

geom_csv = RESULTS / "metrics" / "metrics_geometry.csv"
if geom_csv.is_file():
    geom = pd.read_csv(geom_csv)
    display(geom)
    for col in ("eta2_macro_balanced", "eta2_weighted", "rankme_global"):
        if col in geom.columns and geom[col].notna().any():
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(geom["method"].astype(str), geom[col].astype(float))
            ax.set_title(col)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
else:
    print(
        "Pas de metrics/metrics_geometry.csv — après export des embeddings, lancez :\\n"
        f"  python scripts/evaluate_embeddings_geometry.py --method_name {{METHOD_KEY}} "
        f"--metadata_csv <metadata.csv> --embeddings_csv <embeddings dim_*.csv>"
    )

# Embeddings exportés (recherche récursive)
def _has_dim_cols(path: Path) -> bool:
    cols = pd.read_csv(path, nrows=0).columns
    return any(str(c).startswith("dim_") for c in cols)

emb_csvs = [p for p in RESULTS.rglob("*.csv") if _has_dim_cols(p)]
if not emb_csvs:
    legacy_emb = list((ROOT / "legacy" / "contrastive_method_v0").rglob("fnembeddings_grid/*.csv"))
    emb_csvs = legacy_emb[:5]
if emb_csvs:
    print("Fichiers embeddings trouvés :")
    for p in emb_csvs[:5]:
        print(" ", p)
else:
    print("Aucun CSV dim_* trouvé sous", RESULTS)

# Lecture seule — pas d'entraînement dans ce notebook.
'''


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


def _setup_cell() -> str:
    return """
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "configs").is_dir() and (ROOT / "text" / "configs").is_dir():
    ROOT = ROOT / "text"
elif (ROOT / "text").is_dir() and not (ROOT / "configs").is_dir():
    ROOT = ROOT / "text"
"""


def main() -> None:
    NB.mkdir(parents=True, exist_ok=True)
    for method_key, display_name, config_file in METHODS:
        nb_name = f"05_view_{method_key}_results.ipynb"
        md = (
            f"# Résultats — {display_name}\n\n"
            f"Lecture des sorties sous `resultats/{method_key}/` (chemins définis dans "
            f"`configs/methods/{config_file}.yaml`). **Pas d'entraînement** — le corpus "
            f"dépend du CSV configuré, pas du nom du notebook.\n"
        )
        view_code = VIEW_CODE_TEMPLATE.format(
            method_key=method_key,
            display_name=display_name,
            config_file=config_file,
        )
        cells = [
            _cell(md, "markdown"),
            _cell(_setup_cell()),
            _cell(view_code),
        ]
        out = NB / nb_name
        out.write_text(json.dumps(_nb(cells), indent=1), encoding="utf-8")
        print("Écrit", out)


if __name__ == "__main__":
    main()
