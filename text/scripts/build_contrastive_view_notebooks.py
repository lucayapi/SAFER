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

train_log = RESULTS / "metrics" / "train_log.csv"
if train_log.is_file():
    tl = pd.read_csv(train_log)
    display(tl.tail(10))
    ycol = "val_delta_macro_pct" if "val_delta_macro_pct" in tl.columns else "val_loss"
    if ycol in tl.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tl["epoch"], tl[ycol], marker="o", label=ycol)
        if "train_loss" in tl.columns:
            ax.plot(tl["epoch"], tl["train_loss"], marker="o", label="train_loss")
        ax.set_xlabel("epoch")
        ax.set_title(f"{{DISPLAY_NAME}} — courbe d'apprentissage")
        ax.legend()
        plt.tight_layout()
        plt.show()
else:
    print("Pas de metrics/train_log.csv (entraînement non lancé).")

tuning_summary = RESULTS / "tuning" / "grid_summary.csv"
if tuning_summary.is_file():
    print("\\n=== Tuning grid_summary ===")
    tdf = pd.read_csv(tuning_summary)
    display(tdf.sort_values("selection_score", ascending=False).head(15))
    if "selection_score" in tdf.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        top = tdf.sort_values("selection_score", ascending=False).head(min(12, len(tdf)))
        ax.barh(top["combo_id"].astype(str), top["selection_score"].astype(float))
        ax.set_xlabel("δ_macro (%) = selection_score")
        ax.set_title(f"{{DISPLAY_NAME}} — grille tuning")
        plt.tight_layout()
        plt.show()

resolved_cfg = RESULTS / "configs" / "config_resolved.yaml"
if resolved_cfg.is_file():
    print("\\n=== config_resolved.yaml ===")
    display(yaml.safe_load(resolved_cfg.read_text(encoding="utf-8")))

DATA_TEST = ROOT / "dataset/test/data_metallurgie.csv"
print("Corpus test (hors domaine) :", DATA_TEST)

print(
    "\\nNote : le test métallurgie utilise le modèle entraîné sur 100 % BTP "
    "(checkpoints/best_model), pas les checkpoints des folds."
)

kfold_summary = RESULTS / "metrics" / "kfold_summary.csv"
kfold_per_fold = RESULTS / "metrics" / "kfold_per_fold.csv"

if kfold_per_fold.is_file():
    print("\\n=== K-fold validation (par fold) ===")
    display(pd.read_csv(kfold_per_fold))

if kfold_summary.is_file():
    print("\\n=== K-fold validation (μ±σ) ===")
    kval = pd.read_csv(kfold_summary)
    display(kval)
    if "mean_delta_macro_pct" in kval.columns:
        m = float(kval["mean_delta_macro_pct"].iloc[0])
        s = float(kval.get("std_delta_macro_pct", pd.Series([0])).iloc[0])
        print(f"δ_macro val : {{m:.2f}} ± {{s:.2f}} %")

for corpus, stem in (("BTP (in-domain, modèle final)", "btp"), ("Test métallurgie (modèle final)", "test")):
    geom_csv = RESULTS / "metrics" / f"metrics_geometry_{{stem}}.csv"
    if not geom_csv.is_file():
        geom_csv = RESULTS / "metrics" / "metrics_geometry.csv" if stem == "btp" else None
    if geom_csv is not None and geom_csv.is_file():
        geom = pd.read_csv(geom_csv)
        print(f"\\n=== Géométrie {{corpus}} ===")
        display(geom)
        for col in ("delta_macro_pct", "eta2_macro_balanced", "rankme_global", "c1_global", "c10_global"):
            if col in geom.columns and geom[col].notna().any():
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(geom["method"].astype(str), geom[col].astype(float))
                ax.set_title(f"{{DISPLAY_NAME}} — {{col}} ({{corpus}})")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.show()
    else:
        print(f"Pas de metrics_geometry_{{stem}}.csv pour {{corpus}}.")

# Embeddings exportés (recherche récursive)
def _has_dim_cols(path: Path) -> bool:
    cols = pd.read_csv(path, nrows=0).columns
    return any(str(c).startswith("dim_") for c in cols)

final_emb = RESULTS / "embeddings" / "final_embeddings.csv"
emb_csvs = [final_emb] if final_emb.is_file() else [p for p in RESULTS.rglob("*.csv") if _has_dim_cols(p)]
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
