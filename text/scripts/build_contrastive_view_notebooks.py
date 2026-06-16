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
CONFIG_PATH = ROOT / "configs/methods" / "{method_key}.yaml"

cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
_data = cfg.get("data") or {{}}
_model = cfg.get("model") or {{}}
DATASET_CSV = ROOT / _data.get("dataset_path", cfg.get("dataset_path", "dataset/data_btp.csv"))
OUTPUT_DIR = cfg.get("output_dir", f"output/{{METHOD_KEY}}")
RESULTS = ROOT / OUTPUT_DIR
LABEL_COL = _data.get("label_col", cfg.get("label_col", "pred_label"))
BACKBONE = _model.get("backbone_name", cfg.get("backbone_name", "(non défini)"))

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
    ycol = "val_eta2_macro_balanced_perc" if "val_eta2_macro_balanced_perc" in tl.columns else "val_loss"
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
        ax.set_xlabel("η² macro balanced (%) = selection_score")
        ax.set_title(f"{{DISPLAY_NAME}} — grille tuning")
        plt.tight_layout()
        plt.show()

resolved_cfg = RESULTS / "configs" / "config_resolved.yaml"
if resolved_cfg.is_file():
    print("\\n=== config_resolved.yaml ===")
    display(yaml.safe_load(resolved_cfg.read_text(encoding="utf-8")))

TEST_CORPUS = "metallurgie"  # configs/test_corpora.yaml
from safer_core.test_corpus import method_test_results_dir, resolve_test_corpus

_test_spec = resolve_test_corpus(TEST_CORPUS, anchor=ROOT)
_test_results = method_test_results_dir(METHOD_KEY, TEST_CORPUS, anchor=ROOT)
DATA_TEST = _test_spec.data_csv
print(f"Corpus test (hors domaine) : {{_test_spec.display_name}} ({{TEST_CORPUS}})")
print("  →", DATA_TEST)
print("  Métriques test :", _test_results)

print(
    "\\nNote : le corpus test utilise le modèle entraîné sur 100 % BTP "
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
    if "mean_eta2_macro_balanced_perc" in kval.columns:
        m = float(kval["mean_eta2_macro_balanced_perc"].iloc[0])
        s = float(kval.get("std_eta2_macro_balanced_perc", pd.Series([0])).iloc[0])
        print(f"η² macro balanced (%) val : {{m:.2f}} ± {{s:.2f}}")

for corpus, stem, base in (
    ("BTP (in-domain, modèle final)", "btp", RESULTS),
    (f"Test {{_test_spec.display_name}} (modèle final)", "test", _test_results),
):
    geom_csv = base / "metrics" / f"metrics_geometry_{{stem}}.csv"
    if not geom_csv.is_file():
        geom_csv = base / "metrics" / "metrics_geometry.csv" if stem == "btp" else None
    if geom_csv is not None and geom_csv.is_file():
        geom = pd.read_csv(geom_csv)
        print(f"\\n=== Géométrie {{corpus}} ===")
        display(geom)
        for col in (
            "eta2_macro_balanced_perc",
            "eta2_macro_balanced",
            "eta2_weighted",
        ):
            if col in geom.columns and geom[col].notna().any():
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(geom["method"].astype(str), geom[col].astype(float))
                ax.set_title(f"{{DISPLAY_NAME}} — {{col}} ({{corpus}})")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.show()
    else:
        print(f"Pas de metrics_geometry_{{stem}}.csv pour {{corpus}}.")

# Lecture seule — pas d'entraînement dans ce notebook.
'''

PCA_BTP_MD = """### PCA / t-SNE — BTP (embeddings fine-tunés)

Carte 2D sur `output/<method>/embeddings/final_embeddings_btp.csv` (repli `final_embeddings.csv`), couleur = macro.
"""

PCA_BTP_CODE = '''
from scgm_text.notebook_viz import plot_embeddings_csv_pca_tsne
from safer_core.test_corpus import method_btp_results_dir, resolve_contrastive_embeddings_csv

FIGURES_DIR = method_btp_results_dir(METHOD_KEY, anchor=ROOT) / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name: str) -> Path:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path

emb_btp = resolve_contrastive_embeddings_csv(METHOD_KEY, "btp", anchor=ROOT)

p_btp = plot_embeddings_csv_pca_tsne(
    emb_btp,
    DATASET_CSV,
    LABEL_COL,
    corpus_name=f"{DISPLAY_NAME} — BTP",
    save_fig=save_fig,
    png_name=f"{METHOD_KEY}_btp_pca_tsne.png",
    max_points=8000,
    seed=42,
)
if p_btp is None:
    print(f"(absent) embeddings BTP — relancer : python scripts/train_{METHOD_KEY}.py")
else:
    print(p_btp)
'''

TSNE_PER_MACRO_BTP_MD = """### t-SNE par macro — BTP

Grille 2×2 (A0, A1, B, C) : t-SNE recalculé **séparément** sur chaque macro pour visualiser la structure intra-rôle.
"""

TSNE_PER_MACRO_BTP_CODE = '''
from scgm_text.notebook_viz import plot_embeddings_csv_tsne_per_macro

p_btp_pm = plot_embeddings_csv_tsne_per_macro(
    emb_btp,
    DATASET_CSV,
    LABEL_COL,
    corpus_name=f"{DISPLAY_NAME} — BTP",
    save_fig=save_fig,
    png_name=f"{METHOD_KEY}_btp_tsne_per_macro.png",
    max_points=8000,
    seed=42,
)
if p_btp_pm is None:
    print("(absent) t-SNE par macro BTP — mêmes prérequis que PCA/t-SNE global")
else:
    print(p_btp_pm)
'''

PCA_TEST_MD = """### PCA / t-SNE — Test métallurgie

Carte 2D sur `output_test/<corpus>/<method>/embeddings/final_embeddings_test.csv` (pas sous `output/<method>/`).
"""

PCA_TEST_CODE = '''
from scgm_text.notebook_viz import plot_embeddings_csv_pca_tsne
from safer_core.test_corpus import resolve_contrastive_embeddings_csv

FIGURES_TEST = _test_results / "figures"
FIGURES_TEST.mkdir(parents=True, exist_ok=True)

def save_fig_test(name: str) -> Path:
    path = FIGURES_TEST / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path

emb_test = resolve_contrastive_embeddings_csv(
    METHOD_KEY, "test", corpus_id=TEST_CORPUS, anchor=ROOT
)
p_test = plot_embeddings_csv_pca_tsne(
    emb_test,
    DATA_TEST,
    LABEL_COL,
    corpus_name=f"{DISPLAY_NAME} — test métallurgie",
    save_fig=save_fig_test,
    png_name=f"{METHOD_KEY}_test_pca_tsne.png",
    max_points=8000,
    seed=42,
)
if p_test is None:
    print(f"(absent) {emb_test}")
    print("Relancer l'éval test : fit final 100 % BTP puis evaluate_btp_and_test")
    print(f"  Dossier test : {_test_results}")
else:
    print(p_test)
'''

TSNE_PER_MACRO_TEST_MD = """### t-SNE par macro — Test métallurgie

Grille 2×2 par étape (A0, A1, B, C) sur les embeddings fine-tunés du corpus test.
"""

TSNE_PER_MACRO_TEST_CODE = '''
from scgm_text.notebook_viz import plot_embeddings_csv_tsne_per_macro

p_test_pm = plot_embeddings_csv_tsne_per_macro(
    emb_test,
    DATA_TEST,
    LABEL_COL,
    corpus_name=f"{DISPLAY_NAME} — test métallurgie",
    save_fig=save_fig_test,
    png_name=f"{METHOD_KEY}_test_tsne_per_macro.png",
    max_points=8000,
    seed=42,
)
if p_test_pm is None:
    print("(absent) t-SNE par macro test — relancer éval test")
else:
    print(p_test_pm)
'''

SOFTTRIPLE_CENTERS_MD = """### Centres SoftTriple effectifs

Tableau des centres (`centers/softtriple_effective_centers.csv`) et projection PCA/t-SNE sur les embeddings BTP et test (losanges `A0#0`, …).

Prérequis : `export_effective_centers: true` et `center_regularization_type` ≠ `none` dans la config (défaut : `diversity`).
"""

SOFTTRIPLE_CENTERS_BTP_CODE = '''
if METHOD_KEY == "softtriple":
    from scgm_text.notebook_viz import (
        plot_embeddings_csv_pca_tsne_with_softtriple_centers,
        resolve_softtriple_centers_csv,
        softtriple_centers_summary_table,
    )

    centers_csv = resolve_softtriple_centers_csv(RESULTS)
    if centers_csv is None:
        print("(absent) centres SoftTriple — relancer train_softtriple.py avec export_effective_centers")
    else:
        print("Centres :", centers_csv.relative_to(ROOT))
        display(softtriple_centers_summary_table(centers_csv))
        p_centers = plot_embeddings_csv_pca_tsne_with_softtriple_centers(
            emb_btp,
            DATASET_CSV,
            LABEL_COL,
            results_dir=RESULTS,
            corpus_name=f"{DISPLAY_NAME} — BTP (centres)",
            save_fig=save_fig,
            png_name="softtriple_btp_pca_tsne_centers.png",
            max_points=8000,
            seed=42,
        )
        if p_centers is not None:
            print(p_centers)
'''

SOFTTRIPLE_CENTERS_TEST_CODE = '''
if METHOD_KEY == "softtriple" and emb_test is not None:
    from scgm_text.notebook_viz import plot_embeddings_csv_pca_tsne_with_softtriple_centers

    p_centers_test = plot_embeddings_csv_pca_tsne_with_softtriple_centers(
        emb_test,
        DATA_TEST,
        LABEL_COL,
        results_dir=RESULTS,
        corpus_name=f"{DISPLAY_NAME} — test métallurgie (centres)",
        save_fig=save_fig_test,
        png_name="softtriple_test_pca_tsne_centers.png",
        max_points=8000,
        seed=42,
    )
    if p_centers_test is not None:
        print(p_centers_test)
'''

RAW_TEST_MD = """### Embedding brut — test métallurgie (PCA / t-SNE / UMAP)

Vecteurs **encodeur Qwen** (`embeddings/test/…`), couleur = **`pred_label`** (étape chaîne accidentelle). Référence avant/après fine-tuning {display_name} — indépendant de `output/<method>/`.
"""

RAW_TEST_CODE = '''
from macro_transfer.notebook_viz import plot_test_corpus_raw_embeddings

FIGURES_RAW = _test_results / "figures" / "raw_encoder"
FIGURES_RAW.mkdir(parents=True, exist_ok=True)

_raw_emb = plot_test_corpus_raw_embeddings(
    TEST_CORPUS,
    fig_dir=FIGURES_RAW,
    anchor=ROOT,
    label_col=LABEL_COL,
    max_points=12000,
    seed=42,
    prefix="raw_test_embedding",
    show=True,
    display_metrics=True,
)
if _raw_emb.missing:
    print("Embedding brut test — fichiers manquants :", ", ".join(_raw_emb.missing))
    print("  → embeddings/test/Qwen3-Embedding-0.6B__<corpus>.csv")
    print("  → dataset/test/data_<corpus>.csv (colonne pred_label)")
else:
    print(
        "Figures embedding brut :",
        _raw_emb.pca_tsne_path,
        _raw_emb.tsne_per_macro_path,
        _raw_emb.umap_png_path,
    )
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
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
for _ in range(6):
    if (ROOT / "configs" / "methods").is_dir():
        break
    nested = ROOT / "text" / "configs" / "methods"
    if nested.is_dir():
        ROOT = (ROOT / "text").resolve()
        break
    if ROOT.parent == ROOT:
        break
    ROOT = ROOT.parent
if not (ROOT / "configs" / "methods").is_dir():
    raise FileNotFoundError(
        "Racine projet introuvable (configs/methods/). "
        "Lancez Jupyter depuis text/ ou SAFER/, ou placez le notebook sous text/notebooks/."
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""


def main() -> None:
    NB.mkdir(parents=True, exist_ok=True)
    for method_key, display_name, config_file in METHODS:
        nb_name = f"05_view_{method_key}_results.ipynb"
        md = (
            f"# Résultats — {display_name}\n\n"
            f"Lecture des sorties sous `output/{method_key}/` (chemins définis dans "
            f"`configs/methods/{method_key}.yaml`). **Pas d'entraînement** — le corpus "
            f"dépend du CSV configuré, pas du nom du notebook.\n"
        )
        view_code = VIEW_CODE_TEMPLATE.format(
            method_key=method_key,
            display_name=display_name,
        )
        cells = [
            _cell(md, "markdown"),
            _cell(_setup_cell()),
            _cell(view_code),
            _cell(PCA_BTP_MD, "markdown"),
            _cell(PCA_BTP_CODE),
            _cell(TSNE_PER_MACRO_BTP_MD, "markdown"),
            _cell(TSNE_PER_MACRO_BTP_CODE),
        ]
        if method_key == "softtriple":
            cells.extend(
                [
                    _cell(SOFTTRIPLE_CENTERS_MD, "markdown"),
                    _cell(SOFTTRIPLE_CENTERS_BTP_CODE),
                ]
            )
        cells.extend(
            [
                _cell(RAW_TEST_MD.format(display_name=display_name), "markdown"),
                _cell(RAW_TEST_CODE),
                _cell(PCA_TEST_MD, "markdown"),
                _cell(PCA_TEST_CODE),
                _cell(TSNE_PER_MACRO_TEST_MD, "markdown"),
                _cell(TSNE_PER_MACRO_TEST_CODE),
            ]
        )
        if method_key == "softtriple":
            cells.append(_cell(SOFTTRIPLE_CENTERS_TEST_CODE))
        out = NB / nb_name
        out.write_text(json.dumps(_nb(cells), indent=1), encoding="utf-8")
        print("Écrit", out)


if __name__ == "__main__":
    main()
