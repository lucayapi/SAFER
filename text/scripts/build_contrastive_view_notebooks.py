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
_test_corpora = _data.get("test_corpora") or cfg.get("test_corpora") or ["metallurgie", "caou"]

print("Méthode :", DISPLAY_NAME)
print("Config   :", CONFIG_PATH.relative_to(ROOT))
print("Dataset  :", DATASET_CSV)
print("Sorties  :", RESULTS)
print("Label    :", LABEL_COL)
print("Backbone :", BACKBONE)
print("Corpus OOD :", _test_corpora)

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
    ycol = next(
        (c for c in ("val_balanced_accuracy", "val_balanced_acc", "val_eta2_macro_balanced_perc", "val_loss") if c in tl.columns),
        None,
    )
    if ycol is not None:
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
        ax.set_xlabel("balanced_accuracy (selection_score)")
        ax.set_title(f"{{DISPLAY_NAME}} — grille tuning")
        plt.tight_layout()
        plt.show()

resolved_cfg = RESULTS / "configs" / "config_resolved.yaml"
if resolved_cfg.is_file():
    print("\\n=== config_resolved.yaml ===")
    display(yaml.safe_load(resolved_cfg.read_text(encoding="utf-8")))

from safer_core.test_corpus import resolve_test_corpus

print(
    "\\nNote : les corpus OOD utilisent le modèle entraîné sur 100 % BTP "
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
    for mean_key, std_key, label in (
        ("mean_val_balanced_accuracy", "std_val_balanced_accuracy", "balanced accuracy val"),
        ("mean_eta2_macro_balanced_perc", "std_eta2_macro_balanced_perc", "η² macro balanced (%) val"),
    ):
        if mean_key in kval.columns:
            m = float(kval[mean_key].iloc[0])
            s = float(kval.get(std_key, pd.Series([0])).iloc[0])
            print(f"{{label}} : {{m:.4f}} ± {{s:.4f}}")
            break

# --- Métriques classification ---
_cls_cols = ("balanced_accuracy", "macro_f1", "accuracy", "ba_ood_avg", "ba_ood_worst")

btp_cls = RESULTS / "metrics" / "metrics_classification_btp.csv"
if btp_cls.is_file():
    print("\\n=== Classification BTP (in-domain) ===")
    btp_df = pd.read_csv(btp_cls)
    display(btp_df)
    for col in _cls_cols:
        if col in btp_df.columns and btp_df[col].notna().any():
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(btp_df["method"].astype(str), btp_df[col].astype(float))
            ax.set_title(f"{{DISPLAY_NAME}} — {{col}} (BTP)")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
else:
    print(f"Pas de {{btp_cls.name}} — relancer eval_corpus.")

cross_csv = RESULTS / "metrics" / "cross_domain_generalization.csv"
if cross_csv.is_file():
    print("\\n=== Généralisation cross-domain ===")
    cross_df = pd.read_csv(cross_csv)
    display(cross_df)
    for col in _cls_cols:
        if col in cross_df.columns and cross_df[col].notna().any():
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(cross_df["method"].astype(str), cross_df[col].astype(float))
            ax.set_title(f"{{DISPLAY_NAME}} — {{col}} (cross-domain)")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()

all_test = RESULTS / "metrics" / "all_test_corpora_metrics.csv"
if all_test.is_file():
    print("\\n=== Tous les corpus test (OOD) ===")
    display(pd.read_csv(all_test))

for corpus_id in _test_corpora:
    spec = resolve_test_corpus(corpus_id, anchor=ROOT)
    cls_csv = RESULTS / "metrics" / f"metrics_classification_test_{{corpus_id}}.csv"
    if cls_csv.is_file():
        print(f"\\n=== Classification test — {{spec.display_name}} ({{corpus_id}}) ===")
        cdf = pd.read_csv(cls_csv)
        display(cdf)
        for col in _cls_cols:
            if col in cdf.columns and cdf[col].notna().any():
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(cdf["method"].astype(str), cdf[col].astype(float))
                ax.set_title(f"{{DISPLAY_NAME}} — {{col}} ({{corpus_id}})")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.show()
    else:
        print(f"Pas de metrics_classification_test_{{corpus_id}}.csv")

# Lecture seule — pas d'entraînement dans ce notebook.
'''

PCA_BTP_MD = """### PCA / t-SNE — BTP (embeddings projetés avant classification LR)

Carte 2D sur `output/<method>/embeddings/projected_btp.npy` + `projected_btp_metadata.csv`, couleur = macro (`pred_label`).
"""

PCA_BTP_CODE = '''
import numpy as np

from scgm_text.notebook_viz import (
    plot_corpus_projections,
    plot_projected_embeddings_pca_tsne,
    plot_tsne_per_macro_grid,
    sample_projection_indices,
)
from safer_core.test_corpus import method_btp_results_dir, resolve_contrastive_embeddings_csv, resolve_projected_embeddings_paths

FIGURES_DIR = method_btp_results_dir(METHOD_KEY, anchor=ROOT) / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name: str) -> Path:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path

proj_btp = resolve_projected_embeddings_paths(METHOD_KEY, "btp", anchor=ROOT)
projected_btp = meta_btp = None
if proj_btp is not None:
    projected_btp = np.load(proj_btp[0])
    meta_btp = pd.read_csv(proj_btp[1])
    paths = plot_projected_embeddings_pca_tsne(
        proj_btp[0],
        proj_btp[1],
        LABEL_COL,
        corpus_name=f"{DISPLAY_NAME} — BTP (avant classif. LR)",
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        png_name=f"{METHOD_KEY}_btp_pca_tsne.png",
        max_points=8000,
        seed=42,
    )
    if paths:
        print(paths)
else:
    print(f"(absent) projected_btp — relancer : python scripts/train_{METHOD_KEY}.py")

emb_btp = resolve_contrastive_embeddings_csv(METHOD_KEY, "btp", anchor=ROOT)
'''

TSNE_PER_MACRO_BTP_MD = """### t-SNE par macro — BTP

Grille 2×2 (A0, A1, B, C) : t-SNE recalculé **séparément** sur chaque macro pour visualiser la structure intra-rôle.
"""

TSNE_PER_MACRO_BTP_CODE = '''
if projected_btp is None or meta_btp is None:
    print("(absent) t-SNE par macro BTP — mêmes prérequis que PCA/t-SNE global")
else:
    idx = sample_projection_indices(meta_btp, LABEL_COL, max_points=8000, seed=42)
    p_btp_pm = plot_tsne_per_macro_grid(
        projected_btp[idx],
        meta_btp.loc[idx, LABEL_COL].astype(str).to_numpy(),
        corpus_name=f"{DISPLAY_NAME} — BTP (avant classif. LR)",
        save_fig=save_fig,
        png_name=f"{METHOD_KEY}_btp_tsne_per_macro.png",
        seed=42,
    )
    if p_btp_pm is not None:
        print(p_btp_pm)
'''

PCA_TEST_MD = """### PCA / t-SNE — Corpus test OOD (embeddings projetés avant classification LR)

Cartes 2D sur `output/<method>/embeddings/projected_<corpus>.npy` pour chaque corpus configuré (`test_corpora`).
"""

PCA_TEST_CODE = '''
for corpus_id in _test_corpora:
    spec = resolve_test_corpus(corpus_id, anchor=ROOT)
    pair = resolve_projected_embeddings_paths(METHOD_KEY, corpus_id, anchor=ROOT)
    if pair is None:
        print(f"(absent) projected_{corpus_id}.npy sous {RESULTS / 'embeddings'}")
        continue
    fig_dir = FIGURES_DIR / f"ood_{corpus_id}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig_ood(name: str, _dir=fig_dir) -> Path:
        path = _dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        return path

    paths = plot_projected_embeddings_pca_tsne(
        pair[0],
        pair[1],
        LABEL_COL,
        corpus_name=f"{DISPLAY_NAME} — {spec.display_name} (avant classif. LR)",
        save_fig=_save_fig_ood,
        figures_dir=fig_dir,
        png_name=f"{METHOD_KEY}_{corpus_id}_pca_tsne.png",
        max_points=8000,
        seed=42,
    )
    if paths:
        print(f"{corpus_id}:", paths)
'''

TSNE_PER_MACRO_TEST_MD = """### t-SNE par macro — Corpus test OOD

Grille 2×2 par étape (A0, A1, B, C) sur les embeddings projetés de chaque corpus test.
"""

TSNE_PER_MACRO_TEST_CODE = '''
import numpy as np

for corpus_id in _test_corpora:
    spec = resolve_test_corpus(corpus_id, anchor=ROOT)
    pair = resolve_projected_embeddings_paths(METHOD_KEY, corpus_id, anchor=ROOT)
    if pair is None:
        print(f"(absent) t-SNE par macro — projected_{corpus_id}")
        continue
    projected = np.load(pair[0])
    meta = pd.read_csv(pair[1])
    fig_dir = FIGURES_DIR / f"ood_{corpus_id}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig_ood(name: str, _dir=fig_dir) -> Path:
        path = _dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.show()
        return path

    idx = sample_projection_indices(meta, LABEL_COL, max_points=8000, seed=42)
    p_test_pm = plot_tsne_per_macro_grid(
        projected[idx],
        meta.loc[idx, LABEL_COL].astype(str).to_numpy(),
        corpus_name=f"{DISPLAY_NAME} — {spec.display_name}",
        save_fig=_save_fig_ood,
        png_name=f"{METHOD_KEY}_{corpus_id}_tsne_per_macro.png",
        seed=42,
    )
    if p_test_pm is not None:
        print(f"{corpus_id}:", p_test_pm)
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
if METHOD_KEY == "softtriple":
    from scgm_text.notebook_viz import plot_embeddings_csv_pca_tsne_with_softtriple_centers

    for corpus_id in _test_corpora:
        spec = resolve_test_corpus(corpus_id, anchor=ROOT)
        emb_test_csv = resolve_contrastive_embeddings_csv(
            METHOD_KEY, "test", corpus_id=corpus_id, anchor=ROOT
        )
        if not emb_test_csv.is_file():
            print(f"(absent) final_embeddings test {corpus_id}")
            continue
        fig_dir = FIGURES_DIR / f"ood_{corpus_id}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        def _save_fig_ood(name: str, _dir=fig_dir) -> Path:
            path = _dir / name
            plt.tight_layout()
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.show()
            return path

        p_centers_test = plot_embeddings_csv_pca_tsne_with_softtriple_centers(
            emb_test_csv,
            spec.data_csv,
            LABEL_COL,
            results_dir=RESULTS,
            corpus_name=f"{DISPLAY_NAME} — {spec.display_name} (centres)",
            save_fig=_save_fig_ood,
            png_name=f"softtriple_{corpus_id}_pca_tsne_centers.png",
            max_points=8000,
            seed=42,
        )
        if p_centers_test is not None:
            print(f"{corpus_id}:", p_centers_test)
'''

RAW_TEST_MD = """### Embedding brut — corpus test OOD (PCA / t-SNE / UMAP)

Vecteurs **encodeur Qwen** (`embeddings/Qwen3-Embedding-0.6B_<corpus>.csv`), couleur = **`pred_label`** (étape chaîne accidentelle). Référence avant fine-tuning {display_name}.
"""

RAW_TEST_CODE = '''
from macro_transfer.notebook_viz import plot_test_corpus_raw_embeddings

for corpus_id in _test_corpora:
    spec = resolve_test_corpus(corpus_id, anchor=ROOT)
    fig_dir = RESULTS / "figures" / f"raw_{corpus_id}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _raw_emb = plot_test_corpus_raw_embeddings(
        corpus_id,
        fig_dir=fig_dir,
        anchor=ROOT,
        label_col=LABEL_COL,
        max_points=12000,
        seed=42,
        prefix=f"raw_{corpus_id}",
        show=True,
        display_metrics=True,
    )
    if _raw_emb.missing:
        print(f"Embedding brut {corpus_id} — fichiers manquants :", ", ".join(_raw_emb.missing))
    else:
        print(
            f"{spec.display_name} — figures embedding brut :",
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
            f"`configs/methods/{method_key}.yaml`). **Pas d'entraînement** — métriques "
            f"de classification LR et t-SNE sur embeddings projetés (`projected_*.npy`) "
            f"avant la régression logistique.\n"
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
