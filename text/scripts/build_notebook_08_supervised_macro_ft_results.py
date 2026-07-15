"""Génère notebooks/08_view_supervised_macro_ft_results.ipynb."""

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


TITLE_MD = """# Résultats supervised_macro_ft

**Lecture seule** — visualisation des runs `supervised_macro_ft` (métriques, courbes, t-SNE, vraie vs prédite).

Modifiez `RESULTS_DIR` pour pointer vers une autre expérience (run standard, combo tuning, dossier custom).
"""

CONFIG_CODE = """# --- Paramètres (modifier ici) ---
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import HTML, display

ROOT = TEXT_ROOT  # défini par la cellule bootstrap

from safer_core.brand_style import apply_matplotlib_brand

SCATTER_POINT_SIZE = 4
PROJECTION_FIGSIZE = (10, 3.8)

apply_matplotlib_brand()
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = PROJECTION_FIGSIZE
plt.rcParams["figure.dpi"] = 96
plt.rcParams["figure.autolayout"] = True

# Dossier de résultats (relatif à ROOT ou chemin absolu)
RESULTS_DIR = ROOT / "output/supervised_macro_ft"
# Exemples :
# RESULTS_DIR = ROOT / "output/supervised_macro_ft/tuning/combos/projectionlinear_hiddim128_abc12345"
# RESULTS_DIR = Path("/chemin/vers/mon_experience")

RUN_INFERENCE = True   # False = saute inférence CE (sections lourdes)
SEED = 42
TSNE_SAMPLE_SIZE = 8000
RAW_TSNE_MAX_POINTS = 12000
LABEL_COL = "pred_label"
TEXT_COL = "sentence"
INFER_BATCH_SIZE = 32
INFER_DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

FIGURES_DIR = RESULTS_DIR / "figures_notebook"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = ROOT / "configs/methods/supervised_macro_ft.yaml"
if DEFAULT_CFG.is_file():
    _cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    _data = _cfg.get("data") or {}
    _model = _cfg.get("model") or {}
    TEST_CORPORA = list(_cfg.get("test_corpora") or ["metallurgie", "caou"])
    BACKBONE_EMB_CSV = _model.get("backbone_emb_csv")
    DATA_CSV = ROOT / _data.get("data_csv", "dataset/data_btp.csv")
else:
    TEST_CORPORA = ["metallurgie", "caou"]
    BACKBONE_EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
    DATA_CSV = ROOT / "dataset/data_btp.csv"

print("Résultats :", RESULTS_DIR)
print("Figures   :", FIGURES_DIR)
print("Inférence :", RUN_INFERENCE, "| device =", INFER_DEVICE)
"""

LOAD_ARTIFACTS_CODE = """from supervised_macro_ft.notebook_viz import (
    discover_projected_corpora,
    export_metrics_latex_table,
    load_macro_ft_artifacts,
    plot_supervised_macro_ft_train_history,
    style_metrics_table,
    validate_results_dir,
)

validate_results_dir(RESULTS_DIR)
ART = load_macro_ft_artifacts(RESULTS_DIR)

print("Checkpoint :", ART.checkpoint_dir)
print("Corpus projetés :", ART.projected_corpora or discover_projected_corpora(RESULTS_DIR))
if ART.train_summary:
    print("\\n=== train_summary.json ===")
    display(ART.train_summary)
"""

TABLES_MD = """## 1. Métriques de classification

Tableau unique : CV BTP (μ±σ), évaluation BTP et corpus OOD.
"""

TABLES_CODE = """from contrastive_methods.view_metrics import (
    build_macro_ft_classification_summary_table,
    format_ood_summary_line,
)

summary = build_macro_ft_classification_summary_table(
    ART.cv_summary,
    ART.metrics_by_corpus,
    test_corpora=TEST_CORPORA,
)
display(style_metrics_table(summary, ("balanced_accuracy", "macro_f1", "accuracy")))
ood_line = format_ood_summary_line(RESULTS_DIR)
if ood_line:
    print(ood_line)
if not summary.empty:
    print(export_metrics_latex_table(summary, ("balanced_accuracy", "macro_f1", "accuracy")))
"""

TRAIN_HISTORY_MD = """## 2. Courbes d'entraînement

Historique CV (`cv/train_history.csv`) + fit final 100 % BTP (`train_history_final.csv`).
"""

TRAIN_HISTORY_CODE = """if ART.train_history is not None and not ART.train_history.empty:
    display(ART.train_history.tail(12))
    plot_supervised_macro_ft_train_history(
        ART.train_history,
        fig_dir=FIGURES_DIR,
        filename="01_train_history_curves.png",
    )
else:
    print("(absent) train_history")
"""

PROJECTED_MD = """## 3. Embeddings projetés — PCA / t-SNE (vraie classe)

Couleur = macro vraie (`pred_label`). Figures exportées dans `figures_notebook/`.
"""

PROJECTED_CODE = """from scgm_text.notebook_viz import plot_projected_embeddings_pca_tsne

def save_fig(name: str) -> Path:
    fig = plt.gcf()
    fig.set_size_inches(*PROJECTION_FIGSIZE)
    path = FIGURES_DIR / name
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    return path

for stem in ART.projected_corpora or discover_projected_corpora(RESULTS_DIR):
    npy = RESULTS_DIR / "embeddings" / f"projected_{stem}.npy"
    meta_csv = RESULTS_DIR / "embeddings" / f"projected_{stem}_metadata.csv"
    if not npy.is_file() or not meta_csv.is_file():
        print(f"(absent) projected {stem}")
        continue
    print(f"\\n=== Projeté — {stem} ===")
    paths = plot_projected_embeddings_pca_tsne(
        npy,
        meta_csv,
        LABEL_COL,
        corpus_name=stem,
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        max_points=TSNE_SAMPLE_SIZE,
        seed=SEED,
        png_name=f"02_projected_{stem}_pca_tsne.png",
        point_size=SCATTER_POINT_SIZE,
        figsize=PROJECTION_FIGSIZE,
        include_plotly=False,
    )
    if paths:
        print(paths)
"""

INFERENCE_MD = """## 4. Vraie classe vs prédite (tête CE)

Charge d'abord `predictions/predictions_<corpus>.csv` produit par le job.
Sinon inférence via `checkpoints/best_model/` si `RUN_INFERENCE = True`.
"""

INFERENCE_CODE = """from safer_core.classification_eval import load_saved_predictions
from safer_core.test_corpus import resolve_test_corpus
from supervised_macro_ft.notebook_viz import (
    build_prediction_df,
    get_misclassification_sample,
    plot_calibration_histograms,
    plot_confusion_matrix_brand,
    plot_tsne_true_vs_pred_brand,
)
from scgm_text.notebook_viz import load_projected_embeddings_pair

corpora_to_run = [("btp", DATA_CSV)] + [
    (cid, resolve_test_corpus(cid, anchor=ROOT).data_csv) for cid in TEST_CORPORA
]
for corpus_id, data_path in corpora_to_run:
    pred_df = load_saved_predictions(RESULTS_DIR, corpus_id)
    if pred_df is not None:
        print(f"\\n=== Prédictions job — {corpus_id} (n={len(pred_df)}) ===")
    elif not RUN_INFERENCE:
        print(f"(skip) {corpus_id} — pas de predictions/ et RUN_INFERENCE=False")
        continue
    elif ART.checkpoint_dir is None:
        print(f"(absent) checkpoints/best_model — inférence impossible pour {corpus_id}")
        continue
    else:
        if not Path(data_path).is_file():
            print(f"(absent) {data_path}")
            continue
        meta_full = pd.read_csv(data_path)
        if TEXT_COL not in meta_full.columns or LABEL_COL not in meta_full.columns:
            print(f"(skip) colonnes manquantes pour {corpus_id}")
            continue
        texts = meta_full[TEXT_COL].astype(str).tolist()
        print(f"\\n=== Inférence CE — {corpus_id} (n={len(texts)}) ===")
        pred_df = build_prediction_df(
            ART.checkpoint_dir,
            meta_full,
            texts,
            label_col=LABEL_COL,
            text_col=TEXT_COL,
            device=INFER_DEVICE,
            batch_size=INFER_BATCH_SIZE,
            results_dir=RESULTS_DIR,
            corpus_id=corpus_id,
            anchor=ROOT,
            backbone_emb_csv=BACKBONE_EMB_CSV,
        )
    if "true_macro" not in pred_df.columns and LABEL_COL in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["true_macro"] = pred_df[LABEL_COL].astype(str)
    plot_confusion_matrix_brand(
        pred_df["true_macro"],
        pred_df["pred_macro"],
        title=f"Confusion — {corpus_id}",
        fig_dir=FIGURES_DIR,
        filename=f"03_confusion_{corpus_id}.png",
    )
    plot_calibration_histograms(
        pred_df,
        fig_dir=FIGURES_DIR,
        filename=f"03_calibration_{corpus_id}.png",
    )
    err_sample = get_misclassification_sample(pred_df, text_col=TEXT_COL, n=12)
    if not err_sample.empty:
        print("Erreurs (marge faible) :")
        display(err_sample)
    npy = RESULTS_DIR / "embeddings" / f"projected_{corpus_id}.npy"
    if npy.is_file():
        emb, _ = load_projected_embeddings_pair(
            npy,
            RESULTS_DIR / "embeddings" / f"projected_{corpus_id}_metadata.csv",
        )
        if len(emb) == len(pred_df):
            plot_tsne_true_vs_pred_brand(
                emb,
                pred_df,
                true_col="true_macro",
                pred_col="pred_macro",
                title=corpus_id,
                fig_dir=FIGURES_DIR,
                filename=f"03_tsne_true_vs_pred_{corpus_id}.png",
                max_points=TSNE_SAMPLE_SIZE,
                seed=SEED,
            )
"""

RAW_MD = """## 5. Embeddings bruts Qwen — comparaison avec vraies classes

Sources : `cache/backbone_hidden*.npy` ou CSV Qwen (`backbone_emb_csv` / registre test).
"""

RAW_CODE = """from supervised_macro_ft.notebook_viz import (
    load_raw_backbone_embeddings,
    plot_raw_embeddings_pca_tsne,
    plot_raw_vs_projected_tsne_pair,
)
from scgm_text.notebook_viz import load_projected_embeddings_pair

raw_corpora = ["btp", *TEST_CORPORA]
for corpus_id in raw_corpora:
    raw_h, raw_meta, missing = load_raw_backbone_embeddings(
        RESULTS_DIR,
        corpus_id,
        anchor=ROOT,
        backbone_emb_csv=BACKBONE_EMB_CSV if corpus_id == "btp" else None,
    )
    if raw_h is None or raw_meta is None:
        print(f"(absent) brut {corpus_id} :", missing)
        continue
    if LABEL_COL not in raw_meta.columns:
        print(f"(skip) {LABEL_COL} absent pour {corpus_id}")
        continue
    print(f"\\n=== Brut Qwen — {corpus_id} (n={len(raw_meta)}) ===")
    plot_raw_embeddings_pca_tsne(
        raw_h,
        raw_meta,
        LABEL_COL,
        corpus_name=corpus_id,
        fig_dir=FIGURES_DIR,
        filename=f"04_raw_{corpus_id}_pca_tsne.png",
        max_points=RAW_TSNE_MAX_POINTS,
        seed=SEED,
    )
    proj_npy = RESULTS_DIR / "embeddings" / f"projected_{corpus_id}.npy"
    proj_meta_csv = RESULTS_DIR / "embeddings" / f"projected_{corpus_id}_metadata.csv"
    if proj_npy.is_file() and proj_meta_csv.is_file():
        pair = load_projected_embeddings_pair(proj_npy, proj_meta_csv)
        if pair is not None:
            proj_emb, proj_meta = pair
            n_align = min(len(raw_h), len(proj_emb), len(raw_meta))
            plot_raw_vs_projected_tsne_pair(
                raw_h[:n_align],
                proj_emb[:n_align],
                raw_meta.iloc[:n_align],
                LABEL_COL,
                corpus_name=corpus_id,
                fig_dir=FIGURES_DIR,
                filename=f"04_raw_vs_projected_{corpus_id}.png",
                max_points=TSNE_SAMPLE_SIZE,
                seed=SEED,
            )
"""

OVERLAY_MD = """## 6. Overlay BTP + corpus test (embeddings projetés)

t-SNE joint : couleur = dataset, marqueur = macro.
"""

OVERLAY_CODE = """from macro_transfer.notebook_viz import plot_tsne_datasets_overlay
from scgm_text.notebook_viz import load_projected_embeddings_pair

emb_list = []
meta_list = []
labels = []
for stem in ["btp", *TEST_CORPORA]:
    npy = RESULTS_DIR / "embeddings" / f"projected_{stem}.npy"
    meta_csv = RESULTS_DIR / "embeddings" / f"projected_{stem}_metadata.csv"
    if not npy.is_file() or not meta_csv.is_file():
        continue
    e, m = load_projected_embeddings_pair(npy, meta_csv)
    emb_list.append(e)
    meta_list.append(m)
    labels.append(stem)

if len(emb_list) >= 2:
    plot_tsne_datasets_overlay(
        emb_list,
        meta_list,
        labels,
        LABEL_COL,
        fig_dir=FIGURES_DIR,
        filename="05_tsne_btp_test_overlay.png",
        max_points=TSNE_SAMPLE_SIZE,
        seed=SEED,
    )
else:
    print("(absent) au moins 2 corpus projetés pour overlay")
"""

TUNING_MD = """## 7. Tuning (si présent)

Grille hyperparamètres — `tuning/grid_summary.csv` ou parent tuning pour les combos.
"""

TUNING_CODE = """if ART.tuning_grid is not None and not ART.tuning_grid.empty:
    print("=== Grille tuning ===")
    display(ART.tuning_grid.sort_values("selection_score", ascending=False).head(20))
else:
    print("(absent) tuning/grid_summary.csv")
"""

BERTOPIC_MD = """## 8. Topic modeling BERTopic (intra-macro)

Segmentation par macro prédite (tête CE). Embeddings projetés. Sorties sous `bertopic_notebook/<corpus>/` pour le notebook **06** BN.
"""

BERTOPIC_CODE = """BERTOPIC_CORPUS = "metallurgie"
BERTOPIC_SEGMENT_MODE = "predicted"
RESTIMATE_BERTOPIC = False
BERTOPIC_UMAP_ENABLED = None
BERTOPIC_MIN_TOPIC_SIZE = None

from safer_core.classification_eval import load_saved_predictions
from macro_transfer.notebook_bertopic import (
    bertopic_run_dir,
    display_notebook_bertopic_results,
    load_notebook_bertopic_config,
    run_notebook_bertopic,
)

cfg_full = load_notebook_bertopic_config(anchor=ROOT)
nb_cfg = cfg_full.get("notebook") or {}
bertopic_out = bertopic_run_dir(RESULTS_DIR, BERTOPIC_CORPUS, output_subdir=str(nb_cfg.get("output_subdir", "bertopic_notebook")))

preds_bertopic = load_saved_predictions(RESULTS_DIR, BERTOPIC_CORPUS)
if preds_bertopic is None:
    _pred_cache = RESULTS_DIR / "transfer" / "target_macro_predictions.csv"
    if _pred_cache.is_file():
        _cached = pd.read_csv(_pred_cache)
        if "pred_macro" in _cached.columns:
            preds_bertopic = _cached
            print("Prédictions transfer/ :", _pred_cache)
else:
    print("Prédictions job :", RESULTS_DIR / "predictions" / f"predictions_{BERTOPIC_CORPUS}.csv")

if preds_bertopic is None and RUN_INFERENCE and ART.checkpoint_dir is not None:
    from supervised_macro_ft.notebook_viz import build_prediction_df
    from safer_core.test_corpus import resolve_test_corpus

    _data_path = (
        DATA_CSV
        if BERTOPIC_CORPUS == "btp"
        else resolve_test_corpus(BERTOPIC_CORPUS, anchor=ROOT).data_csv
    )
    _meta_full = pd.read_csv(_data_path)
    preds_bertopic = build_prediction_df(
        ART.checkpoint_dir,
        _meta_full,
        _meta_full[TEXT_COL].astype(str).tolist(),
        label_col=LABEL_COL,
        text_col=TEXT_COL,
        device=INFER_DEVICE,
        batch_size=INFER_BATCH_SIZE,
        results_dir=RESULTS_DIR,
        corpus_id=BERTOPIC_CORPUS,
        anchor=ROOT,
        backbone_emb_csv=BACKBONE_EMB_CSV,
    )

_assign = bertopic_out / "topics_bertopic" / "assignments.csv"
if RESTIMATE_BERTOPIC or not _assign.is_file():
    bt_cfg = dict(cfg_full)
    if BERTOPIC_UMAP_ENABLED is not None:
        bt_cfg.setdefault("bertopic", {}).setdefault("umap", {})["enabled"] = bool(BERTOPIC_UMAP_ENABLED)
    if BERTOPIC_MIN_TOPIC_SIZE is not None:
        bt_cfg.setdefault("bertopic", {})["min_topic_size"] = int(BERTOPIC_MIN_TOPIC_SIZE)
    run_notebook_bertopic(
        RESULTS_DIR,
        BERTOPIC_CORPUS,
        method_name="supervised_macro_ft",
        view_kind="macro_ft",
        segment_mode=BERTOPIC_SEGMENT_MODE or str(nb_cfg.get("segment_mode", "predicted")),
        label_col=LABEL_COL,
        text_col=TEXT_COL,
        preds=preds_bertopic,
        bertopic_cfg=bt_cfg.get("bertopic"),
        topics_export_cfg=bt_cfg.get("topics_export"),
        topic_judge_cfg=bt_cfg.get("topic_judge"),
        anchor=ROOT,
        method_key="supervised_macro_ft",
        seed=SEED,
        export_for_bn=bool(nb_cfg.get("export_for_bn", True)),
    )
    print("BERTopic terminé :", bertopic_out)
else:
    print("Cache BERTopic :", bertopic_out)

display_notebook_bertopic_results(bertopic_out)
print("→ Entrée notebook 06 BN :", bertopic_out)
"""

FOOTER_MD = """---

**Figures exportées** : `{RESULTS_DIR}/figures_notebook/`

Entraînement : `bash jobs/train_supervised_macro_ft.sh` — baseline sklearn Qwen brut : notebook **07**.
"""


def build_notebook() -> dict:
    cells = [
        md(TITLE_MD),
        py(NOTEBOOK_PATH_SETUP, cell_id="bootstrap"),
        py(CONFIG_CODE, cell_id="config"),
        md(TABLES_MD),
        py(LOAD_ARTIFACTS_CODE, cell_id="load_artifacts"),
        py(TABLES_CODE, cell_id="tables"),
        md(TRAIN_HISTORY_MD),
        py(TRAIN_HISTORY_CODE, cell_id="train_history"),
        md(PROJECTED_MD),
        py(PROJECTED_CODE, cell_id="projected"),
        md(INFERENCE_MD),
        py(INFERENCE_CODE, cell_id="inference"),
        md(RAW_MD),
        py(RAW_CODE, cell_id="raw"),
        md(OVERLAY_MD),
        py(OVERLAY_CODE, cell_id="overlay"),
        md(TUNING_MD),
        py(TUNING_CODE, cell_id="tuning"),
        md(BERTOPIC_MD),
        py(BERTOPIC_CODE, cell_id="bertopic"),
        md(FOOTER_MD),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Écrit : {NB_PATH}")


if __name__ == "__main__":
    main()
