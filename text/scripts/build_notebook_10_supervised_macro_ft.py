"""Génère notebooks/10_supervised_macro_ft_results.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "10_supervised_macro_ft_results.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP
from macro_transfer.notebook_viz import (
    notebook_topic_judge_section_md,
    notebook_topic_judge_source,
)


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
# 10 — Fine-tuning supervisé macro (tête softmax)

Encodeur **Qwen fine-tuné** (θ) + tête lineaire CE : \(h=E_\theta(u)\), \(p(y|r)=\mathrm{softmax}(Wh+b)\).

**Prérequis** :
```bash
bash jobs/train_supervised_macro_ft.sh
CORPUS=metallurgie bash jobs/run_supervised_macro_ft_transfer.sh
python scripts/build_notebook_10_supervised_macro_ft.py
```

Distinction notebook **07** : sklearn sur embeddings Qwen **figés** (CSV). Ici θ est adapté ; BERTopic utilise **h_t fine-tuné**.
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

from macro_transfer.notebook_viz import (
    compute_fsp_confidence_calibration,
    get_fsp_top_confident_errors,
    load_supervised_macro_ft_run_artifacts,
    load_supervised_macro_ft_vs_baseline07_metrics,
    plot_fsp_confusion_heatmap,
    plot_fsp_distribution_histograms,
    plot_fsp_pred_macro_distribution,
    plot_tsne_true_vs_pred,
)
from supervised_macro_ft.transfer import supervised_macro_ft_output_dir

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
TRAIN_OUT = TEXT_ROOT / "output" / "supervised_macro_ft"
OUT_DIR = supervised_macro_ft_output_dir(TEST_CORPUS, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"10_supervised_macro_ft_{TEST_CORPUS}"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("TRAIN_OUT :", TRAIN_OUT)
print("OUT_DIR :", OUT_DIR)
sns.set_theme(style="whitegrid")
"""
        ),
        md("## § CV BTP (entraînement)"),
        py(
            r"""
cv_summary = TRAIN_OUT / "cv" / "cv_summary.csv"
cv_folds = TRAIN_OUT / "cv" / "cv_per_fold.csv"
if cv_summary.is_file():
    display(pd.read_csv(cv_summary))
if cv_folds.is_file():
    display(pd.read_csv(cv_folds).head())
else:
    print("CV absente — lancer jobs/train_supervised_macro_ft.sh")
"""
        ),
        md("## § Comparaison vs baseline 07 (sklearn Qwen brut)"),
        py(
            r"""
cmp = load_supervised_macro_ft_vs_baseline07_metrics(TEST_CORPUS, anchor=TEXT_ROOT)
display(cmp)
"""
        ),
        md("## § Métriques test + confusion"),
        py(
            r"""
ART = load_supervised_macro_ft_run_artifacts(OUT_DIR)
pred = ART.predictions.copy()
metrics = ART.metrics or {}
display(pd.DataFrame([metrics]))
print("run_bertopic :", metrics.get("run_bertopic"))
if ART.confusion is not None and not ART.confusion.empty:
    display(ART.confusion)
    plot_fsp_confusion_heatmap(ART.confusion, fig_dir=FIG_DIR)
"""
        ),
        md("## § Distribution et calibration"),
        py(
            r"""
plot_fsp_pred_macro_distribution(pred, fig_dir=FIG_DIR)
plot_fsp_distribution_histograms(pred, fig_dir=FIG_DIR)
cal = compute_fsp_confidence_calibration(pred)
if cal is not None and not cal.empty:
    display(cal)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cal["mean_confidence"], cal["accuracy"], "o-")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Calibration confiance (FT supervisé)")
    fig.savefig(FIG_DIR / "calibration.png", dpi=120, bbox_inches="tight")
    plt.show()
top_err = get_fsp_top_confident_errors(pred, top_k=20)
if not top_err.empty:
    display(top_err[[c for c in ["sentence", "true_macro", "pred_macro", "confidence"] if c in top_err.columns]])
"""
        ),
        md(
            r"""
## § t-SNE 2D — embeddings fine-tunés (vraie vs prédite)

Même projection t-SNE pour les deux panneaux ; couleur gauche = **vraie** étape, droite = **prédite**.

- **BTP** : `output/supervised_macro_ft/embeddings/btp_embeddings.npy` (export job train)
- **Test** : `embeddings/target_embeddings.npy` (export job transfert)
"""
        ),
        py(
            r"""
import numpy as np

# --- BTP (train) ---
btp_emb_path = TRAIN_OUT / "embeddings" / "btp_embeddings.npy"
btp_meta_path = TRAIN_OUT / "embeddings" / "btp_embeddings_metadata.csv"
if btp_emb_path.is_file() and btp_meta_path.is_file():
    h_btp = np.load(btp_emb_path)
    meta_btp = pd.read_csv(btp_meta_path)
    plot_tsne_true_vs_pred(
        h_btp,
        meta_btp,
        true_col="pred_label",
        pred_col="pred_macro",
        title="BTP (entraînement)",
        fig_dir=FIG_DIR,
        filename="tsne_btp_true_vs_pred.png",
        max_points=2500,
    )
else:
    print("Embeddings BTP absents — relancer jobs/train_supervised_macro_ft.sh")

# --- Test ---
test_emb_path = OUT_DIR / "embeddings" / "target_embeddings.npy"
if test_emb_path.is_file() and len(pred) == len(np.load(test_emb_path)):
    h_test = np.load(test_emb_path)
    true_col = "true_macro" if "true_macro" in pred.columns else "pred_label"
    if true_col in pred.columns and "pred_macro" in pred.columns:
        plot_tsne_true_vs_pred(
            h_test,
            pred,
            true_col=true_col,
            pred_col="pred_macro",
            title=f"Test {TEST_CORPUS}",
            fig_dir=FIG_DIR,
            filename="tsne_test_true_vs_pred.png",
            max_points=2500,
        )
    else:
        print("Colonnes true_macro / pred_macro absentes pour t-SNE test.")
else:
    print("Embeddings test absents — relancer jobs/run_supervised_macro_ft_transfer.sh")
"""
        ),
        md("## § BERTopic (embeddings h_t adaptés)"),
        py(
            r"""
if not metrics.get("run_bertopic", True):
    print("run_bertopic=false — BERTopic non exécuté.")
else:
    themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"
    if themes_path.is_file():
        display(pd.read_csv(themes_path).head(20))
    else:
        print("themes_by_macro.csv absent.")
    stats_path = OUT_DIR / "summary" / "macro_topic_stats.csv"
    if stats_path.is_file():
        display(pd.read_csv(stats_path))
"""
        ),
        md(notebook_topic_judge_section_md()),
        py(notebook_topic_judge_source("OUT_DIR", "FIG_DIR", restimate_var=None, topic_judge_cfg_var=None)),
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
