"""Génère notebooks/08_tpn_macro_transfer_results.ipynb (lecture seule TPN)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "08_tpn_macro_transfer_results.ipynb"

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


def main() -> None:
    cells = [
        md(
            r"""
# 08 — Transfert macro TPN (encodeur modulable + adaptateur)

Lecture des artefacts sous `output_test/<TEST_CORPUS>/macro_transfer/tpn_<encodeur>/`.

**Encodeur** : paramètre `BASE_METHOD` (`softtriple`, `supcon`, `batch_triplet`, `scgm_text`) → dossier `tpn_<BASE_METHOD>`.

**Prérequis** : `bash jobs/run_tpn_macro_transfer.sh` ou `BASE_METHOD=supcon CONFIG=configs/tpn_macro_transfer_supcon.yaml bash jobs/run_tpn_macro_transfer.sh`
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from safer_core.test_corpus import macro_transfer_output_dir

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
OUT_DIR = macro_transfer_output_dir("tpn_softtriple", TEST_CORPUS, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"08_tpn_{TEST_CORPUS}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Corpus test :", TEST_CORPUS)
print("Répertoire TPN :", OUT_DIR)
sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Manifeste et résumé"),
        py(
            r"""
manifest_path = OUT_DIR / "run_manifest.json"
summary_path = OUT_DIR / "summary" / "tpn_summary.json"
if manifest_path.is_file():
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    print("run_manifest.json :")
    for k in ("method", "base_encoder", "checkpoint", "n_source", "n_target", "skip_bertopic"):
        print(f"  {k}: {manifest.get(k)}")
if summary_path.is_file():
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    display(pd.DataFrame([summary.get("metrics_initial", {}), summary.get("metrics_adapted", {})],
                         index=["initial", "adapted"]))
"""
        ),
        md("## § Métriques initial vs adapté"),
        py(
            r"""
def load_metrics(prefix):
    p = OUT_DIR / "transfer" / f"transfer_metrics_{prefix}.json"
    return json.load(open(p, encoding="utf-8")) if p.is_file() else {}

m_init = load_metrics("initial")
m_adapt = load_metrics("adapted")
rows = []
for label, m in (("initial", m_init), ("adapted", m_adapt)):
    if m:
        rows.append({
            "phase": label,
            "n_eval": m.get("n_eval"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "mean_q_conf": m.get("mean_q_conf"),
            "mean_margin": m.get("mean_margin"),
            "mean_entropy": m.get("mean_entropy"),
        })
metrics_df = pd.DataFrame(rows)
display(metrics_df)

if len(metrics_df) == 2:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    w = 0.35
    for i, col in enumerate(["accuracy", "macro_f1", "balanced_accuracy"]):
        ax.bar(i - w/2, metrics_df.loc[metrics_df["phase"] == "initial", col].values[0], w, label="initial" if i == 0 else "")
        ax.bar(i + w/2, metrics_df.loc[metrics_df["phase"] == "adapted", col].values[0], w, label="adapted" if i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels(["accuracy", "macro_f1", "balanced_accuracy"])
    ax.set_title(f"Métriques TPN — {TEST_CORPUS}")
    ax.legend()
    fig.savefig(FIG_DIR / "metrics_compare.png", dpi=120, bbox_inches="tight")
    plt.show()
"""
        ),
        md("## § Distributions q_conf, margin, entropy"),
        py(
            r"""
meta_init = pd.read_csv(OUT_DIR / "transfer" / "metadata_with_initial_macro_probs.csv")
meta_adapt = pd.read_csv(OUT_DIR / "transfer" / "metadata_with_tpn_macro_probs.csv")

for col, title in [("q_conf", "Confiance"), ("margin", "Marge top1-top2"), ("entropy", "Entropie")]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    if col in meta_init.columns:
        axes[0].hist(meta_init[col].dropna(), bins=40, color="steelblue", edgecolor="white")
        axes[0].set_title(f"{title} — initial")
    if col in meta_adapt.columns:
        axes[1].hist(meta_adapt[col].dropna(), bins=40, color="darkorange", edgecolor="white")
        axes[1].set_title(f"{title} — adapté")
    fig.suptitle(f"{title} ({TEST_CORPUS})")
    fig.savefig(FIG_DIR / f"hist_{col}.png", dpi=120, bbox_inches="tight")
    plt.show()
"""
        ),
        md("## § Matrices de confusion (si labels disponibles)"),
        py(
            r"""
def plot_confusion(meta, title):
    if "pred_label" not in meta.columns:
        print("pred_label absent — skip", title)
        return
    from macro_transfer.constants import MACRO_NAMES, VALID_LABELS
    sub = meta[meta["pred_label"].isin(VALID_LABELS) & meta["m_hat"].isin(MACRO_NAMES)]
    if sub.empty:
        return
    cm = pd.crosstab(sub["pred_label"], sub["m_hat"]).reindex(index=MACRO_NAMES, columns=MACRO_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    plt.show()

plot_confusion(meta_init, "Confusion — initial")
plot_confusion(meta_adapt, "Confusion — adapté")
"""
        ),
        md("## § Distances entre prototypes"),
        py(
            r"""
for phase in ("initial", "final"):
    p = OUT_DIR / "prototypes" / f"prototype_distances_{phase}.csv"
    if p.is_file():
        print(f"=== Prototypes {phase} ===")
        display(pd.read_csv(p))
"""
        ),
        md("## § Couverture / performance par seuil"),
        py(
            r"""
cov_path = OUT_DIR / "transfer" / "coverage_by_threshold.csv"
if cov_path.is_file():
    cov = pd.read_csv(cov_path)
    display(cov.head(20))
    for thr_type in cov["threshold_type"].unique():
        sub = cov[cov["threshold_type"] == thr_type]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["threshold"], sub["coverage"], "o-", label="coverage")
        ax2 = ax.twinx()
        ax2.plot(sub["threshold"], sub["macro_f1"], "s--", color="crimson", label="macro_f1")
        ax.set_title(f"Couverture & F1 — {thr_type} ({TEST_CORPUS})")
        ax.set_xlabel("seuil")
        fig.savefig(FIG_DIR / f"coverage_{thr_type}.png", dpi=120, bbox_inches="tight")
        plt.show()
"""
        ),
        md("## § Embeddings 2D (PCA) — avant / après adaptation"),
        py(
            r"""
from sklearn.decomposition import PCA

z_proj = np.load(OUT_DIR / "embeddings" / "target_projected.npy")
z_adapt = np.load(OUT_DIR / "embeddings" / "target_adapted.npy")
color_col = "m_hat" if "m_hat" in meta_adapt.columns else None
label_col = "pred_label" if "pred_label" in meta_adapt.columns else None

def pca_plot(z, meta, title, color_by):
    if color_by is None or color_by not in meta.columns:
        return
    n = min(len(z), len(meta), 8000)
    idx = np.random.default_rng(42).choice(len(z), size=n, replace=False) if len(z) > n else np.arange(len(z))
    xy = PCA(n_components=2, random_state=42).fit_transform(z[idx])
    labels = meta.iloc[idx][color_by].astype(str)
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in sorted(labels.unique()):
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, label=lab)
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)
    plt.show()

pca_plot(z_proj, meta_init, f"PCA projected — {TEST_CORPUS}", color_col or label_col)
pca_plot(z_adapt, meta_adapt, f"PCA adapted — {TEST_CORPUS}", color_col)
if label_col and label_col != color_col:
    pca_plot(z_adapt, meta_adapt, f"PCA adapted (pred_label) — {TEST_CORPUS}", label_col)
"""
        ),
        md("## § Topics BERTopic"),
        py(
            r"""
themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"
summary_path = OUT_DIR / "summary" / "topics_summary.csv"
if summary_path.is_file():
    display(pd.read_csv(summary_path))
if themes_path.is_file():
    th = pd.read_csv(themes_path)
    cols = ["macro", "topic_id", "n_units", "top_words", "theme_label", "top_sentences"]
    display(th[[c for c in cols if c in th.columns]])
else:
    print("themes_by_macro.csv absent (BERTopic non exécuté ou échec)")
"""
        ),
        md("## § Courbe d'entraînement"),
        py(
            r"""
log_path = OUT_DIR / "training" / "training_log.csv"
if log_path.is_file():
    log_df = pd.read_csv(log_path)
    display(log_df.tail(10))
    if "loss_total" in log_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log_df["epoch"], log_df["loss_total"], label="loss_total")
        for c in ["loss_src", "loss_proto", "loss_kl", "loss_pres"]:
            if c in log_df.columns:
                ax.plot(log_df["epoch"], log_df[c], alpha=0.7, label=c)
        ax.set_xlabel("epoch")
        ax.set_title("Losses TPN")
        ax.legend()
        fig.savefig(FIG_DIR / "training_losses.png", dpi=120, bbox_inches="tight")
        plt.show()
"""
        ),
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
