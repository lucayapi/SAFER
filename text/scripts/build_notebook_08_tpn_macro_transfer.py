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
# 08 — Résultats transfert macro TPN

Artefacts sous `output_test/<TEST_CORPUS>/macro_transfer/tpn_<encodeur>/`.

**Variables** : `TEST_CORPUS`, `BASE_METHOD` (`softtriple`, `supcon`, `batch_triplet`, `scgm_text`).

**Prérequis** : `bash jobs/run_tpn_macro_transfer.sh`
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

from macro_transfer.tpn_encode import tpn_method_name
from safer_core.test_corpus import macro_transfer_output_dir

TEST_CORPUS = os.environ.get("TEST_CORPUS", "metallurgie")
BASE_METHOD = os.environ.get("BASE_METHOD", "softtriple")
OUT_DIR = macro_transfer_output_dir(tpn_method_name(BASE_METHOD), TEST_CORPUS, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"08_tpn_{TEST_CORPUS}_{OUT_DIR.name}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from macro_transfer.constants import MACRO_NAMES
def _macro_prob_columns():
    return [f"p_{m}" for m in MACRO_NAMES]

def enrich_gating_meta(df: pd.DataFrame) -> pd.DataFrame:
    # margin / entropy dérivées des p_* si absentes (colonnes optionnelles).
    out = df.copy()
    prob_cols = [c for c in _macro_prob_columns() if c in out.columns]
    if not prob_cols:
        return out
    p = out[prob_cols].to_numpy(dtype=np.float64)
    rs = p.sum(axis=1, keepdims=True)
    rs = np.where(rs > 1e-12, rs, 1.0)
    p = p / rs
    sp = np.sort(p, axis=1)
    if "margin" not in out.columns:
        out["margin"] = sp[:, -1] - sp[:, -2]
    if "entropy" not in out.columns:
        out["entropy"] = -np.sum(p * np.log(p + 1e-12), axis=1)
    return out

print("Corpus test :", TEST_CORPUS)
print("BASE_METHOD :", BASE_METHOD)
print("OUT_DIR :", OUT_DIR)
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
elif manifest_path.is_file() and manifest.get("bertopic_summary"):
    print("bertopic_summary :", manifest.get("bertopic_summary"))
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
    if not m:
        continue
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

if len(metrics_df) >= 1:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    phases = metrics_df["phase"].tolist()
    w = 0.8 / max(len(phases), 1)
    for pi, phase in enumerate(phases):
        sub = metrics_df.loc[metrics_df["phase"] == phase]
        if sub.empty:
            continue
        off = (pi - (len(phases) - 1) / 2) * w
        for i, col in enumerate(["accuracy", "macro_f1", "balanced_accuracy"]):
            if col in sub.columns:
                ax.bar(i + off, float(sub[col].iloc[0]), w * 0.9, label=phase if i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels(["accuracy", "macro_f1", "balanced_accuracy"])
    ax.set_title(f"Métriques — {OUT_DIR.name} ({TEST_CORPUS})")
    ax.legend()
    fig.savefig(FIG_DIR / "metrics_compare.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("Aucune métrique transfer_metrics*.json trouvée.")
"""
        ),
        md("## § Distributions q_conf, margin, entropy"),
        py(
            r"""
p_init = OUT_DIR / "transfer" / "metadata_with_initial_macro_probs.csv"
p_adapt = OUT_DIR / "transfer" / "metadata_with_tpn_macro_probs.csv"
if not p_init.is_file() or not p_adapt.is_file():
    raise FileNotFoundError(
        f"Métadonnées TPN manquantes sous {OUT_DIR / 'transfer'}. "
        "Lancer : bash jobs/run_tpn_macro_transfer.sh"
    )
meta_init = enrich_gating_meta(pd.read_csv(p_init))
meta_adapt = enrich_gating_meta(pd.read_csv(p_adapt))
print("Colonnes gating init :", [c for c in ("q_conf", "margin", "entropy") if c in meta_init.columns])
print("Colonnes gating adapt :", [c for c in ("q_conf", "margin", "entropy") if c in meta_adapt.columns])

for col, title in [("q_conf", "Confiance"), ("margin", "Marge top1-top2"), ("entropy", "Entropie")]:
    if col not in meta_init.columns and col not in meta_adapt.columns:
        print(f"Skip {title} : colonne absente")
        continue
    ncols = 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.5))
    if ncols == 1:
        axes = [axes]
    if col in meta_init.columns:
        axes[0].hist(meta_init[col].dropna(), bins=40, color="steelblue", edgecolor="white")
        axes[0].set_title(f"{title} — initial")
    if col in meta_adapt.columns:
        axes[1].hist(meta_adapt[col].dropna(), bins=40, color="darkorange", edgecolor="white")
        axes[1].set_title(f"{title} — adapté")
    fig.suptitle(f"{title} ({TEST_CORPUS})")
    fig.tight_layout()
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
gating_stats_path = OUT_DIR / "transfer" / "gating_stats.json"
if gating_stats_path.is_file():
    with open(gating_stats_path, encoding="utf-8") as f:
        gs = json.load(f)
    print("gating_stats.json :", gs)

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
else:
    print("coverage_by_threshold.csv absent")
"""
        ),
        md("## § Diagnostics BERTopic / compression (si présents)"),
        py(
            r"""
for rel in (
    "macro_compression_diagnostics.csv",
    "bertopic_warnings.txt",
    "bertopic_grid_A0_A1.csv",
    "bertopic_grid_A0_A1_best.csv",
):
    p = OUT_DIR / rel
    if p.is_file():
        print(f"=== {rel} ===")
        if rel.endswith(".csv"):
            display(pd.read_csv(p).head(15))
        else:
            print(p.read_text(encoding="utf-8")[:2000])
comp_path = OUT_DIR / "macro_compression_diagnostics.csv"
if comp_path.is_file():
    comp = pd.read_csv(comp_path)
    if "compression_ratio" in comp.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(comp["macro"].astype(str), comp["compression_ratio"].astype(float))
        ax.axhline(1.0, color="gray", ls="--")
        ax.set_title("Compression ratio adapt/init")
        plt.show()
"""
        ),
        md("## § Embeddings 2D (PCA) — avant / après adaptation"),
        py(
            r"""
from sklearn.decomposition import PCA

emb_dir = OUT_DIR / "embeddings"
p_proj = emb_dir / "target_projected.npy"
p_adapt = emb_dir / "target_adapted.npy"
if not p_proj.is_file() or not p_adapt.is_file():
    raise FileNotFoundError(f"Embeddings TPN manquants sous {emb_dir}")

color_col = "m_hat" if "m_hat" in meta_adapt.columns else None
label_col = "pred_label" if "pred_label" in meta_adapt.columns else None

def pca_plot(z, meta, title, color_by):
    if color_by is None or color_by not in meta.columns:
        print("skip PCA:", title, "(colonne couleur absente)")
        return
    if len(z) != len(meta):
        print(f"skip PCA: {title} — z ({len(z)}) != meta ({len(meta)})")
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

z_proj = np.load(p_proj)
z_adapt = np.load(p_adapt)
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
        for c in ["loss_src", "loss_proto", "loss_kl", "loss_ent", "loss_div", "loss_pres"]:
            if c in log_df.columns:
                ax.plot(log_df["epoch"], log_df[c], alpha=0.7, label=c)
        ax.set_xlabel("epoch")
        ax.set_title("Losses entraînement adaptateur TPN")
        ax.legend(fontsize=8)
        fig.savefig(FIG_DIR / "training_losses.png", dpi=120, bbox_inches="tight")
        plt.show()
else:
    print("training/training_log.csv absent — relancer jobs/run_tpn_macro_transfer.sh")
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
