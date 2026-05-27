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
BASE_METHOD = os.environ.get("BASE_METHOD", "scgm_text")
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
from macro_transfer.report_tables import (
    encoder_display_name,
    format_transfer_metrics_table,
    load_transfer_metrics_pair,
)

def load_metrics(prefix):
    p = OUT_DIR / "transfer" / f"transfer_metrics_{prefix}.json"
    return json.load(open(p, encoding="utf-8")) if p.is_file() else {}

m_init, m_adapt = load_transfer_metrics_pair(OUT_DIR)
cmp_csv = OUT_DIR / "summary" / "transfer_metrics_comparison.csv"
if cmp_csv.is_file():
    print("=== Tableau métriques (CSV pipeline) ===")
    display(pd.read_csv(cmp_csv))
else:
    metrics_table = format_transfer_metrics_table(m_init, m_adapt, BASE_METHOD)
    print(f"=== Tableau Modèle — {encoder_display_name(BASE_METHOD)} ({TEST_CORPUS}) ===")
    display(metrics_table)

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
if len(metrics_df):
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
from macro_transfer.notebook_viz import plot_confusion_from_metrics

enc_slug = BASE_METHOD.replace("_", "")
if m_init:
    plot_confusion_from_metrics(
        m_init,
        f"Confusion — {encoder_display_name(BASE_METHOD)} initial",
        FIG_DIR / f"confusion_{enc_slug}_initial_meta.png",
    )
if m_adapt:
    plot_confusion_from_metrics(
        m_adapt,
        f"Confusion — {encoder_display_name(BASE_METHOD)} adapté",
        FIG_DIR / f"confusion_{enc_slug}_adapted_meta.png",
    )
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
        md("## § t-SNE domaines source (BTP) vs test — initial vs adapté"),
        py(
            r"""
from macro_transfer.notebook_viz import plot_domain_tsne_side_by_side

plot_domain_tsne_side_by_side(
    OUT_DIR,
    FIG_DIR,
    max_points=4000,
    seed=42,
    source_label="Source BTP",
    target_label=f"Test {TEST_CORPUS}",
    test_corpus_name=TEST_CORPUS,
)
"""
        ),
        md("## § Topics BERTopic"),
        py(
            r"""
from macro_transfer.report_tables import load_macro_topic_stats
from macro_transfer.topics_export import format_macro_topic_stats_display

topic_stats = load_macro_topic_stats(OUT_DIR)
if not topic_stats.empty:
    print("=== Tableau récapitulatif par macro ===")
    display(format_macro_topic_stats_display(topic_stats))

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
        md(
            "## § Lecture d'un récit (texte coloré par thème / macro)\n"
            "\n"
            "Phrases surlignées selon le topic BERTopic intra-macro (`color_by='topic'`) "
            "ou la macro TPN (`color_by='macro'`). Modifier `ACCIDENT_ID` ou "
            "`os.environ['ACCIDENT_ID']` pour changer de récit."
        ),
        py(
            r"""
from macro_transfer.notebook_viz import (
    build_topics_display_dataframe,
    load_run_artifacts,
    pick_accident_id_for_colored_text,
    show_colored_text_inline,
)

_assign_path = OUT_DIR / "topics_bertopic" / "assignments.csv"
if not _assign_path.is_file():
    print(
        "assignments.csv absent — relancer le pipeline TPN avec BERTopic "
        f"(attendu : {_assign_path})"
    )
elif "accident_id" not in meta_adapt.columns:
    print("Colonne accident_id absente des métadonnées TPN.")
else:
    artifacts = load_run_artifacts(OUT_DIR)
    _df_topics = build_topics_display_dataframe(artifacts, confidence_threshold=0.0)
    _counts = _df_topics.groupby("accident_id").size().sort_values(ascending=False)
    print("Top accidents (nb phrases / unités) :")
    display(_counts.head(10).to_frame("n_units"))

    _acc_env = os.environ.get("ACCIDENT_ID", "").strip()
    if _acc_env:
        _candidates = _df_topics["accident_id"].dropna().unique()
        _dtype = type(_candidates[0]) if len(_candidates) else str
        try:
            _prefer = _dtype(_acc_env)
        except (TypeError, ValueError):
            _prefer = _acc_env
    else:
        _prefer = None

    ACCIDENT_ID = pick_accident_id_for_colored_text(
        _df_topics, min_units=5, prefer_id=_prefer
    )
    _n_units = int(_counts.get(ACCIDENT_ID, 0))
    print(f"Récit affiché : accident_id={ACCIDENT_ID} ({_n_units} unités textuelles)")

    show_colored_text_inline(
        _df_topics,
        ACCIDENT_ID,
        min_prob=0.0,
        font_size_px=10,
        legend_font_size_px=9,
        legend_title="Thèmes",
        show_prob=False,
        color_by="topic",
        highlight_style="border",
        keep_outliers_plain=True,
    )
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
