"""Génère notebooks/06_macro_transfer_topics.ipynb (lecture seule, corpus test configurable)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "06_macro_transfer_topics.ipynb"

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
# 06 — Transfert macro TPN + topics intra-macro (corpus test)

Comparaison **TPN-SCGM** vs **TPN-SoftTriple** (`output_test/<corpus>/macro_transfer/tpn_<encodeur>/`) :
- Gating macro adapté (`transfer/metadata_with_tpn_macro_probs.csv`, `transfer_metrics_adapted.json`)
- Topics **BERTopic** par macro + libellés OpenAI (`theme_label`)
- Cartes **UMAP + DataMapPlot** (embeddings `target_adapted.npy`)

**Prérequis** : lancer les deux encodeurs, ex. `BASE_METHOD=scgm_text` puis `BASE_METHOD=softtriple` via `jobs/run_tpn_macro_transfer.sh`.
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from safer_core.test_corpus import resolve_test_corpus, macro_transfer_output_dir

# --- Parameters (modifier ici ou via papermill) ---
TEST_CORPUS = "metallurgie"
CONFIDENCE_THRESHOLD = 0.5  # filtre q_conf pour cartes topics
UMAP_MAX_POINTS = 8000
TOPIC_UMAP_MAX_POINTS = 4000
USE_DATAMAP = True
PLOT_MACROS = None  # None → toutes (A0, A1, B, C) ; ou liste ex. ["A1", "B"]
RUN_PCA_TSNE_PER_MACRO = True

_spec = resolve_test_corpus(TEST_CORPUS, anchor=TEXT_ROOT)
SCGM_DIR = macro_transfer_output_dir("tpn_scgm_text", _spec.id, anchor=TEXT_ROOT)
SOFT_DIR = macro_transfer_output_dir("tpn_softtriple", _spec.id, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"06_tpn_macro_{_spec.id}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Corpus test : {_spec.display_name} ({_spec.id})")
print("TPN-SCGM :", SCGM_DIR)
print("TPN-SoftTriple :", SOFT_DIR)

sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Transfert macro"),
        py(
            r"""
def load_transfer(root: Path, label: str):
    tdir = root / "transfer"
    metrics_path = tdir / "transfer_metrics_adapted.json"
    meta_path = tdir / "metadata_with_tpn_macro_probs.csv"
    if not metrics_path.is_file():
        print(f"[{label}] absent:", metrics_path)
        return None, None
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    meta = pd.read_csv(meta_path) if meta_path.is_file() else None
    return metrics, meta

rows = []
for label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    m, _ = load_transfer(root, label)
    if m:
        rows.append({
            "method": label,
            "n_eval": m.get("n_eval"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "mean_q_conf": m.get("mean_q_conf"),
        })
transfer_df = pd.DataFrame(rows)
display(transfer_df)
"""
        ),
        py(
            r"""
for label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    m, meta = load_transfer(root, label)
    if meta is None or "pred_label" not in meta.columns:
        continue
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(meta["q_conf"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="crimson", ls="--", label="seuil 0.5")
    ax.set_title(f"Confiance macro — {label} ({TEST_CORPUS})")
    ax.set_xlabel("q_conf = max_m p(m|u)")
    ax.legend()
    fig.savefig(FIG_DIR / f"hist_q_conf_{label.lower()}.png", dpi=120, bbox_inches="tight")
    plt.show()
"""
        ),
        md("## § Tableau récapitulatif topics (corpus test)"),
        py(
            r"""
from macro_transfer.report_tables import load_macro_topic_stats
from macro_transfer.topics_export import format_macro_topic_stats_display

for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    stats = load_macro_topic_stats(root)
    if stats.empty:
        print(f"[{method_label}] macro_topic_stats absent — relancer jobs/run_tpn_macro_transfer.sh")
        continue
    print(f"\n=== {method_label} — topics par macro ({TEST_CORPUS}) ===")
    display(format_macro_topic_stats_display(stats))
"""
        ),
        md("## § Topics BERTopic"),
        py(
            r"""
def load_topics(root: Path):
    p = root / "topics_bertopic" / "themes_by_macro.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()

for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    summary_path = root / "summary" / "topics_summary.csv"
    if summary_path.is_file():
        print("===", method_label, "— résumé topics ===")
        display(pd.read_csv(summary_path))
    th = load_topics(root)
    if len(th):
        print(method_label, "BERTopic — effectifs par macro")
        display(th.groupby("macro")["n_units"].sum().reset_index())
"""
        ),
        py(
            r"""
for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    th = load_topics(root)
    if len(th) and "theme_label" in th.columns:
        print(f"\n{method_label} — libellés (bertopic.representation.OpenAI)")
        cols = ["macro", "topic_id", "n_units", "theme_label", "top_words"]
        display(th[[c for c in cols if c in th.columns]].head(12))
    elif len(th) and "top_words" in th.columns:
        print(f"\n{method_label} — top_words (sans theme_label)")
        cols = ["macro", "topic_id", "n_units", "top_words"]
        display(th[[c for c in cols if c in th.columns]].head(12))
"""
        ),
        md("## § Qualité (support, topics vides)"),
        py(
            r"""
quality_rows = []
for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    th = load_topics(root)
    if not len(th):
        continue
    quality_rows.append({
        "method": method_label,
        "n_topics": th["topic_id"].nunique(),
        "n_empty": int((th["n_units"] == 0).sum()),
        "min_support": int(th["n_units"].min()),
        "median_support": float(th["n_units"].median()),
    })
display(pd.DataFrame(quality_rows))
"""
        ),
        md(
            r"""
## § Cartes 2D — topics avec libellés (`theme_label`)

Embeddings : `embeddings/target_adapted.npy`. Assignations : `topics_bertopic/assignments.csv` + `themes_by_macro.csv`.

**Important** : la carte « macro seule » (A0, A1, B, C) ne montre **pas** les libellés de topics. Les cellules ci-dessous utilisent **`theme_label`** (colonne OpenAI / BERTopic) sur **chaque topic** — format `A1·T5: Absence de protection…`.

- **Globale topics** : tous les topics assignés (TPN-SCGM puis TPN-SoftTriple)
- **Comparaison** : panneau 1×2 TPN-SCGM vs TPN-SoftTriple (scatter + centroïdes annotés)
- **Par macro** : zoom intra-macro (optionnel, section suivante)
- **Macro seule** (`m_hat`) : section optionnelle en fin de notebook
"""
        ),
        py(
            r"""
from macro_transfer.constants import MACRO_NAMES
from macro_transfer.notebook_viz import (
    load_run_artifacts,
    plot_global_embedding_map,
    plot_global_topics_compare_methods,
    plot_global_topics_datamap,
    plot_topics_per_macro,
)

_macros = list(MACRO_NAMES) if PLOT_MACROS is None else list(PLOT_MACROS)


def _load_artifacts_or_none(method_label: str, root: Path):
    try:
        art = load_run_artifacts(root)
        print(f"[{method_label}] OK — {len(art.z)} vecteurs, {root}")
        return art
    except FileNotFoundError as exc:
        print(f"[{method_label}] carte 2D ignorée : {exc}")
        return None


ARTIFACTS = {}
for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    ARTIFACTS[method_label] = _load_artifacts_or_none(method_label, root)
"""
        ),
        py(
            r"""
# Libellés topics (theme_label) — une carte DataMapPlot par méthode
for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    art = ARTIFACTS.get(method_label)
    if art is None:
        continue
    method_fig = FIG_DIR / method_label.lower()
    method_fig.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Topics + theme_label — {method_label} ===")
    plot_global_topics_datamap(
        art,
        algo_tag=method_label,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_points=UMAP_MAX_POINTS,
        fig_dir=method_fig,
        use_datamap=USE_DATAMAP,
        label_font_size=7,
    )
"""
        ),
        py(
            r"""
# TPN-SCGM vs TPN-SoftTriple côte à côte (même seuil q_conf)
_compare = {k: v for k, v in ARTIFACTS.items() if v is not None}
if _compare:
    plot_global_topics_compare_methods(
        _compare,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_points=UMAP_MAX_POINTS,
        seed=42,
        fig_dir=FIG_DIR,
        use_datamap=USE_DATAMAP,
    )
"""
        ),
        md("### Option — carte macro seule (`m_hat`, sans libellés topics)"),
        py(
            r"""
for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    art = ARTIFACTS.get(method_label)
    if art is None:
        continue
    method_fig = FIG_DIR / method_label.lower()
    print(f"\n=== Macro m_hat seule — {method_label} ===")
    plot_global_embedding_map(
        art,
        max_points=UMAP_MAX_POINTS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        fig_dir=method_fig,
        use_datamap=USE_DATAMAP,
    )
"""
        ),
        py(
            r"""
for method_label, root in (("TPN-SCGM", SCGM_DIR), ("TPN-SoftTriple", SOFT_DIR)):
    art = ARTIFACTS.get(method_label)
    if art is None:
        continue
    method_fig = FIG_DIR / method_label.lower()
    print(f"\n=== Topics intra-macro — {method_label} ===")
    for macro in _macros:
        print(f"--- {method_label} / {macro} ---")
        plot_topics_per_macro(
            art,
            topic_subdir="topics_bertopic",
            algo_tag=method_label,
            macro=macro,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            max_points=TOPIC_UMAP_MAX_POINTS,
            fig_dir=method_fig,
            use_datamap=USE_DATAMAP,
            run_pca_tsne=RUN_PCA_TSNE_PER_MACRO,
        )
print("Figures :", FIG_DIR)
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("OK:", NB_PATH)


if __name__ == "__main__":
    main()
