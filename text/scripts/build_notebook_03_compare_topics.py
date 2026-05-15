"""Génère notebooks/03_compare_malt_bertopic_kmeans_topics.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "03_compare_malt_bertopic_kmeans_topics.ipynb"


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
## 1 — Introduction

# 03 — Comparaison MALT vs BERTopic intra-macro vs KMeans intra-macro

**MALT** produit des motifs latents \(z\) sur la cible après adaptation.

**BERTopic intra-macro via p0** : transfert source → projecteur → ancres → **\(p_0(y|x)\)** sur la cible ; pseudo-macro dure par argmax (`p0_macro_name`) ; **BERTopic** séparé par macro (sans adaptation MALT).

**KMeans intra-macro + c-TF-IDF** : même découpage macro que BERTopic ; KMeans + c-TF-IDF.

**Métriques principales** : C_v, NPMI, Topic Diversity, Redundancy, Coverage. Section **macro** : distributions \(p_0\), \(p_t\), transition \(p_0 \rightarrow p_t\).
"""
        ),
        md("## 2 — Imports et paramètres"),
        py(
            r"""
# --- Paramètres (Papermill / exécution manuelle) ---
# Chemins relatifs ci-dessous : racine = dossier du dépôt (parent de topic_eval/), pas notebooks/.
MALT_EXPORTS_DIR = "runs/malt_btp_to_mettalurgie_qwen06/exports"
MALT_EVAL_DIR = "runs/malt_btp_to_mettalurgie_qwen06/evaluation"
OUTPUT_DIR = "outputs/topic_comparison"
TEXT_COL = "sentence"
MACRO_P0_COL = "p0_macro_name"
MACRO_PT_COL = "pt_macro_name"
Z_COL = "z_hat"
N_TOP_WORDS = 10
N_REPRESENTATIVE_SENTENCES = 8
N_BERTOPIC_TOPICS_PER_MACRO = "auto"
N_KMEANS_CLUSTERS_PER_MACRO = 8
RANDOM_STATE = 42
MIN_TOPIC_SIZE = 15
MAX_DOCS_FOR_BERTOPIC = None
USE_PROJECTED_SOURCE_EMBEDDINGS = True
USE_PROJECTED_ADAPTED_EMBEDDINGS_FOR_MALT = True
"""
        ),
        py(
            r"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np

if int(np.__version__.split(".", 1)[0]) >= 2:
    _py = sys.executable
    raise ImportError(
        "NumPy 2.x est chargé alors que le matplotlib de cet environnement (Anaconda) "
        "est souvent compilé pour NumPy 1.x → conflit à l’import de matplotlib.\n\n"
        f"Interpréteur du noyau : {_py}\n\n"
        "Corrigez puis Kernel → Restart, par exemple :\n\n"
        f'  "{_py}" -m pip install "numpy<2" --force-reinstall\n\n'
        "ou :\n\n"
        '  conda install "numpy<2" "matplotlib" "scipy" -y\n\n'
        "Réinstallez aussi les deps du dépôt : pip install -r requirements.txt"
    )

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import display

def _find_repo_root() -> Path:
    # Racine du pipeline texte (cwd peut être notebooks/ ou racine SAFER).
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "topic_eval" / "__init__.py").is_file():
            return candidate
        if (candidate / "text" / "topic_eval" / "__init__.py").is_file():
            return candidate / "text"
    return here


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print("PYTHONPATH racine :", REPO_ROOT)

from topic_eval.bertopic_baseline import run_bertopic_intra_macro
from topic_eval.compare_topics import (
    build_malt_topics_df,
    build_topics_by_macro_qualitative,
    check_malt_export_files,
    load_embeddings_and_metadata,
    write_topic_comparison_report_md,
)
from topic_eval.paths import resolve_repo_path
from topic_eval.kmeans_ctfidf_baseline import run_kmeans_ctfidf_intra_macro
from topic_eval.metrics_topic_quality import (
    build_metrics_both_filters,
    macro_level_metrics,
)
from topic_eval.topic_cleaning import load_domain_stopwords
from topic_eval.visualization import save_all_comparison_figures

sns.set_theme(style="whitegrid")
"""
        ),
        md("## 3 — Chargement des données et contrôle d’alignement"),
        py(
            r"""
exports_path = resolve_repo_path(MALT_EXPORTS_DIR, REPO_ROOT)
OUTPUT_DIR = str(resolve_repo_path(OUTPUT_DIR, REPO_ROOT))
check_malt_export_files(exports_path)

meta, emb_source, emb_adapted, extras = load_embeddings_and_metadata(exports_path, TEXT_COL)
p0_arr = extras["p0"]
pz_arr = extras["pz"]
prob_y_z = extras["prob_y_z"]
nu_arr = extras["nu"]

if TEXT_COL not in meta.columns:
    raise KeyError(f"Colonne texte absente : {TEXT_COL}")

text_series = meta[TEXT_COL].astype(str)
valid_mask = text_series.notna() & (text_series.str.strip() != "") & (text_series.str.lower() != "nan")
idx = np.flatnonzero(valid_mask.to_numpy())
meta = meta.iloc[idx].reset_index(drop=True)
emb_source = np.asarray(emb_source)[idx]
emb_adapted = np.asarray(emb_adapted)[idx]
extras["p0"] = p0_arr = p0_arr[idx]
extras["pt"] = extras["pt"][idx]
extras["pz"] = pz_arr = pz_arr[idx]

if MAX_DOCS_FOR_BERTOPIC is not None and int(MAX_DOCS_FOR_BERTOPIC) > 0:
    n_cap = int(MAX_DOCS_FOR_BERTOPIC)
    meta = meta.iloc[:n_cap].reset_index(drop=True)
    sl = slice(None, n_cap)
    emb_source = emb_source[sl]
    emb_adapted = emb_adapted[sl]
    p0_arr = p0_arr[sl]
    pz_arr = pz_arr[sl]
    extras["pt"] = extras["pt"][sl]

docs = meta[TEXT_COL].astype(str).tolist()
n_valid = len(meta)
print("Documents avec texte valide :", n_valid)

STOPWORDS_PATH = Path(OUTPUT_DIR) / "stopwords_domain.txt"
stopwords_domain = load_domain_stopwords(STOPWORDS_PATH)

tables_dir = Path(OUTPUT_DIR) / "tables"
fig_dir = Path(OUTPUT_DIR) / "figures"
tables_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

display(meta[MACRO_P0_COL].value_counts().rename("p0_macro_name"))
display(meta[MACRO_PT_COL].value_counts().rename("pt_macro_name"))
vc_z = meta[Z_COL].value_counts()
print("Nombre de z actifs :", int(vc_z.shape[0]))
display(vc_z.head(20))
"""
        ),
        md("### Graphiques exploratoires (distributions macro et z)"),
        py(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
meta[MACRO_P0_COL].value_counts().reindex(["A0", "A1", "B", "C"]).plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Distribution p0_macro_name")
meta[MACRO_PT_COL].value_counts().reindex(["A0", "A1", "B", "C"]).plot(kind="bar", ax=axes[1], color="coral")
axes[1].set_title("Distribution pt_macro_name")
plt.tight_layout()
plt.savefig(fig_dir / "distributions_p0_pt.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 4))
meta[Z_COL].value_counts().sort_index().plot(kind="bar")
plt.title("Tailles des clusters MALT (z_hat)")
plt.xlabel("z")
plt.ylabel("Effectif")
plt.tight_layout()
plt.savefig(fig_dir / "z_cluster_sizes.png", dpi=150)
plt.show()

ct = pd.crosstab(meta[MACRO_P0_COL], meta[MACRO_PT_COL])
plt.figure(figsize=(6, 5))
sns.heatmap(ct, annot=True, fmt="d", cmap="Purples")
plt.title("Transition p0_macro → pt_macro")
plt.xlabel("pt_macro_name")
plt.ylabel("p0_macro_name")
plt.tight_layout()
plt.savefig(fig_dir / "heatmap_p0_to_pt.png", dpi=150)
plt.show()
"""
        ),
        md(
            """
## 4 — Nettoyage des top words

Implémenté dans `topic_eval/topic_cleaning.py` : `load_domain_stopwords`, `normalize_token`, `clean_top_words`, `preprocess_for_ctfidf` (minuscules, ponctuation, tokens courts, chiffres, stopwords FR + fichier métier).
"""
        ),
        md("## 5 — Topics MALT (z_hat) + c-TF-IDF"),
        py(
            r"""
emb_malt = emb_adapted if USE_PROJECTED_ADAPTED_EMBEDDINGS_FOR_MALT else emb_source

malt_topics_df = build_malt_topics_df(
    meta=meta,
    docs=docs,
    projected_adapted=emb_malt,
    pz=pz_arr,
    prob_y_z=prob_y_z,
    nu=nu_arr,
    stopwords_domain=stopwords_domain,
    text_col=TEXT_COL,
    n_top_words=N_TOP_WORDS,
    n_representative_sentences=N_REPRESENTATIVE_SENTENCES,
    z_col=Z_COL,
)
malt_topics_df.to_csv(tables_dir / "malt_topics.csv", index=False)
display(malt_topics_df.head(12))
"""
        ),
        md("## 6 — Baseline BERTopic intra-macro (p0 uniquement)"),
        py(
            r"""
emb_baseline = emb_source if USE_PROJECTED_SOURCE_EMBEDDINGS else emb_adapted

bertopic_topics_df = run_bertopic_intra_macro(
    docs=docs,
    embeddings=emb_baseline,
    macro_labels=meta[MACRO_P0_COL].tolist(),
    stopwords_domain=stopwords_domain,
    min_topic_size=MIN_TOPIC_SIZE,
    random_state=RANDOM_STATE,
    n_top_words=N_TOP_WORDS,
    n_representative_sentences=N_REPRESENTATIVE_SENTENCES,
)
bertopic_topics_df.to_csv(tables_dir / "bertopic_intra_macro_topics.csv", index=False)
display(bertopic_topics_df.head(12))
"""
        ),
        md("## 7 — Baseline KMeans intra-macro + c-TF-IDF"),
        py(
            r"""
kmeans_topics_df = run_kmeans_ctfidf_intra_macro(
    docs=docs,
    embeddings=emb_baseline,
    macro_labels=meta[MACRO_P0_COL].tolist(),
    n_clusters_per_macro=N_KMEANS_CLUSTERS_PER_MACRO,
    stopwords_domain=stopwords_domain,
    random_state=RANDOM_STATE,
    n_top_words=N_TOP_WORDS,
    n_representative_sentences=N_REPRESENTATIVE_SENTENCES,
)
kmeans_topics_df.to_csv(tables_dir / "kmeans_intra_macro_topics.csv", index=False)
display(kmeans_topics_df.head(12))
"""
        ),
        md("## 8–9 — Métriques et tableau comparatif"),
        py(
            r"""
methods = [
    ("MALT adapted + c-TF-IDF", malt_topics_df),
    ("BERTopic intra-macro via p0", bertopic_topics_df),
    ("KMeans intra-macro + c-TF-IDF", kmeans_topics_df),
]

rows_main = []
rows_raw = []
for label, tdf in methods:
    m, r = build_metrics_both_filters(tdf, docs, label, stopwords_domain, N_TOP_WORDS, MIN_TOPIC_SIZE)
    rows_main.append(m.iloc[0].to_dict())
    rows_raw.append(r.iloc[0].to_dict())

metrics_filtered = pd.DataFrame(rows_main)
metrics_unfiltered = pd.DataFrame(rows_raw)
print("=== Métriques (topics avec n_docs >= MIN_TOPIC_SIZE) ===")
display(metrics_filtered.round(3))

comparison_metrics_df = metrics_filtered[
    [
        "method",
        "n_topics",
        "coverage",
        "cv",
        "npmi",
        "topic_diversity",
        "redundancy",
        "mean_topic_size",
        "median_topic_size",
    ]
].copy()
comparison_metrics_df.to_csv(tables_dir / "topic_quality_comparison.csv", index=False)
comparison_metrics_df.to_latex(tables_dir / "topic_quality_comparison.tex", index=False, float_format="%.3f")
metrics_unfiltered.to_csv(tables_dir / "topic_quality_comparison_unfiltered.csv", index=False)
"""
        ),
        md("## 10 — Figures comparatives"),
        py(
            r"""
topics_long = pd.concat([malt_topics_df, bertopic_topics_df, kmeans_topics_df], ignore_index=True)
save_all_comparison_figures(comparison_metrics_df, topics_long, fig_dir)
print("Figures enregistrées dans", fig_dir)

# Tableau visuel (top 10 par méthode)
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
for ax, (title, df) in zip(
    axes,
    [
        ("MALT (top 10)", malt_topics_df),
        ("BERTopic (top 10)", bertopic_topics_df),
        ("KMeans (top 10)", kmeans_topics_df),
    ],
):
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    sub = df.sort_values("n_docs", ascending=False).head(10)
    if sub.empty:
        continue
    tbl = ax.table(
        cellText=sub[["topic_id", "macro", "n_docs", "top_words"]].values,
        colLabels=["topic_id", "macro", "n_docs", "top_words"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.4)
plt.tight_layout()
plt.savefig(fig_dir / "top10_topics_table_by_method.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md("## 11 — Exemples de topics"),
        py(
            r"""
from typing import Optional

def display_topic_examples(
    topics_df: pd.DataFrame,
    method_name: Optional[str] = None,
    macro: Optional[str] = None,
    n_topics: int = 5,
):
    df = topics_df.copy()
    if method_name:
        df = df[df["method"].astype(str).str.contains(method_name, case=False, na=False)]
    if macro:
        df = df[df["macro"].astype(str) == macro]
    df = df.sort_values("n_docs", ascending=False).head(n_topics)
    for _, r in df.iterrows():
        print("\n---")
        print("méthode:", r.get("method"))
        print("topic_id:", r.get("topic_id"), "| macro:", r.get("macro"), "| n_docs:", r.get("n_docs"))
        print("top_words:", r.get("top_words"))
        print("phrases:\n", str(r.get("top_sentences", "")).replace(" || ", "\n"))

display_topic_examples(malt_topics_df, "MALT", None, 5)
display_topic_examples(bertopic_topics_df, "BERTopic", None, 5)
display_topic_examples(kmeans_topics_df, "KMeans", None, 5)
for macro in ["A0", "A1", "B", "C"]:
    print("\n===== Macro", macro, "=====")
    display_topic_examples(malt_topics_df, "MALT", macro, 4)
    display_topic_examples(bertopic_topics_df, "BERTopic", macro, 4)
    display_topic_examples(kmeans_topics_df, "KMeans", macro, 4)
"""
        ),
        md("## 12 — Tableau qualitatif aligné par macro"),
        py(
            r"""
qual_df = build_topics_by_macro_qualitative(malt_topics_df, bertopic_topics_df, kmeans_topics_df)
qual_df.to_csv(tables_dir / "topics_by_macro_qualitative.csv", index=False)
display(qual_df.head(20))
"""
        ),
        md("## 13 — Qualité par macro (p0 sur les documents)"),
        py(
            r"""
macro_rows = []
for label, tdf in methods:
    macro_rows.append(macro_level_metrics(tdf, docs, MACRO_P0_COL, meta, label, stopwords_domain, N_TOP_WORDS, MIN_TOPIC_SIZE))
macro_topic_quality = pd.concat(macro_rows, ignore_index=True)
macro_topic_quality.to_csv(tables_dir / "macro_topic_quality.csv", index=False)
macro_topic_quality.to_latex(tables_dir / "macro_topic_quality.tex", index=False, float_format="%.3f")
display(macro_topic_quality.round(3))
"""
        ),
        md("## 14 — Rapport Markdown automatique"),
        py(
            r"""
write_topic_comparison_report_md(
    Path(OUTPUT_DIR) / "topic_comparison_report.md",
    comparison_metrics_df,
    str(STOPWORDS_PATH),
    n_valid,
    str(fig_dir),
)
print("Rapport :", Path(OUTPUT_DIR) / "topic_comparison_report.md")
"""
        ),
        md("## 15 — Export LaTeX"),

        py(
            r"""
# Déjà écrit : topic_quality_comparison.tex, macro_topic_quality.tex
print("LaTeX :", tables_dir / "topic_quality_comparison.tex")
"""
        ),
        md(
            """
## 16 — Notes

- Ne pas utiliser `pred_label` comme référence principale.
- Baselines intra-macro : **uniquement** `p0_macro_name`.
- Dépendances manquantes : `pip install bertopic` ou `pip install gensim` selon les messages d’erreur.
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Notebook écrit :", NB_PATH)


if __name__ == "__main__":
    main()
