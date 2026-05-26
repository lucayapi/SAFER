"""Génère notebooks/07_macro_transfer_interactive.ipynb (run local + viz 2D)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "07_macro_transfer_interactive.ipynb"

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
# 07 — Transfert macro TPN interactif (run local + visualisations)

Exécute le pipeline **TPN** sur un corpus de test (sans SLURM), puis affiche métriques, tableaux et cartes **UMAP / DataMapPlot / PCA-t-SNE**.

| Notebook | Usage |
|----------|--------|
| **07** (ce fichier) | Run local TPN (`BASE_METHOD`) + viz 2D |
| **06** | Lecture comparative **TPN-SCGM vs TPN-SoftTriple** après jobs cluster |
| **08** | Diagnostics initial/adapté (lecture seule) |

**Sorties** : `output_test/<TEST_CORPUS>/macro_transfer/tpn_<BASE_METHOD>/`

**Prérequis** :
- Checkpoint BTP : `output/<BASE_METHOD>/checkpoints/`
- Embeddings Qwen test : `embeddings/test/Qwen3-Embedding-0.6B_<corpus>.csv`
- GPU recommandé ; OpenAI si `bertopic.representation` activé dans `configs/tpn_macro_transfer.yaml`
"""
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + """
from pathlib import Path

# --- Parameters (papermill-friendly) ---
TEST_CORPUS = "metallurgie"
BASE_METHOD = "softtriple"  # softtriple | supcon | batch_triplet | scgm_text
RUN_PIPELINE = True
DEVICE = "cuda"  # cpu si pas de GPU
CONFIDENCE_THRESHOLD = 0.5
CONFIG_PATH = "configs/tpn_macro_transfer.yaml"
FIG_DIR = None  # défaut : notebooks/figures/07_tpn_<corpus>/

import matplotlib.pyplot as plt
import seaborn as sns

from macro_transfer.tpn_encode import resolve_tpn_checkpoint, tpn_method_name, validate_encoder_name
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import (
    macro_transfer_output_dir,
    resolve_test_corpus,
    resolve_test_paths_from_config,
)

validate_encoder_name(BASE_METHOD)
_spec = resolve_test_corpus(TEST_CORPUS, anchor=TEXT_ROOT)
OUT_DIR = macro_transfer_output_dir(tpn_method_name(BASE_METHOD), _spec.id, anchor=TEXT_ROOT)
FIG_DIR = Path(FIG_DIR) if FIG_DIR else TEXT_ROOT / "notebooks" / "figures" / f"07_tpn_{_spec.id}_{OUT_DIR.name}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

cfg_path = resolve_repo_path(CONFIG_PATH, repo_root=TEXT_ROOT)
RAW_CFG = load_yaml(cfg_path)
RAW_CFG = {
    **RAW_CFG,
    "corpus": _spec.id,
    "base_method": BASE_METHOD,
    "device": DEVICE,
    "repo_anchor": str(TEXT_ROOT),
}

_spec2, TARGET_DATA_CSV, EMB_CSV = resolve_test_paths_from_config(
    RAW_CFG, corpus_id=_spec.id, anchor=TEXT_ROOT
)
_source = RAW_CFG.get("source") or {}
SOURCE_DATA_CSV = resolve_repo_path(
    _source.get("data_csv", "dataset/data_btp.csv"), repo_root=TEXT_ROOT
)
_method_cfg = dict(RAW_CFG.get("method") or {})
_method_cfg["base_method"] = BASE_METHOD
RAW_CFG["method"] = _method_cfg
CHECKPOINT = resolve_repo_path(
    resolve_tpn_checkpoint(
        BASE_METHOD,
        _method_cfg,
        RAW_CFG.get("checkpoints") or {},
    ),
    repo_root=TEXT_ROOT,
)

print(f"Corpus : {_spec2.display_name} ({_spec2.id})")
print("BASE_METHOD :", BASE_METHOD)
print("Sortie :", OUT_DIR)
print("Checkpoint :", CHECKPOINT)
print("source_data_csv :", SOURCE_DATA_CSV)
print("target_data_csv :", TARGET_DATA_CSV)
print("emb_csv :", EMB_CSV)
print("RUN_PIPELINE :", RUN_PIPELINE)

sns.set_theme(style="whitegrid")
"""
        ),
        md("## Prérequis"),
        py(
            r"""
missing = []
if not CHECKPOINT.is_file() and not CHECKPOINT.is_dir():
    missing.append(f"checkpoint : {CHECKPOINT}")
emb_path = Path(EMB_CSV)
if not emb_path.is_file():
    missing.append(f"embeddings test : {emb_path}")
    print("→ Exporter : python scripts/export_test_embeddings.py --corpus", TEST_CORPUS)
data_path = Path(TARGET_DATA_CSV)
if not data_path.is_file():
    missing.append(f"data test : {data_path}")

if missing:
    raise FileNotFoundError("Prérequis manquants :\n  " + "\n  ".join(missing))
print("Prérequis OK.")
"""
        ),
        md("## Pipeline TPN (optionnel)"),
        py(
            r"""
from macro_transfer.tpn_pipeline import run_tpn_macro_transfer_discovery

if RUN_PIPELINE:
    print("=== Lancement pipeline TPN ===")
    _emb_path = Path(EMB_CSV)
    _emb = str(_emb_path) if _emb_path.is_file() else None
    manifest = run_tpn_macro_transfer_discovery(
        checkpoint=str(CHECKPOINT),
        source_data_csv=str(SOURCE_DATA_CSV),
        target_data_csv=str(TARGET_DATA_CSV),
        output_dir=str(OUT_DIR),
        config=RAW_CFG,
        device=DEVICE,
        emb_csv=_emb,
        skip_bertopic=bool(RAW_CFG.get("skip_bertopic", False)),
    )
    print("Terminé — n_units :", manifest.get("n_units"))
    m = (manifest.get("metrics_adapted") or {})
    if m:
        print(f"accuracy={m.get('accuracy')} macro_f1={m.get('macro_f1')} mean_q_conf={m.get('mean_q_conf')}")
else:
    print("RUN_PIPELINE=False — lecture depuis disque :", OUT_DIR)
"""
        ),
        md("## Transfert macro — métriques et histogrammes"),
        py(
            r"""
from macro_transfer.notebook_viz import load_run_artifacts, plot_transfer_macro_overview
from macro_transfer.report_tables import format_transfer_metrics_table, load_transfer_metrics_pair

artifacts = load_run_artifacts(OUT_DIR)
plot_transfer_macro_overview(
    artifacts,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    fig_dir=FIG_DIR,
)

m_init, m_adapt = load_transfer_metrics_pair(OUT_DIR)
if m_init or m_adapt:
    display(format_transfer_metrics_table(m_init, m_adapt, BASE_METHOD))
"""
        ),
        md("## t-SNE source BTP vs test — initial vs adapté"),
        py(
            r"""
from macro_transfer.notebook_viz import plot_domain_tsne_side_by_side

plot_domain_tsne_side_by_side(
    OUT_DIR,
    FIG_DIR,
    max_points=4000,
    source_label="Source BTP",
    target_label=f"Test {TEST_CORPUS}",
    test_corpus_name=TEST_CORPUS,
)
"""
        ),
        md("## Carte globale — UMAP / DataMapPlot (couleur m_hat)"),
        py(
            r"""
from macro_transfer.notebook_viz import plot_global_embedding_map

plot_global_embedding_map(
    artifacts,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    fig_dir=FIG_DIR,
    use_datamap=True,
)
"""
        ),
        md("## Topics — BERTopic (c-TF-IDF + representation OpenAI)"),
        py(
            r"""
import pandas as pd
from IPython.display import display
from macro_transfer.notebook_viz import display_topics_tables

display_topics_tables(OUT_DIR, "topics_bertopic", "BERTopic")

themes_path = OUT_DIR / "topics_bertopic" / "themes_by_macro.csv"
if themes_path.is_file():
    th = pd.read_csv(themes_path)
    if "theme_label" in th.columns:
        print("=== Libellés (bertopic.representation.OpenAI) ===")
        display(th[["macro", "topic_id", "n_units", "theme_label", "top_words"]].head(20))

summary = OUT_DIR / "summary" / "topics_summary.csv"
if summary.is_file():
    print("=== Résumé topics par macro ===")
    display(pd.read_csv(summary))
"""
        ),
        md("## Visualisations par macro (BERTopic)"),
        py(
            r"""
from macro_transfer.constants import MACRO_NAMES
from macro_transfer.notebook_viz import plot_topics_per_macro

for macro in MACRO_NAMES:
    print(f"--- BERTopic / {macro} ---")
    plot_topics_per_macro(
        artifacts,
        topic_subdir="topics_bertopic",
        algo_tag="bertopic",
        macro=macro,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        fig_dir=FIG_DIR,
        use_datamap=True,
        run_pca_tsne=True,
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
