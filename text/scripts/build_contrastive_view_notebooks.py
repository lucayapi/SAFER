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


def _config_code(method_key: str, display_name: str) -> str:
    return f'''# --- Paramètres (modifier ici) ---
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

ROOT = TEXT_ROOT  # défini par la cellule bootstrap

from safer_core.brand_style import apply_matplotlib_brand

apply_matplotlib_brand()
plt.rcParams["figure.dpi"] = 96
plt.rcParams["figure.autolayout"] = True

METHOD_KEY = "{method_key}"
DISPLAY_NAME = "{display_name}"
CONFIG_PATH = ROOT / "configs/methods" / "{method_key}.yaml"

cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
_data = cfg.get("data") or {{}}
_model = cfg.get("model") or {{}}
DATASET_CSV = ROOT / _data.get("dataset_path", cfg.get("dataset_path", "dataset/data_btp.csv"))
DEFAULT_OUTPUT = ROOT / cfg.get("output_dir", f"output/{{METHOD_KEY}}")

# Dossier de résultats à considérer (relatif à ROOT ou chemin absolu)
# Modifier RESULTS_DIR pour pointer vers un run spécifique (standard, combo tuning, WP3, …)
RESULTS_DIR = DEFAULT_OUTPUT
# Exemples :
# RESULTS_DIR = ROOT / "output/{method_key}/tuning/combos/projectionlinear_hiddim128_abc12345"
# RESULTS_DIR = Path("/chemin/vers/mon_experience")

LABEL_COL = _data.get("label_col", cfg.get("label_col", "pred_label"))
BACKBONE = _model.get("backbone_name", cfg.get("backbone_name", "(non défini)"))
TEST_CORPORA = list(_data.get("test_corpora") or cfg.get("test_corpora") or ["metallurgie", "caou"])
FIGURES_DIR = RESULTS_DIR / "figures_notebook"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Méthode  :", DISPLAY_NAME)
print("Config   :", CONFIG_PATH.relative_to(ROOT))
print("Résultats:", RESULTS_DIR)
print("Figures  :", FIGURES_DIR)
print("Corpus OOD:", TEST_CORPORA)
'''


METRICS_MD = """## 1. Métriques de classification

Tableau unique : CV BTP (μ±σ), évaluation LR in-domain (BTP) et OOD (métallurgie, caou).
"""

METRICS_CODE = '''from contrastive_methods.view_metrics import (
    build_view_classification_summary_table,
    format_ood_summary_line,
    validate_contrastive_results_dir,
)
from supervised_macro_ft.notebook_viz import style_metrics_table

validate_contrastive_results_dir(RESULTS_DIR)
summary = build_view_classification_summary_table(RESULTS_DIR, test_corpora=TEST_CORPORA)
display(style_metrics_table(summary, ("balanced_accuracy", "macro_f1", "accuracy")))
ood_line = format_ood_summary_line(RESULTS_DIR)
if ood_line:
    print(ood_line)
'''

PROJECTED_BTP_MD = """## 2. Embeddings projetés — BTP (PCA / t-SNE)

Couleur = macro (`pred_label`). Figures dans `figures_notebook/`.
"""

PROJECTED_BTP_CODE = '''from scgm_text.notebook_viz import plot_projected_embeddings_pca_tsne
from safer_core.test_corpus import resolve_projected_embeddings_paths

def save_fig(name: str) -> Path:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path

pair_btp = resolve_projected_embeddings_paths(
    METHOD_KEY, "btp", anchor=ROOT, results_dir=RESULTS_DIR
)
if pair_btp is None:
    print("(absent) projected_btp — relancer eval_corpus")
else:
    paths = plot_projected_embeddings_pca_tsne(
        pair_btp[0],
        pair_btp[1],
        LABEL_COL,
        corpus_name=f"{DISPLAY_NAME} — BTP (avant classif. LR)",
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        png_name=f"{METHOD_KEY}_btp_pca_tsne.png",
        max_points=8000,
        seed=42,
        include_plotly=False,
    )
    if paths:
        print(paths)
'''

PROJECTED_OOD_MD = """## 3. Embeddings projetés — corpus test OOD (PCA / t-SNE)

Un graphique par corpus configuré (`test_corpora`).
"""

PROJECTED_OOD_CODE = '''from safer_core.test_corpus import resolve_projected_embeddings_paths, resolve_test_corpus

for corpus_id in TEST_CORPORA:
    spec = resolve_test_corpus(corpus_id, anchor=ROOT)
    pair = resolve_projected_embeddings_paths(
        METHOD_KEY, corpus_id, anchor=ROOT, results_dir=RESULTS_DIR
    )
    if pair is None:
        print(f"(absent) projected_{corpus_id}")
        continue

    def _save_fig_ood(name: str, _dir=FIGURES_DIR / f"ood_{corpus_id}") -> Path:
        _dir.mkdir(parents=True, exist_ok=True)
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
        figures_dir=FIGURES_DIR / f"ood_{corpus_id}",
        png_name=f"{METHOD_KEY}_{corpus_id}_pca_tsne.png",
        max_points=8000,
        seed=42,
        include_plotly=False,
    )
    if paths:
        print(f"{corpus_id}:", paths)
'''

SOFTTRIPLE_CENTERS_MD = """## 4. Centres SoftTriple effectifs (softtriple uniquement)

Tableau des centres + PCA/t-SNE avec losanges `A0#0`, …
"""

SOFTTRIPLE_CENTERS_CODE = '''if METHOD_KEY == "softtriple":
    from scgm_text.notebook_viz import (
        plot_embeddings_csv_pca_tsne_with_softtriple_centers,
        resolve_softtriple_centers_csv,
        softtriple_centers_summary_table,
    )
    from safer_core.test_corpus import resolve_final_embeddings_csv_in_dir, resolve_test_corpus

    centers_csv = resolve_softtriple_centers_csv(RESULTS_DIR)
    if centers_csv is None:
        print("(absent) centres SoftTriple")
    else:
        display(softtriple_centers_summary_table(centers_csv))
        emb_btp = resolve_final_embeddings_csv_in_dir(
            RESULTS_DIR, "btp", method=METHOD_KEY, anchor=ROOT
        )
        if emb_btp.is_file():
            p = plot_embeddings_csv_pca_tsne_with_softtriple_centers(
                emb_btp,
                DATASET_CSV,
                LABEL_COL,
                results_dir=RESULTS_DIR,
                corpus_name=f"{DISPLAY_NAME} — BTP (centres)",
                save_fig=save_fig,
                png_name="softtriple_btp_pca_tsne_centers.png",
                max_points=8000,
                seed=42,
            )
            if p is not None:
                print(p)
        for corpus_id in TEST_CORPORA:
            spec = resolve_test_corpus(corpus_id, anchor=ROOT)
            emb_test = resolve_final_embeddings_csv_in_dir(
                RESULTS_DIR, "test", corpus_id=corpus_id, method=METHOD_KEY, anchor=ROOT
            )
            if not emb_test.is_file():
                print(f"(absent) final_embeddings test {corpus_id}")
                continue
            fig_dir = FIGURES_DIR / f"ood_{corpus_id}"
            fig_dir.mkdir(parents=True, exist_ok=True)

            def _save_centers(name: str, _dir=fig_dir) -> Path:
                path = _dir / name
                plt.tight_layout()
                plt.savefig(path, dpi=160, bbox_inches="tight")
                plt.show()
                return path

            p_test = plot_embeddings_csv_pca_tsne_with_softtriple_centers(
                emb_test,
                spec.data_csv,
                LABEL_COL,
                results_dir=RESULTS_DIR,
                corpus_name=f"{DISPLAY_NAME} — {spec.display_name} (centres)",
                save_fig=_save_centers,
                png_name=f"softtriple_{corpus_id}_pca_tsne_centers.png",
                max_points=8000,
                seed=42,
            )
            if p_test is not None:
                print(f"{corpus_id}:", p_test)
'''

RAW_TEST_MD = """## 5. Embeddings bruts OOD (référence Qwen)

Vecteurs encodeur avant fine-tuning contrastif — PCA / t-SNE / UMAP (PNG statiques).
"""

RAW_TEST_CODE = '''from macro_transfer.notebook_viz import plot_test_corpus_raw_embeddings

for corpus_id in TEST_CORPORA:
    spec = resolve_test_corpus(corpus_id, anchor=ROOT)
    fig_dir = FIGURES_DIR / f"raw_{corpus_id}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _raw = plot_test_corpus_raw_embeddings(
        corpus_id,
        fig_dir=fig_dir,
        anchor=ROOT,
        label_col=LABEL_COL,
        max_points=12000,
        seed=42,
        prefix=f"raw_{corpus_id}",
        show=True,
        display_metrics=False,
        include_plotly=False,
        include_tsne_per_macro=False,
    )
    if _raw.missing:
        print(f"{corpus_id} — manquant :", ", ".join(_raw.missing))
    elif _raw.pca_tsne_path:
        print(f"{spec.display_name} :", _raw.pca_tsne_path, _raw.umap_png_path)
'''

BERTOPIC_MD = """## 6. Topic modeling BERTopic (intra-macro)

Segmentation du corpus choisi en sous-corpus **A0 / A1 / B / C** (classe prédite par défaut).
Embeddings = projection contrastive (`projected_*.npy`).
Hyperparamètres : `configs/bertopic_notebook.yaml`.

Sorties : `{RESULTS_DIR}/bertopic_notebook/<corpus>/` (consommées par le notebook **06** BN).
"""

BERTOPIC_CODE = '''BERTOPIC_CORPUS = "metallurgie"
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
_assign = bertopic_out / "topics_bertopic" / "assignments.csv"

preds_bertopic = load_saved_predictions(RESULTS_DIR, BERTOPIC_CORPUS)
if preds_bertopic is not None:
    print("Prédictions job :", RESULTS_DIR / "predictions" / f"predictions_{BERTOPIC_CORPUS}.csv")
else:
    print("Pas de predictions/ en cache — LR sur embeddings projetés si besoin.")

if RESTIMATE_BERTOPIC or not _assign.is_file():
    bt_cfg = dict(cfg_full)
    if BERTOPIC_UMAP_ENABLED is not None:
        bt_cfg.setdefault("bertopic", {}).setdefault("umap", {})["enabled"] = bool(BERTOPIC_UMAP_ENABLED)
    if BERTOPIC_MIN_TOPIC_SIZE is not None:
        bt_cfg.setdefault("bertopic", {})["min_topic_size"] = int(BERTOPIC_MIN_TOPIC_SIZE)
    run_notebook_bertopic(
        RESULTS_DIR,
        BERTOPIC_CORPUS,
        method_name=METHOD_KEY,
        view_kind="contrastive",
        segment_mode=BERTOPIC_SEGMENT_MODE or str(nb_cfg.get("segment_mode", "predicted")),
        label_col=LABEL_COL,
        preds=preds_bertopic,
        bertopic_cfg=bt_cfg.get("bertopic"),
        topics_export_cfg=bt_cfg.get("topics_export"),
        topic_judge_cfg=bt_cfg.get("topic_judge"),
        anchor=ROOT,
        method_key=METHOD_KEY,
        seed=42,
        export_for_bn=bool(nb_cfg.get("export_for_bn", True)),
    )
    print("BERTopic terminé :", bertopic_out)
else:
    print("Cache BERTopic :", bertopic_out)

display_notebook_bertopic_results(bertopic_out)
print("→ Entrée notebook 06 BN :", bertopic_out)
'''


def main() -> None:
    import sys

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP

    NB.mkdir(parents=True, exist_ok=True)
    for method_key, display_name, _config_file in METHODS:
        nb_name = f"05_view_{method_key}_results.ipynb"
        title_md = (
            f"# Résultats — {display_name}\n\n"
            f"**Lecture seule** — métriques LR + projections sur embeddings projetés "
            f"(`projected_*.npy`).\n\n"
            f"Modifiez `RESULTS_DIR` pour pointer vers un autre run (standard, combo tuning, etc.).\n"
        )
        cells = [
            _cell(title_md, "markdown"),
            _cell(NOTEBOOK_PATH_SETUP),
            _cell(_config_code(method_key, display_name)),
            _cell(METRICS_MD, "markdown"),
            _cell(METRICS_CODE),
            _cell(PROJECTED_BTP_MD, "markdown"),
            _cell(PROJECTED_BTP_CODE),
            _cell(PROJECTED_OOD_MD, "markdown"),
            _cell(PROJECTED_OOD_CODE),
        ]
        if method_key == "softtriple":
            cells.extend([_cell(SOFTTRIPLE_CENTERS_MD, "markdown"), _cell(SOFTTRIPLE_CENTERS_CODE)])
        cells.extend([_cell(RAW_TEST_MD, "markdown"), _cell(RAW_TEST_CODE)])
        cells.extend([_cell(BERTOPIC_MD, "markdown"), _cell(BERTOPIC_CODE)])
        out = NB / nb_name
        out.write_text(json.dumps(_nb(cells), indent=1), encoding="utf-8")
        print("Écrit", out)


if __name__ == "__main__":
    main()
