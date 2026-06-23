"""Génère notebooks/04_bayesian_network_macro_transfer.ipynb (BN sur corpus test, macro_transfer)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "04_bayesian_network_macro_transfer.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP
from macro_transfer.notebook_viz import (
    RAW_TEST_EMBEDDING_SECTION_MD,
    notebook_raw_test_embedding_source,
    notebook_topic_judge_section_md,
    notebook_topic_judge_source,
)


def md(text: str) -> dict:
    body = text.strip() + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [body]}


def py(text: str, *, tags: list[str] | None = None) -> dict:
    src = [text.strip() + "\n"]
    meta: dict = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "metadata": meta,
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


def main() -> None:
    cells = [
        md(
            r"""
# 04 — Réseau bayésien (corpus test)

## 1 — Objectif

Ce notebook construit un **réseau bayésien** à partir des exports FSP-SCGM sur le corpus test. Les variables binaires décrivent la **co-présence de topics par étape** de la chaîne accidentelle (`macro_topic_*`) au niveau accident.

**Prérequis** : `BASE_METHOD=scgm_text CORPUS=<id> bash jobs/run_frozen_source_prototypes.sh`

**Interprétation structurelle** : au niveau **topics**, A0→B direct est autorisé (contexte → événement sans A1 identifié). Au niveau **macro agrégé** (`M_*`), M_A0 ne pointe que vers M_A1. Paramètres : `BN_DISALLOW_A0_TO_B`, `BN_ENSURE_MACRO_CHAIN`.

**Sorties principales** : graphe statique + interactif, tableau des scénarios (chemins du BN, ≥ 2 étapes) avec graphes slide, export LaTeX `bn_network_slide.png`.
"""
        ),
        py(
            r"""
# --- Paramètres (papermill : `papermill ... -p KEY valeur`) ---
TEST_CORPUS = "metallurgie"  # configs/test_corpora.yaml
FSP_BASE_METHOD = "scgm_text"  # encodeur FSP (softtriple, supcon, batch_triplet, raw_embedding, …)
MACRO_TRANSFER_METHOD = f"frozen_source_prototypes/{FSP_BASE_METHOD}"
OUTPUT_DIR = ""  # vide → output_test/<TEST_CORPUS>/bn_results/
MACRO_CONF_THRESHOLD = 0.50
TOPIC_GAMMA_THRESHOLD = 0.50
MIN_TOPIC_ACCIDENT_SUPPORT = 20
MAX_TOPICS_PER_MACRO = 6
INCLUDE_MACRO_NODES = True
INCLUDE_SEVERITY = False
MAX_INDEGREE = 3
BN_DISALLOW_A0_TO_B = False  # False : A0→B direct autorisé (saut A1) ; True : A0 ne lie qu'à A1
BN_ENSURE_MACRO_CHAIN = True  # squelette topics A0→A1, A1→B, B→C ; agrégé M_A0→M_A1 seulement
EQUIVALENT_SAMPLE_SIZE = 5
RANDOM_SEED = 42
WARN_MAX_BINARY_NODES = 30
ENABLE_OPENAI_SCENARIOS = True
OPENAI_SCENARIO_MAX_ROWS = 12
SCENARIO_MIN_SUPPORT = 3
SCENARIO_TOP_N = 30
SCENARIO_MIN_MACROS = 2
SCENARIO_FULL_MIN_MACROS = 3  # export CSV liste complète (≥ N étapes macro)
SCENARIO_FULL_TOP_N = 500
SCENARIO_PATH_MAX_LEN = 8
SCENARIO_SLIDE_TOP_N = 3  # graphes slide affichés en section 5 (top scénarios)
SLIDE_SCENARIO_RANK = 0
SLIDE_FIG_WIDTH = 10.0
SLIDE_FIG_HEIGHT = 4.0
""",
            tags=["parameters"],
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + r"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np

if int(np.__version__.split(".", 1)[0]) >= 2:
    _py = sys.executable
    raise ImportError(
        "NumPy 2.x est chargé alors que le matplotlib de cet environnement (Anaconda) "
        "est souvent compilé pour NumPy 1.x — conflit à l'import de matplotlib.\n\n"
        f"Interpréteur du noyau : {_py}\n\n"
        f'  "{_py}" -m pip install "numpy<2" --force-reinstall'
    )

import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import HTML, Image, display as ipy_display

from safer_core.display_labels import rename_display_columns
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import bn_results_dir, macro_transfer_output_dir

REPO = TEXT_ROOT

_out = str(OUTPUT_DIR).strip()
OUT_ROOT = resolve_repo_path(_out, REPO) if _out else bn_results_dir(TEST_CORPUS, anchor=REPO)
# Rétrocompat : ancien dossier bn_staging → bn_results
if OUT_ROOT.name == "bn_staging":
    OUT_ROOT = OUT_ROOT.parent / "bn_results"
TABLES = OUT_ROOT / "tables"
FIGURES_STATIC = OUT_ROOT / "figures" / "static"
FIGURES_INTERACTIVE = OUT_ROOT / "figures" / "interactive"
MODELS = OUT_ROOT / "models"

MACRO_TRANSFER_ROOT = macro_transfer_output_dir(MACRO_TRANSFER_METHOD, TEST_CORPUS, anchor=REPO)
from bn_pipeline.staging_macro_transfer import stage_bn_exports_from_macro_transfer


def _reload_bn_pipeline_submodules() -> None:
    names = [n for n in sys.modules if n == "bn_pipeline" or n.startswith("bn_pipeline.")]
    for name in sorted(names, key=len, reverse=True):
        importlib.reload(sys.modules[name])


_reload_bn_pipeline_submodules()

from bn_pipeline.utils import ensure_output_dirs, load_metadata_for_bn
from bn_pipeline.aggregate_bn_variables import (
    create_accident_matrix_from_macro_transfer,
    export_aggregate_outputs,
)
from bn_pipeline.bn_structure import (
    build_blacklist,
    export_edge_tables,
    learn_macro_constrained_structure,
    macro_edges_for_export,
    standard_macro_edge_templates,
)
from bn_pipeline.bn_learning import (
    drop_constant_columns,
    export_cpds_to_dir,
    fit_bn_parameters,
    save_bn_pickle,
    try_write_bif,
)
from bn_pipeline.bn_visualization import (
    build_short_title_map,
    build_topic_node_label_map,
    extract_subgraph_for_slide,
    join_theme_summary_to_selected_variables,
    plot_bn_graph_cpd_boxes,
    plot_bn_scenario_slide,
    try_plotly_interactive,
)
from bn_pipeline.bn_paths import extract_bn_path_scenarios
from bn_pipeline.scenario_interpretation import (
    enrich_scenarios_table,
    export_scenario_interpretations,
)

try:
    import pgmpy  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "Le package « pgmpy » n'est pas installé pour l'interpréteur de ce noyau Jupyter.\n\n"
        f"  Interpréteur : {sys.executable}\n\n"
        f'  {sys.executable} -m pip install "pgmpy>=0.1.23,<1.0" "numpy<2"\n'
    ) from _e

BN_EXPORTS = stage_bn_exports_from_macro_transfer(
    MACRO_TRANSFER_METHOD,
    TEST_CORPUS,
    output_dir=OUT_ROOT,
    repo_root=REPO,
)
EXPORTS = BN_EXPORTS
SCGM_TOPICS = MACRO_TRANSFER_ROOT / "topics_bertopic"

ensure_output_dirs(OUT_ROOT)
np.random.seed(int(RANDOM_SEED))
warnings.filterwarnings("ignore", category=UserWarning)
print("OUT_ROOT =", OUT_ROOT)
print("MACRO_TRANSFER_ROOT =", MACRO_TRANSFER_ROOT)
"""
        ),
        md("## 2 — Données et agrégation accident × topics"),
        py(
            r"""
ensure_output_dirs(OUT_ROOT)

meta, exports_path = load_metadata_for_bn(str(EXPORTS), repo_root=REPO)
_assign_path = exports_path / "macro_topic_assignments.csv"
if not _assign_path.is_file():
    raise FileNotFoundError(f"Assignations manquantes : {_assign_path}")
assignments = pd.read_csv(_assign_path)

acc_df, sel, map_df = create_accident_matrix_from_macro_transfer(
    meta,
    assignments,
    accident_id_col="accident_id",
    macro_conf_col="q_conf",
    macro_conf_threshold=float(MACRO_CONF_THRESHOLD),
    topic_gamma_threshold=float(TOPIC_GAMMA_THRESHOLD),
    min_topic_accident_support=int(MIN_TOPIC_ACCIDENT_SUPPORT),
    max_topics_per_macro=int(MAX_TOPICS_PER_MACRO),
    include_macro_aggregate_nodes=bool(INCLUDE_MACRO_NODES),
    warn_max_binary_nodes=int(WARN_MAX_BINARY_NODES),
)
export_aggregate_outputs(acc_df, sel, map_df, TABLES)

topic_cols = [c for c in acc_df.columns if str(c).startswith("macro_topic_")]
print("Matrice accident × variables :", acc_df.shape)
print("Topics retenus :", len(topic_cols))
display(sel.head(10))
"""
        ),
        md(notebook_topic_judge_section_md()),
        py(
            notebook_topic_judge_source(
                "MACRO_TRANSFER_ROOT",
                "FIGURES_STATIC",
                restimate_var=None,
                topic_judge_cfg_var=None,
            )
        ),
        md("## 3 — Apprentissage du réseau bayésien"),
        py(
            r"""
from pgmpy.models import BayesianNetwork

macro_var_map = {f"M_{m}": m for m in ("A0", "A1", "B", "C")}
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    macro_var_map["Severity_high"] = "Severity"

macro_edges_sub = macro_edges_for_export(acc_df, include_severity=bool(INCLUDE_SEVERITY))
macro_node_list = sorted({n for u, v in macro_edges_sub for n in (u, v)})
macro_data = acc_df[macro_node_list].copy()
macro_data, macro_used = drop_constant_columns(macro_data, list(macro_data.columns))
macro_edges_sub = [(u, v) for (u, v) in macro_edges_sub if u in macro_used and v in macro_used]
macro_model = BayesianNetwork(macro_edges_sub)
macro_edges_export = list(macro_edges_sub)
macro_model = fit_bn_parameters(
    macro_model,
    macro_data,
    estimator="bayesian",
    equivalent_sample_size=int(EQUIVALENT_SAMPLE_SIZE),
)
save_bn_pickle(macro_model, MODELS / "bn_macro_chain.pkl")
export_cpds_to_dir(macro_model, TABLES / "cpds_macro", prefix="macro")
try_write_bif(macro_model, MODELS / "bn_macro_chain.bif")

topic_var_map = {str(r["variable"]): str(r["macro"]) for _, r in sel.iterrows()}
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    topic_var_map["Severity_high"] = "Severity"

topic_node_list = topic_cols.copy()
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    topic_node_list = topic_node_list + ["Severity_high"]

topic_data = acc_df[[c for c in topic_node_list if c in acc_df.columns]].copy()
topic_data, topic_used = drop_constant_columns(topic_data, list(topic_data.columns))
topic_var_map_f = {k: v for k, v in topic_var_map.items() if k in topic_used}

topic_model, topic_edges = learn_macro_constrained_structure(
    topic_data,
    topic_var_map_f,
    max_indegree=int(MAX_INDEGREE),
    disallow_a0_to_b_direct=bool(BN_DISALLOW_A0_TO_B),
    ensure_macro_chain_backbone=bool(BN_ENSURE_MACRO_CHAIN),
)
topic_model = fit_bn_parameters(
    topic_model,
    topic_data,
    estimator="bayesian",
    equivalent_sample_size=int(EQUIVALENT_SAMPLE_SIZE),
)
save_bn_pickle(topic_model, MODELS / "bn_topic_constrained.pkl")
export_cpds_to_dir(topic_model, TABLES / "cpds_topic", prefix="topic")
try_write_bif(topic_model, MODELS / "bn_topic_constrained.bif")

bl_topic = build_blacklist(
    list(topic_used),
    topic_var_map_f,
    disallow_a0_to_b_direct=bool(BN_DISALLOW_A0_TO_B),
)
allowed_hint = sorted(
    set(topic_edges)
    | set(macro_edges_export)
    | set(standard_macro_edge_templates(bool(INCLUDE_SEVERITY)))
)
export_edge_tables(macro_edges_export, topic_edges, bl_topic, allowed_hint, TABLES)

print(
    "BN topics —",
    topic_model.number_of_nodes(),
    "nœuds,",
    len(topic_edges),
    "arcs",
    f"(variables agrégées : {len(topic_used)}, dont sans arc dans le graphe affiché)",
)
print(
    "Structure : A0→B direct =",
    "non" if bool(BN_DISALLOW_A0_TO_B) else "oui",
    "| squelette chaîne A0→A1→B→C =",
    "oui" if bool(BN_ENSURE_MACRO_CHAIN) else "non",
)
"""
        ),
        md(
            """## 4 — Graphe du réseau bayésien

Sorties : `figures/static/bn_network.png` et `figures/interactive/bn_network.html`.
"""
        ),
        py(
            r"""
_themes_macro = exports_path / "themes_by_macro.csv"
if _themes_macro.is_file():
    themes_df = pd.read_csv(_themes_macro)
    if "theme_summary" not in themes_df.columns:
        if "theme_label" in themes_df.columns:
            themes_df["theme_summary"] = themes_df["theme_label"]
        else:
            themes_df["theme_summary"] = themes_df.get("top_words", "")
else:
    themes_df = pd.DataFrame()

sel = join_theme_summary_to_selected_variables(sel, themes_df)
sel.to_csv(TABLES / "selected_bn_variables.csv", index=False)

_nodes = list(topic_model.nodes())
node_label_map = build_topic_node_label_map(
    _nodes, themes_df, wrap_width=32, variable_macro_map=topic_var_map_f
)
short_title_map = build_short_title_map(_nodes, themes_df, topic_var_map_f)

_static_png = FIGURES_STATIC / "bn_network.png"
plot_bn_graph_cpd_boxes(
    topic_model,
    topic_var_map_f,
    _static_png,
    title="Réseau bayésien — topics par étape (chaîne accidentelle)",
    short_title_map=short_title_map,
    themes_df=themes_df,
)
print("Graphe statique :", _static_png)
ipy_display(Image(filename=str(_static_png)))

_interactive_html = FIGURES_INTERACTIVE / "bn_network.html"
_ok_plotly = try_plotly_interactive(
    topic_model,
    _interactive_html,
    node_label_map=node_label_map,
    short_title_map=short_title_map,
    variable_macro_map=topic_var_map_f,
    themes_df=themes_df,
    title="Réseau bayésien — exploration interactive (chaîne accidentelle)",
)
print("Graphe interactif :", _interactive_html, "| OK :", _ok_plotly)
if _ok_plotly and _interactive_html.is_file():
    ipy_display(HTML(_interactive_html.read_text(encoding="utf-8")))
"""
        ),
        md("## 5 — Scénarios récurrents (chemins du BN, ≥ 2 étapes)"),
        md(
            r"""
Tableau des chemins les plus fréquents + **graphes slide** (sous-graphe causal par scénario,
même style que l’export LaTeX `bn_network_slide.png`).
"""
        ),
        py(
            r"""
n_accidents_bn = int(acc_df["accident_id"].nunique()) if "accident_id" in acc_df.columns else len(acc_df)

freq_df, _path_diag = extract_bn_path_scenarios(
    acc_df,
    topic_model,
    topic_cols,
    topic_var_map_f,
    accident_id_col="accident_id",
    min_support=int(SCENARIO_MIN_SUPPORT),
    top_n=int(SCENARIO_TOP_N),
    metadata_unit=meta,
    text_col="sentence" if "sentence" in meta.columns else "accident_summary",
    min_macros=int(SCENARIO_MIN_MACROS),
    max_path_len=int(SCENARIO_PATH_MAX_LEN),
)
print(
    "Diagnostics scénarios BN :",
    f"chemins DAG={_path_diag.get('n_paths_dag', 0)}",
    f"| support≥1={_path_diag.get('n_paths_support_ge_1', 0)}",
    f"| min étapes={_path_diag.get('min_macros', SCENARIO_MIN_MACROS)}",
    f"| accidents ≥{SCENARIO_MIN_MACROS} étapes={_path_diag.get('n_accidents_min_macros_cooc', 0)}",
    f"| accidents 4 étapes={_path_diag.get('n_accidents_full_macro_cooc', 0)}",
    f"| seuil={_path_diag.get('min_support_applied', SCENARIO_MIN_SUPPORT)}",
)
if _path_diag.get("support_fallback"):
    print(
        "Note : seuil abaissé pour afficher des scénarios "
        f"(demandé={SCENARIO_MIN_SUPPORT}, appliqué={_path_diag.get('min_support_applied')})."
    )

if len(freq_df):
    scenario_df = enrich_scenarios_table(
        freq_df,
        n_accidents_bn,
        themes_df,
        enable_openai=bool(ENABLE_OPENAI_SCENARIOS),
        max_rows=int(OPENAI_SCENARIO_MAX_ROWS),
        cache_path=TABLES / "scenario_interpretations.csv",
    )
    export_scenario_interpretations(scenario_df, TABLES / "bn_path_scenarios.csv")
    _show_cols = [
        c
        for c in (
            "configuration_probable",
            "macro_path",
            "path_nodes",
            "prob",
            "support",
            "interpretation",
        )
        if c in scenario_df.columns
    ]
    display(rename_display_columns(scenario_df[_show_cols]))
    print("Export :", TABLES / "bn_path_scenarios.csv")

    _slide_figsize = (float(SLIDE_FIG_WIDTH), float(SLIDE_FIG_HEIGHT))
    _n_slides = min(max(1, int(SCENARIO_SLIDE_TOP_N)), len(scenario_df))
    print(f"Graphes slide — top {_n_slides} scénario(s) :")
    for _slide_i in range(_n_slides):
        _slide_row = scenario_df.iloc[_slide_i]
        _slide_png = FIGURES_STATIC / f"bn_scenario_slide_{_slide_i:02d}.png"
        _ok_slide = plot_bn_scenario_slide(
            topic_model,
            _slide_row,
            topic_var_map_f,
            _slide_png,
            short_title_map=short_title_map,
            themes_df=themes_df,
            rank=_slide_i,
            figsize=_slide_figsize,
            col_gap=1.8,
            row_gap=1.0,
            box_width=1.5,
            title_wrap_width=18,
        )
        if _ok_slide:
            print(" ", _slide_png)
            ipy_display(Image(filename=str(_slide_png)))
        else:
            print(f"  Scénario #{_slide_i + 1} : pas de path_nodes exploitable.")

    _rank_latex = max(0, min(int(SLIDE_SCENARIO_RANK), len(scenario_df) - 1))
    _latex_png = FIGURES_STATIC / "bn_network_slide.png"
    if plot_bn_scenario_slide(
        topic_model,
        scenario_df.iloc[_rank_latex],
        topic_var_map_f,
        _latex_png,
        short_title_map=short_title_map,
        themes_df=themes_df,
        rank=_rank_latex,
        title="Scénario typique — chemin causal",
        figsize=_slide_figsize,
        col_gap=1.8,
        row_gap=1.0,
        box_width=1.5,
        title_wrap_width=18,
    ):
        print("Export LaTeX :", _latex_png)

    freq_full, _path_diag_full = extract_bn_path_scenarios(
        acc_df,
        topic_model,
        topic_cols,
        topic_var_map_f,
        accident_id_col="accident_id",
        min_support=int(SCENARIO_MIN_SUPPORT),
        top_n=int(SCENARIO_FULL_TOP_N),
        metadata_unit=meta,
        text_col="sentence" if "sentence" in meta.columns else "accident_summary",
        min_macros=int(SCENARIO_FULL_MIN_MACROS),
        max_path_len=int(SCENARIO_PATH_MAX_LEN),
    )
    _full_csv = TABLES / f"bn_path_scenarios_min{int(SCENARIO_FULL_MIN_MACROS)}_macros.csv"
    if len(freq_full):
        scenario_full = enrich_scenarios_table(
            freq_full,
            n_accidents_bn,
            themes_df,
            enable_openai=False,
            max_rows=None,
        )
        export_scenario_interpretations(scenario_full, _full_csv)
        print(
            f"Export complet (≥ {int(SCENARIO_FULL_MIN_MACROS)} étapes macro) :",
            len(scenario_full),
            "lignes →",
            _full_csv,
        )
        display(rename_display_columns(scenario_full.head(25)))
    else:
        print(
            f"Aucun chemin BN avec ≥ {int(SCENARIO_FULL_MIN_MACROS)} étapes "
            f"et support ≥ {_path_diag_full.get('min_support_applied', SCENARIO_MIN_SUPPORT)}."
        )
else:
    print(
        "Aucun scénario trouvé. Ajuster SCENARIO_MIN_SUPPORT, SCENARIO_MIN_MACROS "
        "ou MAX_TOPICS_PER_MACRO (nombre de topics par étape)."
    )
"""
        ),
        md(RAW_TEST_EMBEDDING_SECTION_MD),
        py(notebook_raw_test_embedding_source("FIGURES_STATIC")),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Écrit :", NB_PATH)


if __name__ == "__main__":
    main()
