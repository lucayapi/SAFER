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
# 04 — Réseau bayésien (corpus test, macro_transfer)

## 1 — Objectif

Ce notebook construit un **réseau bayésien** à partir des exports **macro_transfer** (FSP-SCGM par défaut) sur le corpus test. Les variables binaires décrivent la **co-présence de topics intra-macro** (`macro_topic_*`) au niveau accident.

**Prérequis** : `CORPUS=<id> bash jobs/run_frozen_proto_scgm.sh`

**Interprétation** : \(X_{i,k}=0\) signifie que le motif \(k\) n’est pas identifié dans le récit (pas une preuve d’absence physique). Les **arcs** encodent des dépendances conditionnelles apprises sous contraintes macro (A0→A1→B→C), pas une causalité démontrée.

**Sorties principales** : graphe statique + interactif, tableau des scénarios récurrents (A0 → … → C) avec probabilités.
"""
        ),
        py(
            r"""
# --- Paramètres (papermill : `papermill ... -p KEY valeur`) ---
TEST_CORPUS = "metallurgie"  # configs/test_corpora.yaml
MACRO_TRANSFER_METHOD = "frozen_source_prototypes/scgm"  # défaut BN basé FSP-SCGM
OUTPUT_DIR = ""  # vide → output_test/<TEST_CORPUS>/bn_results/
MACRO_CONF_THRESHOLD = 0.50
TOPIC_GAMMA_THRESHOLD = 0.50
MIN_TOPIC_ACCIDENT_SUPPORT = 20
MAX_TOPICS_PER_MACRO = 6
INCLUDE_MACRO_NODES = True
INCLUDE_SEVERITY = False
MAX_INDEGREE = 3
EQUIVALENT_SAMPLE_SIZE = 5
RANDOM_SEED = 42
WARN_MAX_BINARY_NODES = 30
ENABLE_OPENAI_SCENARIOS = True
OPENAI_SCENARIO_MAX_ROWS = 12
SCENARIO_MIN_SUPPORT = 5
SCENARIO_TOP_N = 30
SCENARIO_REQUIRE_FULL_MACRO_PATH = True
SCENARIO_EXCLUDE_EMPTY = True
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
    join_theme_summary_to_selected_variables,
    plot_bn_graph_cpd_boxes,
    try_plotly_interactive,
)
from bn_pipeline.scenario_mining import extract_typical_scenarios
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

bl_topic = build_blacklist(list(topic_used), topic_var_map_f)
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
    title="Réseau bayésien — topics intra-macro",
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
    title="Réseau bayésien — exploration interactive",
)
print("Graphe interactif :", _interactive_html, "| OK :", _ok_plotly)
if _ok_plotly and _interactive_html.is_file():
    ipy_display(HTML(_interactive_html.read_text(encoding="utf-8")))
"""
        ),
        md("## 5 — Scénarios récurrents (chemins complets A0 → A1 → B → C)"),
        py(
            r"""
_sev_col = None
if bool(INCLUDE_SEVERITY):
    if "Severity_high" in acc_df.columns:
        _sev_col = "Severity_high"
    elif "Severity_ord" in acc_df.columns:
        _sev_col = "Severity_ord"

freq_df, _high_df = extract_typical_scenarios(
    acc_df,
    topic_model,
    topic_cols,
    accident_id_col="accident_id",
    severity_high_col=_sev_col,
    min_support=int(SCENARIO_MIN_SUPPORT),
    top_n=int(SCENARIO_TOP_N),
    metadata_unit=meta,
    text_col="sentence" if "sentence" in meta.columns else "accident_summary",
    exclude_empty=bool(SCENARIO_EXCLUDE_EMPTY),
    require_full_macro_path=bool(SCENARIO_REQUIRE_FULL_MACRO_PATH),
)

n_accidents_bn = int(acc_df["accident_id"].nunique()) if "accident_id" in acc_df.columns else len(acc_df)

if len(freq_df):
    scenario_df = enrich_scenarios_table(
        freq_df,
        n_accidents_bn,
        themes_df,
        enable_openai=bool(ENABLE_OPENAI_SCENARIOS),
        max_rows=int(OPENAI_SCENARIO_MAX_ROWS),
        cache_path=TABLES / "scenario_interpretations.csv",
    )
    export_scenario_interpretations(scenario_df, TABLES / "recurring_scenarios.csv")
    _show_cols = [
        c
        for c in (
            "configuration_probable",
            "macro_path",
            "prob",
            "support",
            "interpretation",
        )
        if c in scenario_df.columns
    ]
    display(scenario_df[_show_cols])
    print("Export :", TABLES / "recurring_scenarios.csv")
else:
    print("Aucun scénario récurrent (augmenter le support ou le nombre de topics retenus).")
"""
        ),
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
