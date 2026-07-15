"""Génère notebooks/06_bn_macro_constrained.ipynb — BN macro-contraint depuis BERTopic notebook."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "06_bn_macro_constrained.ipynb"

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safer_core.notebook_bootstrap import NOTEBOOK_PATH_SETUP


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str, cell_id: str | None = None) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    cell: dict = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


TITLE_MD = """# 06 — Réseau bayésien macro-contraint (topics BERTopic)

Apprentissage d'un BN sur variables binaires **accident × topic intra-macro**, avec contraintes d'ordre **A0 → A1 → B → C** (aucun arc entre classes macro différentes).

**Entrée** : sorties BERTopic des notebooks **05_view** / **07** / **08** sous `{RESULTS_DIR}/bertopic_notebook/{corpus}/`.

**Staging** : `bn_pipeline.staging_macro_transfer.stage_bn_exports_from_bertopic_run` copie les artefacts vers `output_test/<corpus>/bn_results/<method>/staging/bn_exports/`.
"""

PARAMS_CODE = """# --- Paramètres (modifier ici) ---
METHOD = "batch_triplet"
# batch_triplet | softtriple | supcon | supervised_macro_ft | supervised_macro_baseline

TEST_CORPUS = "metallurgie"

# Dossier résultats de la méthode (contient bertopic_notebook/<corpus>/)
RESULTS_DIR = ROOT / "output/batch_triplet"
# Exemples :
# RESULTS_DIR = ROOT / "output/supervised_macro_ft"
# RESULTS_DIR = ROOT / "output_test/metallurgie/supervised_baseline"

# Chemin direct vers le run BERTopic (vide = RESULTS_DIR / "bertopic_notebook" / TEST_CORPUS)
BERTOPIC_RUN_DIR = ""

# Sorties BN (vide = output_test/<corpus>/bn_results/<method>/)
OUTPUT_DIR = ""

MACRO_CONF_THRESHOLD = 0.50
TOPIC_GAMMA_THRESHOLD = 0.50
MIN_TOPIC_ACCIDENT_SUPPORT = 20
MAX_TOPICS_PER_MACRO = 6
INCLUDE_MACRO_NODES = True
INCLUDE_SEVERITY = False
LEARN_UNCONSTRAINED_TOPIC = True
MAX_INDEGREE = 3
EQUIVALENT_SAMPLE_SIZE = 5
RANDOM_SEED = 42
WARN_MAX_BINARY_NODES = 30
"""

SETUP_CODE = """import importlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if int(np.__version__.split(".", 1)[0]) >= 2:
    raise ImportError(
        "NumPy 2.x incompatible avec matplotlib/pgmpy de cet environnement. "
        f"Interpréteur : {sys.executable} — installez numpy<2 puis redémarrez le noyau."
    )

from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import bn_results_dir

from bn_pipeline.staging_macro_transfer import stage_bn_exports_from_bertopic_run
from bn_pipeline.utils import ensure_output_dirs, load_metadata_for_bn
from bn_pipeline.aggregate_bn_variables import (
    create_accident_matrix_from_macro_transfer,
    export_aggregate_outputs,
)
from bn_pipeline.bn_structure import (
    build_blacklist,
    export_edge_tables,
    learn_macro_constrained_structure,
    learn_unconstrained_structure,
    macro_chain_model,
    standard_macro_edge_templates,
)
from bn_pipeline.bn_learning import (
    drop_constant_columns,
    export_cpds_to_dir,
    fit_bn_parameters,
    save_bn_pickle,
    try_write_bif,
)
from bn_pipeline.bn_inference import conditional_prob_table, run_bn_queries
from bn_pipeline.bn_visualization import (
    build_short_title_map,
    join_theme_summary_to_selected_variables,
    plot_adjacency_heatmap,
    plot_bn_graph,
    try_plotly_interactive,
    try_pyvis_bn_graph,
)
from bn_pipeline.bn_diagnostics import compare_structure_rows, run_model_diagnostics
from bn_pipeline.scenario_mining import export_scenarios, extract_typical_scenarios
from bn_pipeline.reporting import write_bn_report

try:
    import pgmpy  # noqa: F401
except ImportError as _e:
    raise ImportError(
        f"pgmpy manquant pour {sys.executable}. pip install 'pgmpy>=0.1.23,<1.0' 'numpy<2'"
    ) from _e


def _reload_bn_pipeline() -> None:
    names = [n for n in sys.modules if n == "bn_pipeline" or n.startswith("bn_pipeline.")]
    for name in sorted(names, key=len, reverse=True):
        importlib.reload(sys.modules[name])


_reload_bn_pipeline()

_out = str(OUTPUT_DIR).strip()
if _out:
    OUT_ROOT = resolve_repo_path(_out, ROOT)
else:
    OUT_ROOT = bn_results_dir(TEST_CORPUS, anchor=ROOT) / METHOD

TABLES = OUT_ROOT / "tables"
FIGURES_STATIC = OUT_ROOT / "figures" / "static"
FIGURES_INTERACTIVE = OUT_ROOT / "figures" / "interactive"
MODELS = OUT_ROOT / "models"
REPORTS = OUT_ROOT / "reports"

_bertopic = str(BERTOPIC_RUN_DIR).strip()
if _bertopic:
    BERTOPIC_ROOT = resolve_repo_path(_bertopic, ROOT)
else:
    BERTOPIC_ROOT = resolve_repo_path(str(RESULTS_DIR), ROOT) / "bertopic_notebook" / TEST_CORPUS

BN_EXPORTS = stage_bn_exports_from_bertopic_run(
    BERTOPIC_ROOT,
    METHOD,
    corpus_id=TEST_CORPUS,
    output_dir=OUT_ROOT,
    repo_root=ROOT,
)
EXPORTS = BN_EXPORTS
ASSIGNMENTS_CSV = EXPORTS / "macro_topic_assignments.csv"
THEMES_CSV = EXPORTS / "themes_by_macro.csv"

ensure_output_dirs(OUT_ROOT)
np.random.seed(int(RANDOM_SEED))
warnings.filterwarnings("ignore", category=UserWarning)

print("METHOD =", METHOD)
print("TEST_CORPUS =", TEST_CORPUS)
print("BERTOPIC_ROOT =", BERTOPIC_ROOT)
print("BN_EXPORTS =", EXPORTS)
print("OUT_ROOT =", OUT_ROOT)
"""

LOAD_MD = """## 1. Chargement métadonnées + assignments BERTopic
"""

LOAD_CODE = """meta, exports_path = load_metadata_for_bn(str(EXPORTS), repo_root=ROOT)
assignments = pd.read_csv(ASSIGNMENTS_CSV)
themes_df = pd.read_csv(THEMES_CSV) if THEMES_CSV.is_file() else pd.DataFrame()

print("metadata :", meta.shape)
print("assignments :", assignments.shape)
display(meta.head(3))
display(assignments.head(3))
"""

AGG_MD = """## 2. Agrégation accident × topics intra-macro

Variables binaires `macro_topic_<macro>_<id>` + agrégats `M_A0` … `M_C`.
"""

AGG_CODE = """acc_df, sel, map_df = create_accident_matrix_from_macro_transfer(
    meta,
    assignments,
    accident_id_col="accident_id",
    macro_conf_threshold=float(MACRO_CONF_THRESHOLD),
    topic_gamma_threshold=float(TOPIC_GAMMA_THRESHOLD),
    min_topic_accident_support=int(MIN_TOPIC_ACCIDENT_SUPPORT),
    max_topics_per_macro=int(MAX_TOPICS_PER_MACRO),
    include_macro_aggregate_nodes=bool(INCLUDE_MACRO_NODES),
    warn_max_binary_nodes=int(WARN_MAX_BINARY_NODES),
)
export_aggregate_outputs(acc_df, sel, map_df, TABLES)
print(acc_df.shape)
display(sel.head(12))
"""

STRUCTURE_MD = """## 3. Apprentissage structure — BN topics macro-contraint

Blacklist : pas d'arcs entre nœuds de macros différentes (ordre A0 → A1 → B → C).
"""

STRUCTURE_CODE = """from pgmpy.models import BayesianNetwork

topic_cols = [c for c in acc_df.columns if str(c).startswith("macro_topic_")]
topic_var_map = {str(r["variable"]): str(r["macro"]) for _, r in sel.iterrows()}
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    topic_var_map["Severity_high"] = "Severity"

topic_node_list = topic_cols.copy()
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    topic_node_list.append("Severity_high")

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
try_write_bif(topic_model, MODELS / "bn_topic_constrained.bif")
export_cpds_to_dir(topic_model, MODELS / "cpds_constrained")

blacklist = build_blacklist(list(topic_used), topic_var_map_f)
allowed_hint = standard_macro_edge_templates(
    include_severity=bool(INCLUDE_SEVERITY and "Severity_high" in acc_df.columns)
)
export_edge_tables([], topic_edges, blacklist, allowed_hint, TABLES)

# BN macro agrégé (chaîne fixe M_A0 → M_A1 → M_B → M_C)
macro_var_map = {f"M_{m}": m for m in ("A0", "A1", "B", "C")}
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    macro_var_map["Severity_high"] = "Severity"
    macro_tpl, _ = macro_chain_model("Severity_high")
else:
    macro_tpl, _ = macro_chain_model(severity_node=None)

macro_node_list = [n for n in macro_tpl.nodes() if n in acc_df.columns]
macro_data = acc_df[macro_node_list].copy()
macro_data, macro_used = drop_constant_columns(macro_data, list(macro_data.columns))
macro_edges_sub = [(u, v) for (u, v) in macro_tpl.edges() if u in macro_used and v in macro_used]
macro_model = BayesianNetwork(macro_edges_sub)
macro_model = fit_bn_parameters(
    macro_model,
    macro_data,
    estimator="bayesian",
    equivalent_sample_size=int(EQUIVALENT_SAMPLE_SIZE),
)
save_bn_pickle(macro_model, MODELS / "bn_macro_chain.pkl")

print("Topics retenus :", len(topic_used))
print("Arcs topic contraint :", len(topic_edges))
"""

VIZ_MD = """## 4. Visualisation graphe + heatmap adjacence
"""

VIZ_CODE = """sel = join_theme_summary_to_selected_variables(sel, themes_df)
sel.to_csv(TABLES / "selected_bn_variables_with_themes.csv", index=False)
short_title_map = build_short_title_map(topic_used, themes_df, topic_var_map_f)
globals()["short_title_map"] = short_title_map

static_png = FIGURES_STATIC / "bn_topic_constrained.png"
plot_bn_graph(
    topic_model,
    topic_var_map_f,
    static_png,
    short_title_map=short_title_map,
    themes_df=themes_df,
)
plot_adjacency_heatmap(
    topic_model,
    list(topic_used),
    FIGURES_STATIC / "bn_topic_adjacency.png",
    variable_macro_map=topic_var_map_f,
    themes_df=themes_df,
)
try_plotly_interactive(
    topic_model,
    FIGURES_INTERACTIVE / "bn_topic_interactive.html",
    short_title_map=short_title_map,
    variable_macro_map=topic_var_map_f,
    themes_df=themes_df,
)
try_pyvis_bn_graph(
    topic_model,
    FIGURES_INTERACTIVE / "bn_topic_pyvis.html",
    short_title_map=short_title_map,
    variable_macro_map=topic_var_map_f,
    themes_df=themes_df,
)
print("Figures :", static_png)
"""

INFER_MD = """## 5. Inférence et scénarios typiques
"""

INFER_CODE = """q_df, lift_df = run_bn_queries(topic_model)
q_df.to_csv(TABLES / "query_results.csv", index=False)
lift_df.to_csv(TABLES / "lift_results.csv", index=False)
display(lift_df.head(15))

_sev_col = "Severity_high" if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns else None
freq_df, high_df = extract_typical_scenarios(
    acc_df,
    topic_model,
    list(topic_used),
    accident_id_col="accident_id",
    severity_high_col=_sev_col,
    min_support=5,
    top_n=30,
    metadata_unit=meta,
    text_col="sentence" if "sentence" in meta.columns else None,
)
export_scenarios(freq_df, high_df, TABLES)
display(freq_df.head(8))
"""

DIAG_MD = """## 6. Diagnostics et comparaison (contraint vs non contraint)
"""

DIAG_CODE = """topic_unc_model = None
topic_unc_edges: list = []
if bool(LEARN_UNCONSTRAINED_TOPIC):
    topic_unc_model, topic_unc_edges = learn_unconstrained_structure(
        topic_data,
        list(topic_used),
        max_indegree=int(MAX_INDEGREE),
    )
    topic_unc_model = fit_bn_parameters(
        topic_unc_model,
        topic_data,
        estimator="bayesian",
        equivalent_sample_size=int(EQUIVALENT_SAMPLE_SIZE),
    )
    save_bn_pickle(topic_unc_model, MODELS / "bn_topic_unconstrained.pkl")

diag_rows = [
    run_model_diagnostics(macro_model, "macro_chain"),
    run_model_diagnostics(topic_model, "topic_constrained"),
]
if topic_unc_model is not None:
    diag_rows.append(run_model_diagnostics(topic_unc_model, "topic_unconstrained"))
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(TABLES / "bn_model_diagnostics.csv", index=False)
display(diag_df)

comp_rows = [
    compare_structure_rows("topic_constrained", topic_model, topic_var_map_f),
]
if topic_unc_model is not None:
    comp_rows.append(compare_structure_rows("topic_unconstrained", topic_unc_model, topic_var_map_f))
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(TABLES / "bn_structure_comparison.csv", index=False)
display(comp_df)

params = {
    "MACRO_CONF_THRESHOLD": MACRO_CONF_THRESHOLD,
    "TOPIC_GAMMA_THRESHOLD": TOPIC_GAMMA_THRESHOLD,
    "MIN_TOPIC_ACCIDENT_SUPPORT": MIN_TOPIC_ACCIDENT_SUPPORT,
    "MAX_TOPICS_PER_MACRO": MAX_TOPICS_PER_MACRO,
}
write_bn_report(
    REPORTS / "bn_summary.md",
    n_accidents=int(acc_df.shape[0]),
    n_topics_selected=int(len(sel)),
    params=params,
    diagnostics_df=diag_df,
    comparison_df=comp_df,
    figure_paths=[
        str(FIGURES_STATIC / "bn_topic_constrained.png"),
        str(FIGURES_STATIC / "bn_topic_adjacency.png"),
        str(FIGURES_INTERACTIVE / "bn_topic_interactive.html"),
    ],
)
print("Rapport :", REPORTS / "bn_summary.md")
"""

FOOTER_MD = """---

**Prérequis** : exécuter la section BERTopic du notebook de vue (05 / 07 / 08) avec `export_for_bn: true`.

**Méthodes supportées** : `batch_triplet`, `softtriple`, `supcon`, `supervised_macro_ft`, `supervised_macro_baseline`.
"""


def build_notebook() -> dict:
    cells = [
        md(TITLE_MD),
        py(NOTEBOOK_PATH_SETUP, cell_id="bootstrap"),
        py(PARAMS_CODE, cell_id="params"),
        py(SETUP_CODE, cell_id="setup"),
        md(LOAD_MD),
        py(LOAD_CODE, cell_id="load"),
        md(AGG_MD),
        py(AGG_CODE, cell_id="aggregate"),
        md(STRUCTURE_MD),
        py(STRUCTURE_CODE, cell_id="structure"),
        md(VIZ_MD),
        py(VIZ_CODE, cell_id="viz"),
        md(INFER_MD),
        py(INFER_CODE, cell_id="infer"),
        md(DIAG_MD),
        py(DIAG_CODE, cell_id="diag"),
        md(FOOTER_MD),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Écrit : {NB_PATH}")


if __name__ == "__main__":
    main()
