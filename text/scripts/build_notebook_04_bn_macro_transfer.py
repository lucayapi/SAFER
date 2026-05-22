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
    # Une seule entrée source : évite de couper les littéraux "...\n\n" du générateur.
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
# 04 — Réseaux bayésiens (corpus test, macro_transfer)

## 1 — Objectif

Ce notebook construit un **réseau bayésien** à partir des exports **`macro_transfer`** sur le corpus test (`output_test/<TEST_CORPUS>/macro_transfer/<MACRO_TRANSFER_METHOD>/`). Le staging copie métadonnées transférées, assignations intra-macro **BERTopic** et thèmes vers `output_test/<TEST_CORPUS>/bn_staging/`.

Les variables binaires au niveau accident décrivent la **co-présence de topics intra-macro** (`macro_topic_*`) ; le graphe est appris avec **pgmpy** (BIC, HillClimbing sous contraintes macro).

**Prérequis** : `CORPUS=<id> bash jobs/run_tpn_macro_transfer.sh` (sortie `macro_transfer/tpn_<encodeur>/`).

### Nuances d’interprétation

- **\(X_{i,k}=0\)** : le motif \(k\) n’est **pas** identifié dans le récit de l’accident \(i\) au-dessus du seuil — ce n’est **pas** une preuve d’absence physique du facteur.
- Les **arcs** du BN encodent des **dépendances conditionnelles** apprises (avec estimateur bayésien ou MLE), **pas** une causalité démontrée.
"""
        ),
        md(
            r"""
## 2 — Rappels formels (Markdown + LaTeX)

### Agrégation accident × motif

Pour un accident \(i\) et un topic \(k\) retenu après filtrage support :

\[
X_{i,k} = \mathbf{1}\left\{ \exists \text{ unité } j \text{ de } i : \hat{z}_j = k,\; \text{conf}(j) \ge \tau \right\}
\]

### Factorisation du BN

\[
P(\mathbf{X}) = \prod_{v \in \mathcal{V}} P\left(X_v \mid \mathrm{Pa}(v)\right)
\]

### BIC (structure)

\[
\mathrm{BIC} = \log \hat{L}(\mathcal{G}, \mathcal{D}) - \frac{d}{2}\log n
\]

### Lift binaire (parent → enfant)

\[
\mathrm{Lift}(Y\mid X) = \frac{P(Y{=}1 \mid X{=}1)}{P(Y{=}1)}
\]

### Sparsité (densité du graphe)

Proportion d’arcs présents parmi les couples ordonnés de nœuds (hors boucles).
"""
        ),
        py(
            r"""
# --- Paramètres (papermill : `papermill ... -p KEY valeur`) ---
TEST_CORPUS = "metallurgie"  # configs/test_corpora.yaml
MACRO_TRANSFER_METHOD = "tpn_scgm_text"  # tpn_softtriple | tpn_scgm_text | tpn_supcon | …
OUTPUT_DIR = ""  # vide → output_test/<TEST_CORPUS>/bn_staging/
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
""",
            tags=["parameters"],
        ),
        py(
            NOTEBOOK_PATH_SETUP
            + r"""
from __future__ import annotations

import shutil
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
        "Corrigez puis Kernel → Restart, par exemple :\n\n"
        f'  "{_py}" -m pip install "numpy<2" --force-reinstall\n\n'
        "ou :\n\n"
        '  conda install "numpy<2" "matplotlib" "scipy" -y\n\n'
        "Réinstallez aussi les deps du dépôt : pip install -r requirements.txt"
    )

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import bn_staging_dir, macro_transfer_output_dir

REPO = TEXT_ROOT

_out = str(OUTPUT_DIR).strip()
OUT_ROOT = resolve_repo_path(_out, REPO) if _out else bn_staging_dir(TEST_CORPUS, anchor=REPO)
TABLES = OUT_ROOT / "tables"
FIGURES_STATIC = OUT_ROOT / "figures" / "static"
FIGURES_INTERACTIVE = OUT_ROOT / "figures" / "interactive"
FIGURES_NODES = OUT_ROOT / "figures" / "nodes"
MODELS = OUT_ROOT / "models"
REPORTS = OUT_ROOT / "reports"

MACRO_TRANSFER_ROOT = macro_transfer_output_dir(MACRO_TRANSFER_METHOD, TEST_CORPUS, anchor=REPO)
from bn_pipeline.staging_macro_transfer import stage_bn_exports_from_macro_transfer

BN_EXPORTS = stage_bn_exports_from_macro_transfer(
    MACRO_TRANSFER_METHOD,
    TEST_CORPUS,
    output_dir=OUT_ROOT,
    repo_root=REPO,
)
EXPORTS = BN_EXPORTS
SCGM_TOPICS = MACRO_TRANSFER_ROOT / "topics_bertopic"

import importlib


def _reload_bn_pipeline_submodules() -> None:
    # Recharge bn_pipeline après édition du code sans redémarrer le noyau Jupyter.
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
    learn_unconstrained_structure,
    macro_chain_model,
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
    build_topic_node_label_map,
    display_node_card,
    export_node_cards_png,
    export_node_marginals_csv,
    join_theme_summary_to_selected_variables,
    plot_adjacency_heatmap,
    plot_bn_graph,
    plot_bn_graph_cpd_boxes,
    plot_macro_causality_schematic,
    try_plotly_interactive,
    try_pyvis_bn_graph,
)
from bn_pipeline.scenario_mining import export_scenarios, extract_typical_scenarios
from bn_pipeline.bn_diagnostics import compare_structure_rows, run_model_diagnostics
from bn_pipeline.reporting import write_bn_report

try:
    import pgmpy  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "Le package « pgmpy » n'est pas installé pour l'interpréteur de ce noyau Jupyter.\n\n"
        f"  Interpréteur : {sys.executable}\n\n"
        "Installez-le (même environnement que le noyau), puis Kernel → Restart :\n\n"
        f'  {sys.executable} -m pip install "pgmpy>=0.1.23,<1.0" "numpy<2"\n\n'
        "ou, à la racine du dépôt :\n\n"
        "  pip install -r requirements.txt\n"
    ) from _e

ensure_output_dirs(OUT_ROOT)
np.random.seed(int(RANDOM_SEED))
warnings.filterwarnings("ignore", category=UserWarning)
print("REPO =", REPO)
print("TEST_CORPUS =", TEST_CORPUS)
print("MACRO_TRANSFER_ROOT =", MACRO_TRANSFER_ROOT)
print("SCGM_TOPICS =", SCGM_TOPICS)
print("EXPORTS (bn_exports) =", EXPORTS)
print("OUT_ROOT =", OUT_ROOT)
"""
        ),
        md("## 3 — Chargement des métadonnées (format BN) et EDA rapide"),
        py(
            r"""
meta, exports_path = load_metadata_for_bn(str(EXPORTS), repo_root=REPO)
_pyz = exports_path / "pt_y_given_z.npy"
prob_y_z = np.load(_pyz) if _pyz.is_file() else None

print(meta.shape)
display(meta.head(3))

_has_sev_panel = bool(INCLUDE_SEVERITY) and "pred_severity" in meta.columns
_nc = 2 if _has_sev_panel else 1
fig, axes = plt.subplots(1, _nc, figsize=(10 if _has_sev_panel else 6, 4))
if _nc == 1:
    axes = [axes]
_conf_col = "q_conf" if "q_conf" in meta.columns else ("z_confidence" if "z_confidence" in meta.columns else None)
if _conf_col:
    sns.histplot(meta[_conf_col].astype(float), bins=40, ax=axes[0])
    axes[0].axvline(float(MACRO_CONF_THRESHOLD), color="red", ls="--", label="τ")
    axes[0].legend()
    axes[0].set_title(f"Confiance macro ({_conf_col})")
if _has_sev_panel:
    meta["pred_severity"].astype(str).value_counts().head(8).plot.bar(ax=axes[1])
    axes[1].set_title("Gravité prédite (unités)")
plt.tight_layout()
p = FIGURES_STATIC / "eda_bn_metadata.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print("Figure :", p)
"""
        ),
        md("## 4 — Agrégation accident × topics intra-macro (`macro_topic_*` et `M_*`)"),
        py(
            r"""
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
print(acc_df.shape)
display(sel.head(10))
"""
        ),
        md("## 5 — EDA de la matrice accident × variables retenues"),
        py(
            r"""
topic_cols = [c for c in acc_df.columns if str(c).startswith("macro_topic_")]
macro_cols = [c for c in acc_df.columns if str(c).startswith("M_")]
print("n topics:", len(topic_cols), "n macros:", len(macro_cols))

if topic_cols:
    M = acc_df[topic_cols].to_numpy(dtype=float)
    share = float(M.mean())
    print("Part moyenne de 1 (topics) :", round(share, 4))
    plt.figure(figsize=(10, 6))
    co = np.corrcoef(M.T)
    sns.heatmap(co, xticklabels=False, yticklabels=False, cmap="vlag", center=0)
    plt.title("Corrélations entre colonnes Z (aperçu)")
    p = FIGURES_STATIC / "topic_correlation_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure :", p)
"""
        ),
        md(
            r"""
## 6 — Contraintes de structure (DAG accidentologique)

Structure **imposée** entre macros (pas de saut de l’évolution) :

- **A0** → contexte (travail, activité, environnement, matériel)
- **A1** → facteurs contributifs (défaillances, conditions dangereuses)
- **B** → événement / déviation
- **C** → dommage, blessure, conséquence

**Arcs autorisés** : `A0→A1`, `A0→B`, `A1→B`, `B→C` (et éventuellement `C→gravité`).  
**Interdits** notamment : `A0→C`, `A1→C`, retours `C→B`, etc.

Les fichiers `forbidden_edges.csv` et `allowed_edges.csv` documentent ces ensembles.
"""
        ),
        md("## 7 — Apprentissage : BN **macro** (chaîne fixe des agrégats `M_*`)"),
        py(
            r"""
from pgmpy.models import BayesianNetwork

macro_var_map = {f"M_{m}": m for m in ("A0", "A1", "B", "C")}
if INCLUDE_SEVERITY and "Severity_high" in acc_df.columns:
    macro_var_map["Severity_high"] = "Severity"
    macro_tpl, macro_edges_tpl = macro_chain_model("Severity_high")
else:
    macro_tpl, macro_edges_tpl = macro_chain_model(severity_node=None)

macro_node_list = [n for n in macro_tpl.nodes() if n in acc_df.columns]
macro_data = acc_df[macro_node_list].copy()
macro_data, macro_used = drop_constant_columns(macro_data, list(macro_data.columns))
macro_edges_sub = [(u, v) for (u, v) in macro_tpl.edges() if u in macro_used and v in macro_used]
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
print("Macro BN — nœuds:", list(macro_model.nodes()))
"""
        ),
        md("## 8 — Apprentissage : BN **topics** sous contraintes (HillClimb + BIC)"),
        py(
            r"""
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
print("Topic BN (contraint) —", topic_model.number_of_nodes(), "nœuds,", len(topic_edges), "arcs")
"""
        ),
        md("## 9 — Diagnostics (`check_model`, DAG, degrés)"),
        py(
            r"""
diag_rows = [
    run_model_diagnostics(macro_model, "macro_chain"),
    run_model_diagnostics(topic_model, "topic_constrained"),
]
diag_df = pd.DataFrame(diag_rows)
display(diag_df)
"""
        ),
        md(
            """## 10 — Visualisation du graphe et heatmap d’adjacence

Figure article : **boîtes CPD séparées** (colonnes A0/A1 → B → C, barres `P(absent)` / `P(présent)`).
Schéma macro + graphe topologique léger + heatmap.
Sorties : `figures/static/bn_topic_constrained_cpd.png`, `figures/interactive/`, `figures/nodes/`, `tables/node_marginals.csv`.
"""
        ),
        py(
            r"""
from IPython.display import HTML, display as ipy_display

plot_macro_causality_schematic(
    FIGURES_STATIC / "macro_causality_schematic.png",
    include_severity=bool(INCLUDE_SEVERITY),
    title="Structure causale des macros (récits d'accidents)",
)
print("Schéma article :", FIGURES_STATIC / "macro_causality_schematic.png")

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
print("Libellés BN :", _themes_macro if _themes_macro.is_file() else "(aucun themes_by_macro)")

sel = join_theme_summary_to_selected_variables(sel, themes_df)
sel.to_csv(TABLES / "selected_bn_variables.csv", index=False)
_cols_show = [c for c in ("macro", "topic_id", "variable", "theme_summary") if c in sel.columns]
display(sel[_cols_show].head(10))

_nodes = list(topic_model.nodes())
node_label_map = build_topic_node_label_map(
    _nodes, themes_df, wrap_width=32, variable_macro_map=topic_var_map_f
)
short_title_map = build_short_title_map(_nodes, themes_df, topic_var_map_f)

export_node_marginals_csv(topic_model, short_title_map, TABLES / "node_marginals.csv")

plot_bn_graph_cpd_boxes(
    topic_model,
    topic_var_map_f,
    FIGURES_STATIC / "bn_topic_constrained_cpd.png",
    title="Réseau bayésien — motifs (CPD, structure contrainte)",
    short_title_map=short_title_map,
    themes_df=themes_df,
)
print("Figure article CPD :", FIGURES_STATIC / "bn_topic_constrained_cpd.png")

plot_bn_graph(
    topic_model,
    topic_var_map_f,
    FIGURES_STATIC / "bn_topic_constrained.png",
    title="Réseau bayésien — motifs (topologie)",
    short_title_map=short_title_map,
    themes_df=themes_df,
    show_cpd_cards=False,
)
plot_adjacency_heatmap(
    topic_model,
    list(topic_used),
    FIGURES_STATIC / "bn_topic_adjacency.png",
    title="Adjacence — BN topics",
    themes_df=themes_df,
    variable_macro_map=topic_var_map_f,
)
export_node_cards_png(topic_model, short_title_map, FIGURES_NODES)

_ok_plotly = try_plotly_interactive(
    topic_model,
    FIGURES_INTERACTIVE / "bn_topic_interactive.html",
    node_label_map=node_label_map,
    short_title_map=short_title_map,
    variable_macro_map=topic_var_map_f,
    themes_df=themes_df,
    title="Réseau bayésien — exploration interactive (Plotly)",
)
_ok_pyvis = try_pyvis_bn_graph(
    topic_model,
    FIGURES_INTERACTIVE / "bn_topic_pyvis.html",
    short_title_map=short_title_map,
    variable_macro_map=topic_var_map_f,
    themes_df=themes_df,
    title="Réseau bayésien — Pyvis",
)
print("Plotly HTML :", _ok_plotly, "| Pyvis HTML :", _ok_pyvis)

if _ok_plotly and (FIGURES_INTERACTIVE / "bn_topic_interactive.html").is_file():
    ipy_display(HTML((FIGURES_INTERACTIVE / "bn_topic_interactive.html").read_text(encoding="utf-8")))
if _ok_pyvis and (FIGURES_INTERACTIVE / "bn_topic_pyvis.html").is_file():
    ipy_display(HTML((FIGURES_INTERACTIVE / "bn_topic_pyvis.html").read_text(encoding="utf-8")))

# Exemple carte nœud (format barres)
if topic_used:
    _demo = str(topic_used[0])
    print(f"\n--- Carte exemple : {_demo} ---")
    display_node_card(topic_model, _demo, short_title_map)
"""
        ),
        md("## 11 — Inférence (VariableElimination) et lifts sur les arcs"),
        py(
            r"""
q_df, lift_df = run_bn_queries(topic_model)
q_df.to_csv(TABLES / "query_results.csv", index=False)
lift_df.to_csv(TABLES / "lift_results.csv", index=False)
display(lift_df.head(15))

# Exemple A1 → B sur variables macro agrégées si présentes
if "M_A1" in topic_model.nodes() and "M_B" in topic_model.nodes():
    cpt = conditional_prob_table(topic_model, "M_B", "M_A1")
    cpt.to_csv(TABLES / "conditional_M_B_given_M_A1.csv", index=False)
    display(cpt)
"""
        ),
        md(
            "## 12 — Configurations typiques de co-présence de motifs"
        ),
        py(
            r"""
_sev_col = None
if bool(INCLUDE_SEVERITY):
    if "Severity_high" in acc_df.columns:
        _sev_col = "Severity_high"
    elif "Severity_ord" in acc_df.columns:
        _sev_col = "Severity_ord"

freq_df, high_df = extract_typical_scenarios(
    acc_df,
    topic_model,
    topic_cols,
    accident_id_col="accident_id",
    severity_high_col=_sev_col,
    min_support=5,
    top_n=30,
    metadata_unit=meta,
    text_col="sentence" if "sentence" in meta.columns else "accident_summary",
)
export_scenarios(freq_df, high_df, TABLES)
display(freq_df.head(8))
if len(high_df):
    display(high_df.head(8))
else:
    print("Pas d’export « risque gravité » (mode sans colonne de gravité ou effectifs nuls).")
"""
        ),
        md("## 13 — Helpers d’affichage (carte CPD, scénario)"),
        py(
            r"""
def display_bn_node_summary(model, node: str, max_lines: int = 40) -> None:
    # Carte barres P(0)/P(1) si short_title_map existe, sinon CPD brut
    if "short_title_map" in globals():
        display_node_card(model, node, short_title_map)
        return
    for cpd in model.get_cpds():
        if cpd.variable == node:
            txt = str(cpd)
            lines = txt.splitlines()
            print("\n".join(lines[:max_lines]))
            if len(lines) > max_lines:
                print("…")
            return
    print("CPD introuvable pour", node)


def display_scenario(row: pd.Series) -> None:
    keys = [
        "scenario_id",
        "macro_path",
        "topics_present",
        "support",
        "representative_accidents",
        "representative_sentences",
    ]
    for k in keys:
        if k in row.index:
            print(f"{k}: {row[k]}")


if len(sel):
    display_bn_node_summary(topic_model, str(sel.iloc[0]["variable"]))
if len(freq_df):
    display_scenario(freq_df.iloc[0])
"""
        ),
        md(
            r"""
## 14 — Comparaison de structures (chaîne macro vs topic contraint vs topic sans contrainte)

Métriques : nombre d’arcs, violations de la blacklist macro (pour le BN non contraint), densité, nœuds isolés.
Le score BIC global pgmpy n’est pas toujours comparable entre structures différentes ; on privilégie ces indicateurs de complexité et de respect des contraintes.
"""
        ),
        py(
            r"""
topic_unc_model = None
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

comp_rows = []
comp_rows.append(compare_structure_rows("macro_chain", macro_model, macro_var_map))
comp_rows.append(compare_structure_rows("topic_constrained", topic_model, topic_var_map_f))
if topic_unc_model is not None:
    comp_rows.append(compare_structure_rows("topic_unconstrained", topic_unc_model, topic_var_map_f))
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(TABLES / "bn_structure_comparison.csv", index=False)
display(comp_df)
"""
        ),
        md("## 15 — Rapport Markdown synthétique"),
        py(
            r"""
params = {
    "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
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
        str(FIGURES_STATIC / "eda_bn_metadata.png"),
        str(FIGURES_STATIC / "bn_topic_constrained.png"),
        str(FIGURES_STATIC / "bn_topic_adjacency.png"),
        str(FIGURES_INTERACTIVE / "bn_topic_interactive.html"),
        str(FIGURES_INTERACTIVE / "bn_topic_pyvis.html"),
    ],
)
print("Rapport :", REPORTS / "bn_summary.md")
"""
        ),
        md("## 16 — Export LaTeX (tableaux diagnostics / comparaison)"),
        py(
            r"""
tex_diag = REPORTS / "bn_model_diagnostics.tex"
tex_comp = REPORTS / "bn_structure_comparison.tex"
diag_df.to_latex(tex_diag, index=False)
comp_df.to_latex(tex_comp, index=False)
print(tex_diag, tex_comp)
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
