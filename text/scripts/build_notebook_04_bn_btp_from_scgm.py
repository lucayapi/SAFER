"""Génère notebooks/04_bayesian_network_btp_from_scgm.ipynb (exports SCGM du notebook 01 BTP)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "04_bayesian_network_btp_from_scgm.ipynb"


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str, *, tags: list[str] | None = None) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
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
# 04 bis — Réseaux bayésiens à partir des exports **SCGM** (BTP, notebook 01)

## 1 — Objectif

Ce notebook reprend la **même chaîne BN** que `04_malt_to_bayesian_network`, mais en entrée les **exports SCGM** produits par `01_scgm_text_btp_experiment` (`metadata_with_predictions.csv`, `prob_z_x.npy`, etc.). Une étape de copie vers un dossier **« type MALT »** (`malt_like_exports`) fournit les noms de fichiers attendus par `bn_malt` sans modifier le package.

Les variables binaires au niveau accident décrivent la **co-présence de motifs latents `z`** (SCGM) au-dessus d’un seuil de confiance ; le graphe est appris avec **pgmpy** (BIC, HillClimbing sous contraintes macro).

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
# Dossier `exports` du notebook 01 (SCGM BTP), après `export_scgm_text_outputs.py`.
SCGM_EXPORTS_DIR = "runs/scgm_text_qwen06_notebook/exports"
# Sorties du BN (et sous-dossier malt_like_exports pour les copies « type MALT »).
OUTPUT_DIR = "outputs/bn_btp_from_scgm"
CONFIDENCE_THRESHOLD = 0.50
MIN_TOPIC_ACCIDENT_SUPPORT = 20
MAX_TOPICS_PER_MACRO = 6
INCLUDE_MACRO_NODES = True
INCLUDE_SEVERITY = False
THEMES_OPENAI_CSV = ""
LEARN_UNCONSTRAINED_TOPIC = True
MAX_INDEGREE = 3
EQUIVALENT_SAMPLE_SIZE = 5
RANDOM_SEED = 42
WARN_MAX_BINARY_NODES = 30
""",
            tags=["parameters"],
        ),
        py(
            r"""
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


def _find_repo_root() -> Path:
    # Racine du dépôt (dossier contenant topic_eval/) sans importer topic_eval.
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())
    seen: set[Path] = set()
    for start in candidates:
        for p in [start, *start.parents]:
            if p in seen:
                continue
            seen.add(p)
            if (p / "topic_eval" / "__init__.py").is_file():
                return p
            if (p / "text" / "topic_eval" / "__init__.py").is_file():
                return p / "text"
    return Path.cwd().resolve()


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from topic_eval.paths import resolve_repo_path

OUT_ROOT = resolve_repo_path(OUTPUT_DIR, REPO)
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"
MODELS = OUT_ROOT / "models"
REPORTS = OUT_ROOT / "reports"

SCGM_EXPORTS_ROOT = resolve_repo_path(SCGM_EXPORTS_DIR, REPO)
MALT_LIKE = OUT_ROOT / "malt_like_exports"
MALT_LIKE.mkdir(parents=True, exist_ok=True)
_copy_map = (
    ("prob_z_x.npy", "pt_z_target.npy"),
    ("prob_y_x.npy", "pt_y_target.npy"),
    ("prob_y_z.npy", "pt_y_given_z.npy"),
    ("metadata_with_predictions.csv", "metadata_with_malt_predictions.csv"),
    ("z_assignments.csv", "z_assignments_target.csv"),
)
for _src_name, _dst_name in _copy_map:
    _src = SCGM_EXPORTS_ROOT / _src_name
    _dst = MALT_LIKE / _dst_name
    if not _src.is_file():
        raise FileNotFoundError(f"Export SCGM manquant pour le BN : {_src}")
    shutil.copy2(_src, _dst)
EXPORTS = MALT_LIKE

from bn_malt.utils import ensure_output_dirs, load_metadata_for_bn
from bn_malt.aggregate_malt_variables import create_accident_topic_matrix, export_aggregate_outputs
from bn_malt.bn_structure import (
    build_blacklist,
    export_edge_tables,
    learn_macro_constrained_structure,
    learn_unconstrained_structure,
    macro_chain_model,
)
from bn_malt.bn_learning import (
    drop_constant_columns,
    export_cpds_to_dir,
    fit_bn_parameters,
    save_bn_pickle,
    try_write_bif,
)
from bn_malt.bn_inference import conditional_prob_table, run_bn_queries
from bn_malt.bn_visualization import (
    build_topic_node_label_map,
    plot_adjacency_heatmap,
    plot_bn_graph,
    try_plotly_interactive,
)
from bn_malt.scenario_mining import export_scenarios, extract_typical_scenarios
from bn_malt.bn_diagnostics import compare_structure_rows, run_model_diagnostics
from bn_malt.reporting import write_bn_malt_report

try:
    import pgmpy  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "Le package « pgmpy » n’est pas installé pour l’interpréteur de ce noyau Jupyter.\n\n"
        f"  Interpréteur : {sys.executable}\n\n"
        "Installez-le (même environnement que le noyau), puis Kernel → Restart :\n\n"
        f"  {sys.executable} -m pip install \"pgmpy>=0.1.23,<1.0\" \"numpy<2\"\n\n"
        "ou, à la racine du dépôt :\n\n"
        "  pip install -r requirements.txt\n"
    ) from _e

ensure_output_dirs(OUT_ROOT)
np.random.seed(int(RANDOM_SEED))
warnings.filterwarnings("ignore", category=UserWarning)
print("REPO =", REPO)
print("SCGM_EXPORTS_ROOT =", SCGM_EXPORTS_ROOT)
print("EXPORTS (malt_like) =", EXPORTS)
print("OUT_ROOT =", OUT_ROOT)
"""
        ),
        md("## 3 — Chargement des métadonnées (format BN) et EDA rapide"),
        py(
            r"""
meta, exports_path = load_metadata_for_bn(str(EXPORTS), repo_root=REPO)
prob_y_z = np.load(exports_path / "pt_y_given_z.npy")

print(meta.shape)
display(meta.head(3))

_has_sev_panel = bool(INCLUDE_SEVERITY) and "pred_severity" in meta.columns
_nc = 2 if _has_sev_panel else 1
fig, axes = plt.subplots(1, _nc, figsize=(10 if _has_sev_panel else 6, 4))
if _nc == 1:
    axes = [axes]
if "z_confidence" in meta.columns:
    sns.histplot(meta["z_confidence"].astype(float), bins=40, ax=axes[0])
    axes[0].axvline(float(CONFIDENCE_THRESHOLD), color="red", ls="--", label="τ")
    axes[0].legend()
    axes[0].set_title("Confiance z (SCGM)")
if _has_sev_panel:
    meta["pred_severity"].astype(str).value_counts().head(8).plot.bar(ax=axes[1])
    axes[1].set_title("Gravité prédite (unités)")
plt.tight_layout()
p = FIGURES / "eda_malt_metadata.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print("Figure :", p)
"""
        ),
        md("## 4 — Agrégation accident × topics (variables binaires `Z_*` et `M_*`)"),
        py(
            r"""
acc_df, sel, map_df = create_accident_topic_matrix(
    meta,
    accident_id_col="accident_id",
    z_col="z_hat",
    z_conf_col="z_confidence",
    z_macro_col="z_dominant_macro",
    confidence_threshold=float(CONFIDENCE_THRESHOLD),
    min_topic_accident_support=int(MIN_TOPIC_ACCIDENT_SUPPORT),
    max_topics_per_macro=int(MAX_TOPICS_PER_MACRO),
    prob_y_z=prob_y_z,
    include_macro_aggregate_nodes=bool(INCLUDE_MACRO_NODES),
    include_severity=bool(INCLUDE_SEVERITY),
    severity_col="pred_severity",
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
topic_cols = [c for c in acc_df.columns if str(c).startswith("Z_")]
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
    p = FIGURES / "topic_correlation_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure :", p)
"""
        ),
        md(
            r"""
## 6 — Contraintes de structure (ordre macro)

On impose une **blacklist** d’arcs interdits : aucun parent ne peut être « après » son enfant dans l’ordre
**A0 → A1 → B → C** (les nœuds hors cette chaîne, s’il en existe, sont positionnés sans colonne dédiée supplémentaire).

Les fichiers `forbidden_edges.csv` et `allowed_edges.csv` documentent ces ensembles (indices + arcs appris).
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
    macro_edges_tpl = [("M_A0", "M_A1"), ("M_A1", "M_B"), ("M_B", "M_C")]
    macro_tpl = BayesianNetwork(macro_edges_tpl)

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
allowed_hint = sorted(set(topic_edges) | set(macro_edges_export))
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
        md("## 10 — Visualisation du graphe et heatmap d’adjacence"),
        py(
            r"""
from pathlib import Path

if str(THEMES_OPENAI_CSV).strip():
    _tp = Path(str(THEMES_OPENAI_CSV).strip()).expanduser()
    if not _tp.is_absolute():
        _tp = resolve_repo_path(str(_tp), REPO)
    _themes_path = _tp if _tp.is_file() else None
else:
    _themes_path = None
    for _c in (
        EXPORTS / "themes_by_z_openai_malt.csv",
        EXPORTS / "themes_by_z_openai.csv",
        EXPORTS / "themes_by_z_malt.csv",
        SCGM_EXPORTS_ROOT / "themes_by_z_openai_malt.csv",
        SCGM_EXPORTS_ROOT / "themes_by_z_openai.csv",
        SCGM_EXPORTS_ROOT / "themes_by_z_malt.csv",
        SCGM_EXPORTS_ROOT / "themes_by_z.csv",
    ):
        if _c.is_file():
            _themes_path = _c
            break

if _themes_path is not None:
    themes_df = pd.read_csv(_themes_path)
    print("Fichier libellés thèmes :", _themes_path)
else:
    themes_df = pd.DataFrame()
    print("Aucun CSV thèmes trouvé ; libellés de repli sur les codes Z_*.")

node_label_map = build_topic_node_label_map(list(topic_model.nodes()), themes_df, wrap_width=32)
plot_bn_graph(
    topic_model,
    topic_var_map_f,
    FIGURES / "bn_topic_constrained.png",
    title="Réseau bayésien — motifs (structure contrainte)",
    node_label_map=node_label_map,
)
plot_adjacency_heatmap(
    topic_model,
    list(topic_used),
    FIGURES / "bn_topic_adjacency.png",
    title="Adjacence — BN topics",
)
ok_html = try_plotly_interactive(
    topic_model,
    FIGURES / "bn_topic_interactive.html",
    node_label_map=node_label_map,
    title="Réseau bayésien — exploration interactive",
)
print("Plotly HTML :", ok_html)
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
        md("## 13 — Helpers d’affichage (résumé nœud, scénario)"),
        py(
            r"""
def display_bn_node_summary(model, node: str, max_lines: int = 40) -> None:
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
write_bn_malt_report(
    REPORTS / "bn_malt_summary.md",
    n_accidents=int(acc_df.shape[0]),
    n_topics_selected=int(len(sel)),
    params=params,
    diagnostics_df=diag_df,
    comparison_df=comp_df,
    figure_paths=[
        str(FIGURES / "eda_malt_metadata.png"),
        str(FIGURES / "bn_topic_constrained.png"),
        str(FIGURES / "bn_topic_adjacency.png"),
    ],
)
print("Rapport :", REPORTS / "bn_malt_summary.md")
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
