"""Generate primary scenario-mining notebooks (global BN + empirical scenarios)."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


def build_notebook(dataset_id: str) -> dict:
    cells = [
        markdown(f"""
# Scénarios récurrents — {dataset_id}

Pipeline principal : matrice accident × facteurs figée → BN global sans Z → mining empirique.
La récurrence est définie par le **count empirique** et les **closed patterns**.
Confidence, lift et support BN sont **descriptifs** uniquement.
Les arcs BN sont des dépendances conditionnelles, pas des effets causaux.
        """),
        code("""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image

SCENARIO_DIR = None
for base_dir in (Path.cwd(), *Path.cwd().parents):
    for candidate in (base_dir / "text" / "recurrent_scenarios", base_dir):
        if (candidate / "scenario_pipeline.py").is_file():
            SCENARIO_DIR = candidate
            break
    if SCENARIO_DIR is not None:
        break
if SCENARIO_DIR is None:
    raise FileNotFoundError("Impossible de trouver scenario_pipeline.py")
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import load_bn_analysis_config, load_selected_configurations
from scenario_analysis import run_global_bn_scenario_mining, _output_paths
        """),
        code(f"""
DATASET_ID = {dataset_id!r}
DISCOVERY_RUN_NAME = "theme_discovery_audit"
RUN_DIR = SCENARIO_DIR / "runs" / DISCOVERY_RUN_NAME / DATASET_ID
CONFIG_PATH = SCENARIO_DIR / "config.yaml"
BN_PARTITION_SELECTIONS = load_selected_configurations(RUN_DIR)
config = load_bn_analysis_config(CONFIG_PATH, DATASET_ID, RUN_DIR)
print("analysis_mode:", config.get("analysis_mode", "global_bn_scenario_mining"))
print("Partitions:", BN_PARTITION_SELECTIONS)
        """),
        markdown("## 1. Accident-level representation"),
        code("""
from scenario_pipeline import build_frozen_bn_inputs, load_units, write_bn_accident_inclusion_audit
from matrix_reporting import export_matrix_artifacts

paths = _output_paths(RUN_DIR)
BN_RESULTS = paths["root"]
MATRIX_DIR = paths["matrix"]
units, _ = load_units(config)
matrix, theme_dictionary, excluded_themes, roles = build_frozen_bn_inputs(
    units, RUN_DIR, BN_PARTITION_SELECTIONS, config, MATRIX_DIR,
)
audit, audit_summary = write_bn_accident_inclusion_audit(units, matrix, roles, MATRIX_DIR)
export_matrix_artifacts(matrix, roles, theme_dictionary, audit, MATRIX_DIR)
print("Dimensions:", matrix.shape)
role_counts = {role: sum(1 for node in roles if roles[node] == role) for role in ("A0", "A1", "B", "C")}
print("Role counts:", role_counts)
display(pd.read_csv(MATRIX_DIR / "input_summary.csv"))
display(pd.read_csv(MATRIX_DIR / "role_factor_summary.csv"))
        """),
        markdown("### Architecture conceptuelle (methods)"),
        code("""
path = MATRIX_DIR / "figures" / "conceptual_role_architecture.png"
if path.is_file():
    display(Image(filename=str(path)))
        """),
        markdown("## 2. Global role-constrained BN"),
        code("""
analysis = run_global_bn_scenario_mining(
    config, RUN_DIR, BN_PARTITION_SELECTIONS,
    units=units, matrix=matrix, theme_dictionary=theme_dictionary, roles=roles,
)
result = analysis["result"]
mining = analysis["mining"]
edges = analysis["edges"]
paths = analysis["paths"]
NETWORK_DIR = paths["network"]
FIGURES_DIR = paths["figures"]
display(pd.read_csv(NETWORK_DIR / "global_bn_summary.csv"))
display(pd.read_csv(NETWORK_DIR / "global_bn_edges.csv").head(15))
        """),
        markdown("## 3. Bootstrap stability"),
        code("""
bootstrap = analysis["bootstrap"]
threshold = config["bayesian_networks"]["bn_display_bootstrap_threshold"]
print(f"Stable edges >= {threshold}:", int((bootstrap["selection_frequency"] >= threshold).sum()) if not bootstrap.empty else 0)
display(bootstrap.sort_values("selection_frequency", ascending=False).head(15))
fig = FIGURES_DIR / "global_bn_stable_dependencies.png"
if fig.is_file():
    display(Image(filename=str(fig)))
        """),
        markdown("## 4. Admissible role-complete configurations"),
        code("""
candidates = mining["candidates_all"]
print("Admissible combinations:", mining["n_admissible"])
print("Candidate rows:", len(candidates))
        """),
        markdown("## 5. Empirically observed configurations"),
        code("""
print("Observed at least once:", mining["n_observed"])
display(candidates[candidates["scenario_accident_count"] > 0].sort_values(
    "scenario_accident_count", ascending=False
).head(20))
        """),
        markdown("## 6. Recurrence threshold and sensitivity"),
        code("""
min_count = config["scenario_mining"]["scenario_min_accident_count"]
print(f"Primary threshold (this application): n >= {min_count}")
display(pd.read_csv(paths["scenarios"] / "scenario_threshold_summary.csv"))
        """),
        markdown("## 7. Closed-pattern redundancy reduction"),
        code("""
recurrent_all = mining["recurrent_all"]
print(f"Closed recurrent patterns at n>={min_count}:", len(recurrent_all))
display(recurrent_all.head(20))
fig = FIGURES_DIR / "recurrent_scenario_reduction.png"
if fig.is_file():
    display(Image(filename=str(fig)))
        """),
        markdown("## 8. Recurrent scenario characterization (descriptive)"),
        code("""
display(recurrent_all[[
    "scenario_id", "upstream_labels", "B_label", "C_label",
    "scenario_accident_count", "scenario_support", "confidence", "lift",
    "positive_bn_path_support", "stable_positive_bn_path_support"
]].head(20))
fig = FIGURES_DIR / "recurrent_scenarios_support_lift.png"
if fig.is_file():
    print("Descriptive only — not used for selection.")
    display(Image(filename=str(fig)))
        """),
        markdown("## 9. Bayesian-network support"),
        code("""
display(recurrent_all[[
    "scenario_id", "upstream_labels", "B_label", "C_label",
    "positive_bn_path_support", "stable_positive_bn_path_support", "path_bootstrap_frequencies"
]].head(20))
        """),
        markdown("## 10. Manuscript-ready scenarios (highest recurrence)"),
        code("""
from scenario_analysis import print_primary_summary

article = mining["article"]
display(article)
display(pd.read_csv(paths["scenarios"] / "scenarios_article_table.csv"))
fig = FIGURES_DIR / "recurrent_scenarios_compact.png"
if fig.is_file():
    display(Image(filename=str(fig)))
print_primary_summary(
    matrix, roles,
    pd.read_csv(NETWORK_DIR / "global_bn_summary.csv"),
    bootstrap, edges, mining, paths, config,
)
        """),
        markdown("### Appendix diagnostics (optional)"),
        code("""
for name in ("factor_prevalence_by_role.png", "accident_factor_count_by_role.png", "learned_global_bn.png"):
    for folder in (MATRIX_DIR / "figures", NETWORK_DIR / "figures"):
        path = folder / name
        if path.is_file():
            print(path)
            display(Image(filename=str(path)))
            break
        """),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id in ("caou", "btp", "metallurgie"):
        notebook = build_notebook(dataset_id)
        out_path = NOTEBOOK_DIR / f"recurrent_scenarios_bn_analysis_{dataset_id}.ipynb"
        out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
