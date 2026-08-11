"""Build the downstream accident-variable and Bayesian-network notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "recurrent_scenarios_bn_analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
    markdown(
        """
# Notebook 2 — Accident variables, Bayesian networks and recurrent scenarios

This notebook starts after Notebook 1. It loads the frozen Pareto-medoid theme
memberships and never reruns UMAP, HDBSCAN, DBCV or resampling stability.

The workflow is: accident-theme matrix → descriptive supports/lift → constrained
BN without `Z` → latent mixture BN → grouped held-out comparison and bootstrap
arc stability → traceable recurrent scenarios.
        """
    ),
    code(
        """
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from IPython.display import display

SCENARIO_DIR = Path.cwd() / "text" / "recurrent_scenarios"
if not (SCENARIO_DIR / "scenario_pipeline.py").is_file():
    SCENARIO_DIR = Path.cwd()
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import (
    ConsensusResult,
    ROLES,
    bootstrap_arc_stability,
    build_accident_topic_matrix,
    descriptive_tables,
    evaluate_bn_cv,
    extract_scenarios,
    fit_latent_model,
    fit_no_z_model,
    prepare_data,
    save_models,
    write_audit_report,
    write_descriptive_figures,
    write_network_figure,
)
        """
    ),
    markdown(
        """
## 1. Paths and downstream model parameters

All parameters used by the downstream analysis are kept in this notebook. The
`RUN_DIR` must point to a completed Notebook 1 run. Changing any parameter below
requires recording the change and rerunning the affected downstream cells.
        """
    ),
    code(
        """
DATASET_ID = "caou"
DISCOVERY_RUN_NAME = "theme_discovery_audit"
RUN_DIR = SCENARIO_DIR / "runs" / DISCOVERY_RUN_NAME / DATASET_ID
DATA_ROOT = SCENARIO_DIR.parent / "dataset"
UNITS_PATH = DATA_ROOT / f"data_{DATASET_ID}.csv"
EMBEDDINGS_PATH = DATA_ROOT / f"Qwen3-Embedding-0.6B_{DATASET_ID}.csv"

# Matrix construction: rare themes can be retained in the dictionary but excluded
# from the main BN because their conditional tables are too sparse.
MIN_TOPIC_ACCIDENT_SUPPORT = 20  # Minimum number of accidents supporting a BN variable.
MAX_TOPICS_PER_ROLE = 6  # Optional cap on retained variables per role; None keeps all.

# BN structure: role-order constraints are implemented in the existing learner.
N_FOLDS = 5  # Grouped accident-level folds for held-out log likelihood.
MAX_INDEGREE = 3  # Maximum observed parents; larger values create sparse CPT cells.
EQUIVALENT_SAMPLE_SIZE = 5.0  # Symmetric smoothing strength for binary CPT estimates.
INCLUDE_A0_TO_B_DIRECT = True  # Allow the direct context-to-event relation.
ENSURE_MACRO_CHAIN_BACKBONE = False  # Do not force edges unless substantively justified.

# Latent mixture model: Z captures upstream heterogeneity, not a process cause.
LATENT_STATES = [2, 3, 4]  # Candidate number of latent accident families.
LATENT_MAX_ITER = 100  # EM iteration budget per initialization/state count.
LATENT_TOLERANCE = 1e-4  # Convergence tolerance on mean observed log likelihood.

# Structural uncertainty and scenario output.
BOOTSTRAP_REPETITIONS = 100  # Accident bootstrap fits for edge selection frequencies.
BOOTSTRAP_ARC_THRESHOLD = 0.75  # Display/retain learned arcs above this frequency.
TOP_SCENARIOS = 30  # Maximum scenarios written to the catalogue.
MIN_SCENARIO_SUPPORT = 3  # Minimum exact number of supporting accidents.

config = {
    "data": {
        "dataset_id": DATASET_ID,
        "units_path": str(UNITS_PATH),
        "embeddings_path": str(EMBEDDINGS_PATH),
        "accident_id_col": "accident_id",
        "fact_id_col": "fact_id",
        "text_col": "sentence",
        "role_col": "pred_label",
        "valid_col": "pred_ok",
        "keep_valid_only": True,
        "normalize_embeddings": True,
        "output_dir": str(RUN_DIR),
    },
    "consensus": {
        "random_state": 42,
        "min_topic_accident_support": MIN_TOPIC_ACCIDENT_SUPPORT,
        "max_topics_per_role": MAX_TOPICS_PER_ROLE,
    },
    "descriptive": {"bootstrap_repetitions": 1000, "cooccurrence_min_support": 5},
    "bayesian_networks": {
        "n_folds": N_FOLDS,
        "max_indegree": MAX_INDEGREE,
        "equivalent_sample_size": EQUIVALENT_SAMPLE_SIZE,
        "latent_states": LATENT_STATES,
        "latent_max_iter": LATENT_MAX_ITER,
        "latent_tolerance": LATENT_TOLERANCE,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_arc_threshold": BOOTSTRAP_ARC_THRESHOLD,
        "include_a0_to_b_direct": INCLUDE_A0_TO_B_DIRECT,
        "ensure_macro_chain_backbone": ENSURE_MACRO_CHAIN_BACKBONE,
        "top_scenarios": TOP_SCENARIOS,
        "min_scenario_support": MIN_SCENARIO_SUPPORT,
    },
}
(RUN_DIR / "bn_analysis_parameters.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Using discovery run:", RUN_DIR)
        """
    ),
    markdown("## 2. Validate and load the frozen discovery outputs"),
    code(
        """
prepared = prepare_data(config, RUN_DIR)
consensus_results = {}
for role in ROLES:
    role_dir = RUN_DIR / "clustering" / role
    assignments_path = role_dir / "topic_assignments.csv"
    topics_path = role_dir / "topics.csv"
    if not assignments_path.is_file() or not topics_path.is_file():
        raise FileNotFoundError(f"Missing frozen outputs for {role}: run Notebook 1 first.")
    consensus_results[role] = ConsensusResult(
        role=role,
        assignments=pd.read_csv(assignments_path),
        topics=pd.read_csv(topics_path),
        edges=pd.DataFrame(),
        replications=pd.DataFrame(),
    )
    print(role, "topics:", len(consensus_results[role].topics), "assigned units:", int(consensus_results[role].assignments["topic_id"].ne("").sum()))
topic_dictionary = pd.read_csv(RUN_DIR / "topics" / "topic_dictionary.csv")
display(topic_dictionary[["topic_id", "role", "label", "n_units", "n_accidents", "stability"]].head(30))
        """
    ),
    markdown(
        """
## 3. Construct the binary accident-theme matrix

One is “theme observed in the available narrative”; zero is not confirmed absence
from the real accident. Repeated factual units within one accident produce one binary
indicator, not a count.
        """
    ),
    code(
        """
accident_topic_matrix, selected_topics, variable_macro_map = build_accident_topic_matrix(
    prepared, consensus_results, topic_dictionary, config, RUN_DIR / "matrices"
)
display(selected_topics)
display(accident_topic_matrix.head())
print("Accidents:", len(accident_topic_matrix), "BN variables:", len(variable_macro_map))
        """
    ),
    markdown("## 4. Descriptive prevalence and pairwise lift"),
    code(
        """
descriptive = descriptive_tables(
    accident_topic_matrix, selected_topics, config, RUN_DIR / "descriptive"
)
display(descriptive["frequencies"].head(30))
display(descriptive["cooccurrence_lift"].head(30))
write_descriptive_figures(descriptive, RUN_DIR / "figures")
        """
    ),
    markdown(
        """
## 5. Constrained BN without `Z` and held-out comparison

The learned observed arcs respect the A0/A1/B/C role ordering. They are conditional
dependencies under constraints, not identified causal effects.
        """
    ),
    code(
        """
cv_log_likelihood = evaluate_bn_cv(
    accident_topic_matrix, variable_macro_map, config, RUN_DIR / "bayesian_networks"
)
display(cv_log_likelihood)
display(cv_log_likelihood.groupby(["model", "latent_states"], dropna=False)["log_likelihood"].agg(["mean", "std"]).reset_index())
        """
    ),
    markdown("## 6. Final latent model, bootstrap arc stability and comparison"),
    code(
        """
no_z_edges, no_z_model = fit_no_z_model(
    accident_topic_matrix[list(variable_macro_map)], variable_macro_map, config
)
latent_models = {
    int(n_states): fit_latent_model(
        accident_topic_matrix[list(variable_macro_map)],
        no_z_edges,
        config,
        int(n_states),
        random_state=int(config["consensus"]["random_state"]) + int(n_states),
    )
    for n_states in LATENT_STATES
}
arc_stability = bootstrap_arc_stability(
    accident_topic_matrix, variable_macro_map, config, RUN_DIR / "bayesian_networks"
)
save_models(no_z_edges, no_z_model, latent_models, variable_macro_map, RUN_DIR / "bayesian_networks")
display(arc_stability.sort_values("bootstrap_frequency", ascending=False).head(50))
        """
    ),
    markdown("## 7. Recurrent scenarios supported by the frozen variables and network"),
    code(
        """
latent_summary = cv_log_likelihood[cv_log_likelihood["model"].eq("BN_with_Z")].groupby("latent_states")["log_likelihood"].mean()
best_latent_state = int(latent_summary.idxmax()) if not latent_summary.empty else int(LATENT_STATES[0])
scenario_catalog = extract_scenarios(
    accident_topic_matrix,
    selected_topics,
    variable_macro_map,
    no_z_model,
    {best_latent_state: latent_models[best_latent_state]},
    config,
    RUN_DIR / "scenarios",
)
display(scenario_catalog)
print("Selected latent states:", best_latent_state)
        """
    ),
    markdown("## 8. Export final figures and audit report"),
    code(
        """
write_network_figure(
    no_z_edges,
    variable_macro_map,
    arc_stability,
    RUN_DIR / "figures" / "constrained_networks_without_with_z.png",
)
write_audit_report(config, prepared, selected_topics, cv_log_likelihood, arc_stability, RUN_DIR)
display(pd.read_csv(RUN_DIR / "audit_input_summary.csv"))
print("Outputs written to:", RUN_DIR)
        """
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(
    json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(f"Notebook written: {NOTEBOOK_PATH}")
