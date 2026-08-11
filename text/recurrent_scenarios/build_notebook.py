"""Build the explicit, auditable recurrent-scenarios notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "recurrent_scenarios_audit.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
    markdown(
        """
# Recurrent accident scenarios — audit notebook

This notebook implements the non-inter-sector part of the protocol:

1. fixed multilingual embeddings;
2. independent A0/A1/B/C UMAP-HDBSCAN clustering;
3. accident-level consensus;
4. binary accident-topic matrix;
5. topic frequencies, co-occurrence and lift;
6. constrained Bayesian networks without and with latent `Z`;
7. grouped out-of-sample log-likelihood, bootstrap arc stability and supported scenarios.

All parameters are visible below and every intermediate result is exported. A zero in the
accident-topic matrix means “not observed in the available units”, not confirmed absence.
Inter-sector transfer and alternative methods are intentionally excluded.
        """
    ),
    code(
        """
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm

SCENARIO_CANDIDATES = [
    Path.cwd() / "text" / "recurrent_scenarios",
    Path.cwd() / "recurrent_scenarios",
    Path.cwd(),
    Path.cwd().parent,
]
SCENARIO_DIR = next(
    candidate.resolve()
    for candidate in SCENARIO_CANDIDATES
    if (candidate / "config.yaml").is_file()
)
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import (
    ROLES,
    bootstrap_arc_stability,
    build_accident_topic_matrix,
    build_topic_dictionary,
    consensus_role,
    descriptive_tables,
    evaluate_bn_cv,
    extract_scenarios,
    fit_latent_model,
    fit_no_z_model,
    load_yaml_config,
    load_topic_stopwords,
    parameter_plan,
    prepare_data,
    resolve_config_paths,
    resolve_admissibility_rules,
    screen_clustering_parameters,
    mark_admissible_configurations,
    select_admissible_parameter_combinations,
    select_dataset_config,
    save_models,
    write_audit_report,
    write_consensus_figures,
    write_descriptive_figures,
    write_network_figure,
)
        """
    ),
    markdown(
        """
## 1. Editable parameters

Change this cell for an audit. `USE_DEBUG_REPETITIONS=True` uses the short repetition
count from `config.yaml`; switch it to `False` for the planned 100-repetition analysis.
        """
    ),
    code(
        """
CONFIG_PATH = SCENARIO_DIR / "config.yaml"
DATASET_ID = "caou"
USE_DEBUG_REPETITIONS = True
REESTIMATE = False  # Set True to recompute screening and consensus caches.
RUN_NAME = "notebook_audit"

raw_config = select_dataset_config(load_yaml_config(CONFIG_PATH), DATASET_ID)
config = resolve_config_paths(raw_config, CONFIG_PATH)
config.setdefault("runtime", {})["use_debug_repetitions"] = USE_DEBUG_REPETITIONS
config["runtime"]["overwrite"] = True
RUN_DIR = Path(config["data"]["output_dir"]) / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "config_resolved.yaml").write_text(
    __import__("yaml").safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print("Configuration:", CONFIG_PATH)
print("Dataset:", config["data"]["dataset_id"])
print("Run directory:", RUN_DIR)
print("Debug repetitions:", USE_DEBUG_REPETITIONS)
print("Reestimate cached results:", REESTIMATE)
print("Screening UMAP options:")
display(pd.Series(config["screening"]["umap"], name="value"))
print("Screening HDBSCAN options:")
display(pd.Series(config["screening"]["hdbscan"], name="value"))
print("Manual topic-representation options:")
display(pd.Series({
    "top_words": config["topics"]["top_words"],
    "top_sentences": config["topics"]["top_sentences"],
    "n_gram_range": config["topics"]["n_gram_range"],
    "min_topic_frequency": config["topics"]["min_topic_frequency"],
    "idf_smoothing": config["topics"]["idf_smoothing"],
}, name="value"))
topic_stopwords = load_topic_stopwords(config)
print("Stopwords utilisés pour la représentation c-TF-IDF:", len(topic_stopwords))
print("Fichier métier:", config["topics"].get("stopwords_file"))
display(pd.DataFrame({"stopword": sorted(topic_stopwords)}))
        """
    ),
    markdown("## 2. Imports, schema checks and frozen embeddings"),
    code(
        """
prepared = prepare_data(config, RUN_DIR)
display(prepared.input_summary)
display(prepared.units[["_accident_id", "_fact_id", "_role", "_text"]].head())
print("Embedding matrix:", prepared.embeddings.shape)
        """
    ),
    markdown(
        """
## 3. Preliminary UMAP-HDBSCAN screening — before consensus

This section must be reviewed before running the consensus stage. Each candidate
configuration is fitted once on a deterministic accident sample. The diagnostics
describe usability rather than optimize a single score:

- `n_clusters`: number of non-noise clusters;
- `noise_fraction` and `coverage`: rejected and retained factual units;
- `median_accident_support`: median number of distinct accidents represented in a cluster;
- `n_components`, `n_neighbors`, `min_cluster_size` and `min_samples`.

Topic-representation options are fixed during this screen. They control topic words and
labels rather than the primary partition and are shown in the configuration cell.

The screening results are saved as CSV files and reused on subsequent executions.
Set `REESTIMATE = True` in the configuration cell to force a new screening run.
        """
    ),
    code(
        """
parameter_grid = pd.DataFrame(parameter_plan(config, apply_selection=False))
display(parameter_grid)
print("Number of parameter combinations:", len(parameter_grid))
        """
    ),
    code("screening_results = {}"),
    markdown("### Screening — role A0"),
    code(
        """
screening_results["A0"] = screen_clustering_parameters(
    "A0", prepared.units, prepared.embeddings, config, RUN_DIR, reestimate=REESTIMATE
)
display(screening_results["A0"].sort_values(["noise_fraction", "median_accident_support"])[[
    "configuration_id", "umap_n_neighbors", "umap_n_components", "umap_min_dist",
    "hdbscan_min_cluster_size", "hdbscan_min_samples", "n_clusters", "noise_fraction",
    "coverage", "median_accident_support", "n_single_accident_clusters",
]])
        """
    ),
    markdown("### Screening — role A1"),
    code(
        """
screening_results["A1"] = screen_clustering_parameters(
    "A1", prepared.units, prepared.embeddings, config, RUN_DIR, reestimate=REESTIMATE
)
display(screening_results["A1"].sort_values(["noise_fraction", "median_accident_support"])[[
    "configuration_id", "umap_n_neighbors", "umap_n_components", "umap_min_dist",
    "hdbscan_min_cluster_size", "hdbscan_min_samples", "n_clusters", "noise_fraction",
    "coverage", "median_accident_support", "n_single_accident_clusters",
]])
        """
    ),
    markdown("### Screening — role B"),
    code(
        """
screening_results["B"] = screen_clustering_parameters(
    "B", prepared.units, prepared.embeddings, config, RUN_DIR, reestimate=REESTIMATE
)
display(screening_results["B"].sort_values(["noise_fraction", "median_accident_support"])[[
    "configuration_id", "umap_n_neighbors", "umap_n_components", "umap_min_dist",
    "hdbscan_min_cluster_size", "hdbscan_min_samples", "n_clusters", "noise_fraction",
    "coverage", "median_accident_support", "n_single_accident_clusters",
]])
        """
    ),
    markdown("### Screening — role C"),
    code(
        """
screening_results["C"] = screen_clustering_parameters(
    "C", prepared.units, prepared.embeddings, config, RUN_DIR, reestimate=REESTIMATE
)
display(screening_results["C"].sort_values(["noise_fraction", "median_accident_support"])[[
    "configuration_id", "umap_n_neighbors", "umap_n_components", "umap_min_dist",
    "hdbscan_min_cluster_size", "hdbscan_min_samples", "n_clusters", "noise_fraction",
    "coverage", "median_accident_support", "n_single_accident_clusters",
]])
        """
    ),
    markdown(
        """
## 4. Admissibility tables and role-specific thresholds

The rules are defined in `config.yaml` before the consensus and Bayesian-network stages.
The support threshold is resolved separately for each role from the number of accidents
containing that role. The summaries below are saved and reused with the screening CSVs.
        """
    ),
    code(
        """
RUN_CONSENSUS = False

role_admissibility_rules = {
    role: resolve_admissibility_rules(
        config,
        role,
        int(prepared.units.loc[prepared.units["_role"] == role, "_accident_id"].nunique()),
    )
    for role in ROLES
}
screening_marked = {
    role: mark_admissible_configurations(screening_results[role], role_admissibility_rules[role])
    for role in ROLES
}

configuration_summary = pd.DataFrame([
    {
        "rôle": role,
        "configurations": int(len(screening_results[role])),
        "admissibles": int(screening_marked[role]["admissible"].sum()),
    }
    for role in ROLES
])
display(configuration_summary)

summary_scope = config["screening"].get("summary_scope", "admissible")
metric_rows = []
for role in ROLES:
    source = screening_marked[role]
    if summary_scope == "admissible":
        source = source[source["admissible"]]
    metric_rows.append({
        "rôle": role,
        "clusters médian": float(source["n_clusters"].median()) if not source.empty else np.nan,
        "bruit médian": float(source["noise_fraction"].median()) if not source.empty else np.nan,
        "support médian": float(source["median_accident_support"].median()) if not source.empty else np.nan,
    })
metrics_summary = pd.DataFrame(metric_rows)
display(metrics_summary)
display(pd.DataFrame([
    {"rôle": role, "accidents du rôle": int(prepared.units.loc[prepared.units["_role"] == role, "_accident_id"].nunique()), **role_admissibility_rules[role]}
    for role in ROLES
]))

screening_dir = RUN_DIR / "screening"
screening_dir.mkdir(parents=True, exist_ok=True)
for role in ROLES:
    screening_marked[role].to_csv(screening_dir / f"parameter_screening_{role}_marked.csv", index=False)
configuration_summary.to_csv(screening_dir / "configuration_admissibility_summary.csv", index=False)
metrics_summary.to_csv(screening_dir / "screening_metrics_summary.csv", index=False)
config["screening"]["resolved_admissibility_by_role"] = role_admissibility_rules
if RUN_CONSENSUS:
    config["consensus"]["selected_parameter_combinations_by_role"] = (
        select_admissible_parameter_combinations(
            screening_results,
            rules_by_role=role_admissibility_rules,
        )
    )
    print({role: len(values) for role, values in config["consensus"]["selected_parameter_combinations_by_role"].items()})
else:
    print("Screening only. Review the tables, set RUN_CONSENSUS=True, then rerun this cell.")
(RUN_DIR / "config_resolved.yaml").write_text(
    __import__("yaml").safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
        """
    ),
    markdown("## 5. Independent accident-level consensus clustering"),
    code(
        """
if not RUN_CONSENSUS:
    raise RuntimeError("Review the screening tables first, set RUN_CONSENSUS=True, and rerun the previous cell.")

consensus_results = {}
for role in tqdm(ROLES, desc="Roles", unit="role"):
    print(f"Running consensus for {role}...")
    consensus_results[role] = consensus_role(
        role,
        prepared.units,
        prepared.embeddings,
        config,
        RUN_DIR,
        reestimate=REESTIMATE,
    )
    display(consensus_results[role].topics.head(20))
    display(consensus_results[role].replications.head())
write_consensus_figures(consensus_results, config, RUN_DIR / "figures")
display(pd.read_csv(RUN_DIR / "figures" / "consensus_tradeoff_summary.csv"))
from IPython.display import Image
display(Image(filename=str(RUN_DIR / "figures" / "consensus_tradeoff_all_roles.png")))
for role in ROLES:
    display(Image(filename=str(RUN_DIR / "figures" / f"coassociation_matrix_{role}.png")))
        """
    ),
    markdown("## 6. Topic dictionary and representative units"),
    code(
        """
topic_dictionary = build_topic_dictionary(prepared, consensus_results, config, RUN_DIR / "topics")
display(topic_dictionary.sort_values(["role", "n_accidents"], ascending=[True, False]))
        """
    ),
    markdown("## 7. Accident-topic matrix"),
    code(
        """
accident_topic_matrix, selected_topics, variable_macro_map = build_accident_topic_matrix(
    prepared,
    consensus_results,
    topic_dictionary,
    config,
    RUN_DIR / "matrices",
)
display(selected_topics)
display(accident_topic_matrix.head())
print("Selected variables:", len(variable_macro_map))
        """
    ),
    markdown("## 8. Descriptive frequencies, co-occurrence and lift"),
    code(
        """
descriptive = descriptive_tables(
    accident_topic_matrix,
    selected_topics,
    config,
    RUN_DIR / "descriptive",
)
display(descriptive["frequencies"].head(30))
display(descriptive["cooccurrence_lift"].head(30))
write_descriptive_figures(descriptive, RUN_DIR / "figures")
        """
    ),
    markdown(
        """
        ## 9. Constrained Bayesian networks

The observed process DAG allows `A0 → A1`, `A0 → B`, `A1 → B` and `B → C`; reverse
directions and intra-role arcs are blocked. The no-`Z` model is the global reference.
The `Z` model is a mixture of constrained binary BNs sharing that process DAG: it changes
the probability tables by latent family without treating `Z` as a process cause.
        """
    ),
    code(
        """
cv_log_likelihood = evaluate_bn_cv(
    accident_topic_matrix,
    variable_macro_map,
    config,
    RUN_DIR / "bayesian_networks",
)
display(cv_log_likelihood)
display(
    cv_log_likelihood.groupby(["model", "latent_states"], dropna=False)["log_likelihood"]
    .agg(["mean", "std"])
    .reset_index()
)
        """
    ),
    markdown("## 10. Final models, bootstrap arc stability and model comparison"),
    code(
        """
no_z_edges, no_z_model = fit_no_z_model(
    accident_topic_matrix[list(variable_macro_map)],
    variable_macro_map,
    config,
)
latent_models = {
    int(n_states): fit_latent_model(
        accident_topic_matrix[list(variable_macro_map)],
        no_z_edges,
        config,
        int(n_states),
        random_state=int(config["consensus"]["random_state"]) + int(n_states),
    )
    for n_states in config["bayesian_networks"]["latent_states"]
}
arc_stability = bootstrap_arc_stability(
    accident_topic_matrix,
    variable_macro_map,
    config,
    RUN_DIR / "bayesian_networks",
)
save_models(no_z_edges, no_z_model, latent_models, variable_macro_map, RUN_DIR / "bayesian_networks")
display(arc_stability.sort_values("bootstrap_frequency", ascending=False).head(50))
        """
    ),
    markdown("## 11. Supported recurrent scenarios"),
    code(
        """
latent_summary = (
    cv_log_likelihood[cv_log_likelihood["model"] == "BN_with_Z"]
    .groupby("latent_states")["log_likelihood"]
    .mean()
)
best_latent_state = int(latent_summary.idxmax()) if not latent_summary.empty else int(next(iter(latent_models)))
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
    markdown("## 12. Network figure and final audit report"),
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
    json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print(f"Notebook written: {NOTEBOOK_PATH}")
