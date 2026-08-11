"""Build the role-conditioned theme-discovery notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "recurrent_scenarios_theme_discovery.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
    markdown(
        """
# Notebook 1 — Role-conditioned stable semantic themes

This notebook discovers stable themes independently for `A0`, `A1`, `B` and `C`.
It does not run the accident-theme matrix, Bayesian networks or scenario extraction.

The primary selection plane is:

- horizontal axis: UMAP-space DBCV `D_U`;
- vertical axis: accident-level resampling stability `S_R`;
- no pre-filtering before the Pareto comparison;
- final configuration: Pareto non-dominated partition with the highest agreement
  with the other non-dominated partitions (partition medoid).

Noise (`-1`) remains unassigned and is never treated as a theme. The second notebook
loads the frozen outputs written here and therefore does not rerun UMAP or HDBSCAN.
        """
    ),
    code(
        """
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

scenario_dir_candidates = []
for base_dir in [Path.cwd(), *Path.cwd().parents]:
    scenario_dir_candidates.extend(
        [
            base_dir / "text" / "recurrent_scenarios",
            base_dir / "recurrent_scenarios",
            base_dir,
        ]
    )
SCENARIO_DIR = next(
    (candidate for candidate in scenario_dir_candidates if (candidate / "scenario_pipeline.py").is_file()),
    None,
)
if SCENARIO_DIR is None:
    raise FileNotFoundError(
        "Could not locate scenario_pipeline.py. Launch Jupyter inside the project or set SCENARIO_DIR explicitly."
    )
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import (
    ROLES,
    build_selected_consensus_results,
    build_topic_dictionary,
    evaluate_pareto_candidates,
    evaluate_resampling_stability,
    load_topic_stopwords,
    prepare_data,
    select_pareto_partitions,
    write_pareto_figure,
)
        """
    ),
    markdown(
        """
## 1. Paths, cache policy and random seeds

Edit this cell before a production run. `REESTIMATE=True` invalidates the cached
candidate fits, DBCV values, resampling values and Pareto selection for this run.
Keeping it `False` is the normal mode after an intermediate result has been audited.
        """
    ),
    code(
        """
DATASET_ID = "btp"  # Registry identifier used in the existing corpus exports.
RUN_NAME = "theme_discovery_audit"
REESTIMATE = False  # False = reuse saved tables; True = recompute every discovery artifact.
RANDOM_SEED = 42  # Global seed; changing it measures stochastic UMAP sensitivity.

# These paths are deliberately visible here rather than hidden in a config file.
DATA_ROOT = SCENARIO_DIR.parent / "dataset"
DATASET_PATHS = {
    "btp": (DATA_ROOT / "data_btp.csv", DATA_ROOT / "Qwen3-Embedding-0.6B_btp.csv"),
    "metallurgie": (DATA_ROOT / "data_metallurgie.csv", DATA_ROOT / "Qwen3-Embedding-0.6B_metallurgie.csv"),
    "caou": (DATA_ROOT / "data_caou.csv", DATA_ROOT / "Qwen3-Embedding-0.6B_caou.csv"),
    "nicollin": (DATA_ROOT / "data_nicollin.csv", DATA_ROOT / "Qwen3-Embedding-0.6B_nicollin.csv"),
}
UNITS_PATH, EMBEDDINGS_PATH = DATASET_PATHS[DATASET_ID]
RUN_DIR = SCENARIO_DIR / "runs" / RUN_NAME / DATASET_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("Dataset:", DATASET_ID)
print("Units:", UNITS_PATH)
print("Embeddings:", EMBEDDINGS_PATH)
print("Run directory:", RUN_DIR)
print("Reestimate:", REESTIMATE)
        """
    ),
    markdown(
        """
## 2. UMAP and HDBSCAN parameters

These values define the candidate grid. `n_neighbors` controls the neighbourhood
scale used by UMAP; higher values favour broader structure. `n_components` controls
the dimension in which HDBSCAN estimates density; it is not a 2-D plotting setting.
`min_dist` controls how tightly UMAP packs nearby points. `min_cluster_size` is the
smallest persistent branch considered a theme. `min_samples` controls density
conservatism; higher values usually create more noise. `eom` keeps broader persistent
branches, while `leaf` explores finer terminal branches.
        """
    ),
    code(
        """
UMAP_PARAMETERS = {
    "n_neighbors": [10, 20, 40],  # Local-to-global neighbourhood scale.
    "n_components": [5, 10, 15],  # Density-clustering dimension, not visualization dimension.
    "min_dist": [0.0, 0.05, 0.1],  # Minimum packing distance in the UMAP space.
    "metric": "cosine",  # Distance between frozen normalized sentence embeddings.
    "low_memory": True,  # Lower memory use at the cost of some runtime.
    "n_jobs": 1,  # Keep one deterministic UMAP worker; increase only for a planned sensitivity run.
}
HDBSCAN_PARAMETERS = {
    "min_cluster_size": [20, 50, 100],  # Minimum candidate theme mass.
    "min_samples": [None, 5, 10],  # Local density conservatism; None follows min_cluster_size.
    "cluster_selection_method": ["eom", "leaf"],  # Broad persistent branches versus fine leaves.
    "metric": "euclidean",  # Metric in the UMAP clustering coordinates.
    "prediction_data": True,  # Retain prediction metadata for auditability.
}
MAX_PARAMETER_COMBINATIONS = 50  # Explicit computational budget; None evaluates the full Cartesian grid.
        """
    ),
    markdown(
        """
## 3. Resampling and DBCV parameters

Stability resamples accidents, not factual-unit rows, so units from one narrative
remain together. For each theme, stability is the mean of its best-match Jaccard
values across resampling repetitions, and `S_R` is the mean of the resulting theme
stabilities. DBCV is computed in the UMAP space used by HDBSCAN. Set
`DBCV_SAMPLE_SIZE=None` for the exact full-role calculation; a finite value is an
explicit approximation and is saved in the diagnostics table.
        """
    ),
    code(
        """
RESAMPLING_FRACTION = 0.80  # Fraction of distinct accidents retained in each replicate.
N_RESAMPLING = 50  # Monte-Carlo repetitions for the primary stability estimate.
DEBUG_RESAMPLING = 5  # Used only when USE_DEBUG_REPETITIONS=True.
USE_DEBUG_REPETITIONS = True  # Short audit run; set False for the planned analysis.
DBCV_SAMPLE_SIZE = None  # None = exact role calculation; finite value = explicit speed approximation.
SHOW_PROGRESS = True
        """
    ),
    markdown(
        """
## 4. Pareto objectives

The main Pareto front uses only UMAP-space DBCV `D_U` and resampling stability
`S_R`. No pre-filtering threshold is applied before the Pareto comparison. The
number of clusters, coverage, noise fraction and accident support are retained
only as diagnostics. The main figure is a two-dimensional scatter plot: leaf
configurations use circles, EOM configurations use triangles, dominated points
are shown as open grey markers, and Pareto points are shown in orange. Pareto
solutions are joined in increasing `D_U` order, and the representative medoid
is marked with a star. The complete Pareto diagnostic table is saved separately
for each role.
        """
    ),
    markdown(
        """
## 5. Text representation parameters

These parameters are applied only after the selected memberships are frozen. They
describe themes with c-TF-IDF-inspired terms and representative sentences; they do
not change UMAP, HDBSCAN, DBCV, stability or Pareto selection.
        """
    ),
    code(
        """
TOP_WORDS = 12  # Number of discriminative terms retained per theme.
TOP_SENTENCES = 5  # Number of central/boundary examples retained per theme.
N_GRAM_RANGE = (1, 2)  # Unigrams and bigrams for audit labels.
MIN_TOPIC_FREQUENCY = 1  # Minimum number of themes containing a term.
IDF_SMOOTHING = 1.0  # Additive smoothing in the c-TF-IDF-inspired score.
STOPWORDS_FILE = SCENARIO_DIR / "stop_metier.txt"  # Domain terms excluded from labels.
        """
    ),
    code(
        """
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
    "screening": {
        "show_progress": SHOW_PROGRESS,
        "umap": UMAP_PARAMETERS,
        "hdbscan": HDBSCAN_PARAMETERS,
    },
    "pareto": {
        "random_state": RANDOM_SEED,
        "max_parameter_combinations": MAX_PARAMETER_COMBINATIONS,
        "n_resampling": N_RESAMPLING,
        "debug_resampling": DEBUG_RESAMPLING,
        "resampling_fraction": RESAMPLING_FRACTION,
        "dbcv_sample_size": DBCV_SAMPLE_SIZE,
        "show_progress": SHOW_PROGRESS,
    },
    "consensus": {"random_state": RANDOM_SEED},
    "topics": {
        "top_words": TOP_WORDS,
        "top_sentences": TOP_SENTENCES,
        "n_gram_range": list(N_GRAM_RANGE),
        "min_topic_frequency": MIN_TOPIC_FREQUENCY,
        "idf_smoothing": IDF_SMOOTHING,
        "stopwords_file": str(STOPWORDS_FILE),
        "stopwords": [],
        "additional_stopwords": [],
    },
    "runtime": {"use_debug_repetitions": USE_DEBUG_REPETITIONS, "save_intermediate_assignments": True},
}
(RUN_DIR / "theme_discovery_parameters.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        """
    ),
    markdown("## 6. Load and audit the role-labelled corpus"),
    code(
        """
prepared = prepare_data(config, RUN_DIR)
display(prepared.input_summary)
display(prepared.units[["_accident_id", "_fact_id", "_role", "_text"]].head())
print("Embedding matrix:", prepared.embeddings.shape)
display(prepared.units.groupby("_role").agg(n_units=("_fact_id", "size"), n_accidents=("_accident_id", "nunique")))
topic_stopwords = load_topic_stopwords(config)
print("Loaded domain stopwords:", len(topic_stopwords))
        """
    ),
    markdown("## 7. Candidate grid"),
    code(
        """
from itertools import product
grid = pd.DataFrame([
    {
        "umap_n_neighbors": n_neighbors,
        "umap_n_components": n_components,
        "umap_min_dist": min_dist,
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples": min_samples,
        "hdbscan_cluster_selection_method": method,
    }
    for n_neighbors, n_components, min_dist, min_cluster_size, min_samples, method
    in product(UMAP_PARAMETERS["n_neighbors"], UMAP_PARAMETERS["n_components"], UMAP_PARAMETERS["min_dist"], HDBSCAN_PARAMETERS["min_cluster_size"], HDBSCAN_PARAMETERS["min_samples"], HDBSCAN_PARAMETERS["cluster_selection_method"])
])
if MAX_PARAMETER_COMBINATIONS is not None and len(grid) > MAX_PARAMETER_COMBINATIONS:
    print(f"The grid contains {len(grid)} combinations; the pipeline will keep the declared budget of {MAX_PARAMETER_COMBINATIONS}.")
display(grid)
        """
    ),
    code(
        """
candidate_tables = {}
theme_stability = {}
stability_summary = {}
selection_tables = {}
agreement_matrices = {}
selected_configuration = {}
        """
    ),
]

for role in ("A0", "A1", "B", "C"):
    cells.extend([
        markdown(f"## 8.{role} Compute validation metrics"),
        code(
            f"""
role = "{role}"
candidate_tables[role] = evaluate_pareto_candidates(
    role, prepared.units, prepared.embeddings, config, RUN_DIR, reestimate=REESTIMATE
)
theme_stability[role], stability_summary[role] = evaluate_resampling_stability(
    role, prepared.units, prepared.embeddings, config, RUN_DIR,
    candidate_tables[role], reestimate=REESTIMATE
)
candidate_tables[role] = candidate_tables[role].merge(
    stability_summary[role], on=["role", "configuration_id"], how="left"
)
display(candidate_tables[role].sort_values(["dbcv_umap", "stability"], ascending=False)[[
    "configuration_id", "hdbscan_cluster_selection_method", "dbcv_umap",
    "stability", "n_clusters", "noise_fraction", "coverage", "median_accident_support",
]])
            """
        ),
        markdown(f"## 8.{role} Construct the Pareto front"),
        code(
            f"""
role = "{role}"
selection_tables[role], agreement_matrices[role], selected_configuration[role] = select_pareto_partitions(
    role, candidate_tables[role], RUN_DIR,
)
display(selection_tables[role].sort_values(["pareto_non_dominated", "dbcv_umap"], ascending=False)[[
    "configuration_id", "hdbscan_cluster_selection_method", "dbcv_umap",
    "stability", "n_clusters", "noise_fraction", "coverage", "median_accident_support",
    "pareto_non_dominated", "representativeness",
]])
print("Selected Pareto-medoid configuration:", selected_configuration[role])
display(agreement_matrices[role])
role_figure = RUN_DIR / "figures" / f"pareto_validation_{{role}}.png"
write_pareto_figure(
    {{role: selection_tables[role]}},
    RUN_DIR / "figures",
    roles=[role],
    filename=role_figure.name,
)
display(Image(filename=str(role_figure)))
            """
        ),
    ])

cells.extend([
    markdown(
        """
## 9. Pareto validation plot

The figure is a two-dimensional scatter plot with `D_U` on the horizontal axis
and `S_R` on the vertical axis. Leaf configurations use circles and EOM
configurations use triangles. Dominated configurations are open grey markers,
Pareto configurations are orange markers joined by the Pareto front, and the
selected partition medoid is marked with a red star. There are deliberately no
universal vertical or horizontal cut-offs for DBCV or stability.
        """
    ),
    code(
        """
write_pareto_figure(selection_tables, RUN_DIR / "figures")
display(Image(filename=str(RUN_DIR / "figures" / "pareto_validation_all_roles.png")))
summary = pd.DataFrame([
    {
        "role": role,
        "selected_configuration": selected_configuration[role],
        "n_candidates": len(selection_tables[role]),
        "n_pareto": int(selection_tables[role]["pareto_non_dominated"].sum()),
    }
    for role in ROLES
])
display(summary)
summary.to_csv(RUN_DIR / "pareto_selection_summary.csv", index=False)
        """
    ),
    markdown(
        """
## 10. Freeze the selected memberships and build the theme dictionary

This is the hand-off boundary to Notebook 2. Cluster memberships are now fixed.
The dictionary terms and representative sentences are descriptive only and cannot
modify the selected assignments.
        """
    ),
    code(
        """
if any(not selected_configuration[role] for role in ROLES):
    raise RuntimeError("At least one role has no selected Pareto configuration. Review the candidate grid and valid D_U/S_R values.")
consensus_results = build_selected_consensus_results(
    prepared, config, RUN_DIR, selected_configuration, theme_stability
)
topic_dictionary = build_topic_dictionary(prepared, consensus_results, config, RUN_DIR / "topics")
display(topic_dictionary.sort_values(["role", "n_accidents"], ascending=[True, False]))
pd.DataFrame([{"role": role, "configuration_id": selected_configuration[role]} for role in ROLES]).to_csv(
    RUN_DIR / "selected_configurations.csv", index=False
)
print("Frozen outputs:", RUN_DIR / "clustering")
print("Notebook 2 can now be run without recomputing UMAP or HDBSCAN.")
        """
    ),
])

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
