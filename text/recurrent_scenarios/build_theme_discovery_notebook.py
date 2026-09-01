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
Candidates are screened on the Pareto front of accident-level reproducibility `S_R`
and UMAP-space DBCV. The geometric knee point on the normalized Pareto front
selects the final partition (deterministic, no LLM in configuration choice).
Seed sensitivity is run after selection.

After this notebook (or the Slurm discovery job), open
`topic_modeling_results_{corpus}.ipynb` for LLM cluster labels, then
`recurrent_scenarios_bn_analysis_{corpus}.ipynb` for the latent BN.
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
    evaluate_candidates,
    evaluate_resampling_stability,
    evaluate_seed_sensitivity,
    load_topic_stopwords,
    materialize_selected_partition,
    prepare_data,
    select_configuration_for_role,
    write_factor_resampling_manuscript_figures,
    write_pareto_normalized_knee_figure,
    write_stability_landscape_figure,
    write_umap_seed_sensitivity_all_roles_figure,
)
from pareto_knee_selection import identify_pareto_front, print_role_selection_summary
        """
    ),
    markdown(
        """
## 1. Paths, cache policy and random seeds

Edit this cell before a production run. Prefer the Slurm job for large corpora;
this notebook mirrors the same API for interactive audits.
        """
    ),
    code(
        """
DATASET_ID = "caou"
RUN_NAME = "theme_discovery_audit"
REESTIMATE = False
RANDOM_SEED = 42

TEXT_ROOT = SCENARIO_DIR.parent
DATA_ROOT = TEXT_ROOT / "dataset"
EMBEDDINGS_ROOT = TEXT_ROOT / "embeddings"
DATASET_PATHS = {
    "caou": (DATA_ROOT / "data_caou.csv", EMBEDDINGS_ROOT / "Qwen3-Embedding-0.6B_caou.csv"),
    "btp": (DATA_ROOT / "data_btp.csv", EMBEDDINGS_ROOT / "Qwen3-Embedding-0.6B_btp.csv"),
    "metallurgie": (DATA_ROOT / "data_metallurgie.csv", EMBEDDINGS_ROOT / "Qwen3-Embedding-0.6B_metallurgie.csv"),
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
    markdown("## 2. UMAP / HDBSCAN / validation parameters (aligned with config.yaml)"),
    code(
        """
UMAP_PARAMETERS = {
    "n_neighbors": [10, 20, 40],
    "n_components": [5, 10, 15],
    "min_dist": [0.0],
    "metric": "cosine",
    "low_memory": True,
    "n_jobs": 1,
}
HDBSCAN_PARAMETERS = {
    "min_cluster_size": [25, 50],
    "min_samples": [5, 10],
    "cluster_selection_method": ["leaf"],
    "metric": "euclidean",
    "prediction_data": True,
}
RESAMPLING_FRACTION = 0.80
N_RESAMPLING = 30
DBCV_SAMPLE_SIZE = None
SEED_SENSITIVITY_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SHOW_PROGRESS = True
TOP_WORDS = 12
TOP_SENTENCES = 5
N_GRAM_RANGE = (1, 2)
MIN_TOPIC_FREQUENCY = 1
IDF_SMOOTHING = 1.0
STOPWORDS_FILE = SCENARIO_DIR / "stop_metier.txt"
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
    "random_state": RANDOM_SEED,
    "validation": {
        "random_state": RANDOM_SEED,
        "n_resampling": N_RESAMPLING,
        "resampling_fraction": RESAMPLING_FRACTION,
        "dbcv_sample_size": DBCV_SAMPLE_SIZE,
        "selection_metric": "pareto_geometric_knee",
        "show_progress": SHOW_PROGRESS,
        "seed_sensitivity": {"enabled": True, "seeds": SEED_SENSITIVITY_SEEDS},
    },
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
    "runtime": {"save_intermediate_assignments": True},
}
(RUN_DIR / "theme_discovery_parameters.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        """
    ),
    markdown("## 3. Load corpus"),
    code(
        """
prepared = prepare_data(config, RUN_DIR)
display(prepared.input_summary)
print("Embedding matrix:", prepared.embeddings.shape)
display(prepared.units.groupby("_role").agg(n_units=("_fact_id", "size"), n_accidents=("_accident_id", "nunique")))
print("Loaded domain stopwords:", len(load_topic_stopwords(config)))
        """
    ),
    code(
        """
candidate_tables = {}
theme_stability = {}
selection_tables = {}
selections = {}
        """
    ),
]

for role in ("A0", "A1", "B", "C"):
    cells.extend([
        markdown(f"## 4.{role} Metrics + selection"),
        code(
            f"""
role = "{role}"
candidates = evaluate_candidates(
    role, prepared.units, prepared.embeddings, config, RUN_DIR, reestimate=REESTIMATE
)
theme_stability[role], summary = evaluate_resampling_stability(
    role, prepared.units, prepared.embeddings, config, RUN_DIR,
    candidates, reestimate=REESTIMATE
)
merged = identify_pareto_front(candidates.merge(summary, on=["role", "configuration_id"], how="left"))
selection_tables[role], selected_id, rule = select_configuration_for_role(merged)
candidate_tables[role] = selection_tables[role]
selections[role] = selected_id
materialize_selected_partition(role, prepared.units, selected_id, RUN_DIR, theme_stability[role])
print_role_selection_summary(role, selection_tables[role], selected_id=selected_id)
cols = [c for c in [
    "configuration_id", "stability", "dbcv_umap", "is_pareto", "is_selected_knee",
    "stability_normalized", "dbcv_normalized", "knee_distance",
    "n_clusters", "noise_fraction", "coverage",
] if c in selection_tables[role].columns]
display(selection_tables[role].sort_values(["is_pareto", "dbcv_umap", "stability"], ascending=[False, True, False])[cols].head(12))
print("Selected:", selected_id, "via", rule)
            """
        ),
    ])

cells.extend([
    markdown("## 5. Pareto figures and seed sensitivity"),
    code(
        """
write_stability_landscape_figure(selection_tables, RUN_DIR / "figures")
write_pareto_normalized_knee_figure(selection_tables, RUN_DIR / "figures")
write_factor_resampling_manuscript_figures(theme_stability, selections, RUN_DIR / "figures")
display(Image(filename=str(RUN_DIR / "figures" / "stability_landscape_all_roles.png")))
display(Image(filename=str(RUN_DIR / "figures" / "pareto_normalized_knee_all_roles.png")))
display(Image(filename=str(RUN_DIR / "figures" / "factor_resampling_A0.png")))
selected_rows = []
for role in ROLES:
    row = selection_tables[role].loc[selection_tables[role]["is_selected_knee"]].iloc[0]
    selected_rows.append({
        "role": role,
        "configuration_id": selections[role],
        "selection_rule": "geometric_knee" if int(selection_tables[role]["is_pareto"].sum()) > 1 else "single_pareto",
        "stability": row["stability"],
        "dbcv_umap": row["dbcv_umap"],
        "stability_normalized": row.get("stability_normalized"),
        "dbcv_normalized": row.get("dbcv_normalized"),
        "knee_distance": row.get("knee_distance"),
        "n_clusters": row["n_clusters"],
        "noise_fraction": row["noise_fraction"],
        "coverage": row["coverage"],
    })
    evaluate_seed_sensitivity(
        role,
        prepared.units,
        prepared.embeddings,
        config,
        RUN_DIR,
        selections[role],
        row.to_dict(),
        reestimate=REESTIMATE,
    )
write_umap_seed_sensitivity_all_roles_figure(RUN_DIR / "figures", run_dir=RUN_DIR)
display(Image(filename=str(RUN_DIR / "figures" / "umap_seed_sensitivity_all_roles.png")))
pd.DataFrame(selected_rows).to_csv(RUN_DIR / "selected_configurations.csv", index=False)
display(pd.DataFrame(selected_rows))
print("Open topic_modeling_results_<corpus>.ipynb for LLM labels, then the BN notebook.")
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
