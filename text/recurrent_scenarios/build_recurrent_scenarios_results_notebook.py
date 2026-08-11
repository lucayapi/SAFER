"""Build the post-run results notebook for recurrent-accident discovery."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "recurrent_scenarios_results.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
    markdown(
        """
# Recurrent accident scenarios — results notebook

This notebook presents the artifacts produced by the Slurm theme-discovery job.
It does **not** rerun UMAP, HDBSCAN, DBCV or resampling. The primary selection
plane is `D_U` (UMAP-space DBCV) versus `S_R` (accident-level Jaccard stability).
The selected partition is the medoid of the non-dominated configurations.
        """
    ),
    code(
        """
from pathlib import Path
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

SCENARIO_DIR = Path.cwd()
if not (SCENARIO_DIR / "scenario_pipeline.py").is_file():
    candidates = [
        Path.cwd() / "text" / "recurrent_scenarios",
        Path.cwd() / "recurrent_scenarios",
        Path.cwd().parent / "recurrent_scenarios",
    ]
    SCENARIO_DIR = next((path for path in candidates if (path / "scenario_pipeline.py").is_file()), SCENARIO_DIR)

# Set SAFER_THEME_RUN_DIR before launching Jupyter, or edit this cell.
RUN_DIR = Path(os.environ.get("SAFER_THEME_RUN_DIR", SCENARIO_DIR / "runs" / "audit_caou"))
ROLES = ("A0", "A1", "B", "C")
if not RUN_DIR.is_dir():
    raise FileNotFoundError(f"Run directory not found: {RUN_DIR}")
print("Scenario directory:", SCENARIO_DIR)
print("Run directory:", RUN_DIR)
        """
    ),
    markdown("## 1. Run manifest and resolved parallelism"),
    code(
        """
def read_json(name, default=None):
    path = RUN_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default

manifest = read_json("theme_discovery_manifest.json", {})
parallel = read_json("parallel_runtime.json", {})
config = read_json("theme_discovery_parameters.json", {})
display(pd.DataFrame([{
    "dataset": manifest.get("dataset_id", config.get("data", {}).get("dataset_id")),
    "selection_objectives": ", ".join(manifest.get("selection_objectives", ["dbcv_umap", "stability"])),
    "n_workers": parallel.get("n_workers", manifest.get("n_workers")),
    "slurm_cpus_per_task": parallel.get("slurm_cpus_per_task"),
    "backend": parallel.get("backend"),
    "inner_umap_n_jobs": parallel.get("inner_umap_n_jobs"),
}]))
display(pd.read_csv(RUN_DIR / "audit_input_summary.csv"))
        """
    ),
    markdown("## 2. Selected configurations"),
    code(
        """
selected = pd.read_csv(RUN_DIR / "selected_configurations.csv") if (RUN_DIR / "selected_configurations.csv").is_file() else pd.DataFrame()
display(selected)
for role in ROLES:
    path = RUN_DIR / "pareto" / role / "pareto_frontier.csv"
    if path.is_file():
        print(f"{role} — Pareto frontier")
        display(pd.read_csv(path))
        print()
        """
    ),
    markdown(
        """
## 3. Validation landscapes

Grey/open markers are dominated candidates, orange markers are non-dominated
candidates, and the red star is the selected partition medoid. The horizontal
axis is `D_U`, the primary DBCV metric used for Pareto selection.
        """
    ),
    code(
        """
figure_path = RUN_DIR / "figures" / "pareto_validation_all_roles.png"
if figure_path.is_file():
    display(Image(filename=str(figure_path)))
for role in ROLES:
    figure_path = RUN_DIR / "figures" / f"pareto_validation_{role}.png"
    if figure_path.is_file():
        display(Image(filename=str(figure_path)))
        """
    ),
    markdown("## 4. Candidate diagnostics and stability"),
    code(
        """
for role in ROLES:
    metrics_path = RUN_DIR / "pareto" / role / "candidate_metrics.csv"
    stability_path = RUN_DIR / "pareto" / role / "stability_summary.csv"
    theme_path = RUN_DIR / "pareto" / role / "stability_theme.csv"
    print(f"### {role}")
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        if stability_path.is_file():
            metrics = metrics.merge(pd.read_csv(stability_path), on=["role", "configuration_id"], how="left")
        display(metrics.sort_values(["dbcv_umap", "stability"], ascending=False))
    if theme_path.is_file():
        print("Theme-level median Jaccard stability")
        display(pd.read_csv(theme_path).sort_values(["configuration_id", "theme_stability"], ascending=[True, False]))
        """
    ),
    markdown("## 5. Frozen themes and audit dictionary"),
    code(
        """
topics_path = RUN_DIR / "topics" / "topic_dictionary.csv"
if topics_path.is_file():
    topics = pd.read_csv(topics_path)
    display(topics)
else:
    print("Topic dictionary not found; the discovery run may not have completed selection for every role.")

for role in ROLES:
    assignments_path = RUN_DIR / "clustering" / role / "topic_assignments.csv"
    if assignments_path.is_file():
        assignments = pd.read_csv(assignments_path)
        print(role, "assignments:", assignments.shape)
        display(assignments.head())
        """
    ),
    markdown("## 6. Export locations"),
    code(
        """
outputs = [
    "config_resolved.yaml", "parallel_runtime.json", "pareto_selection_summary.csv",
    "figures/pareto_validation_all_roles.png", "topics/topic_dictionary.csv",
    "clustering/A0/topic_assignments.csv", "clustering/A1/topic_assignments.csv",
    "clustering/B/topic_assignments.csv", "clustering/C/topic_assignments.csv",
]
display(pd.DataFrame({"path": outputs, "exists": [(RUN_DIR / path).exists() for path in outputs]}))
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
