"""Generate one frozen-theme latent-BN notebook per dataset."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


DEFAULT_PARTITIONS = {
    "caou": {"A0": "A0_cfg_005", "A1": "A1_cfg_009", "B": "B_cfg_016", "C": "C_cfg_035"},
    "metallurgie": {"A0": "A0_cfg_007", "A1": "A1_cfg_010", "B": "B_cfg_011", "C": "C_cfg_019"},
}


def build_notebook(dataset_id: str) -> dict:
    partitions = DEFAULT_PARTITIONS[dataset_id]
    cells = [
        markdown(f"""
# Analyse BN des scénarios récurrents — {dataset_id}

Cette analyse commence après la sélection des partitions Pareto dans le
notebook `topic_modeling_results_{dataset_id}.ipynb`. Elle ne relance ni UMAP,
ni HDBSCAN, ni le resampling. Une configuration Pareto est choisie explicitement
pour chaque rôle, puis un seul réseau bayésien latent est ajusté par Structural EM.
        """),
        code("""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

SCENARIO_DIR = Path.cwd() / "text" / "recurrent_scenarios"
if not (SCENARIO_DIR / "scenario_pipeline.py").is_file():
    SCENARIO_DIR = Path.cwd()
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import (
    load_yaml_config,
    select_dataset_config,
    resolve_config_paths,
    load_units,
    run_frozen_bn_analysis,
)
        """),
        markdown("""
## 1. Configuration des partitions BN

Modifie uniquement les identifiants ci-dessous pour analyser d'autres
partitions Pareto. Il faut exactement une configuration par rôle. Ces choix
n'entraînent aucune nouvelle étape de clustering.
        """),
        code(f"""
DATASET_ID = {dataset_id!r}
DISCOVERY_RUN_NAME = "theme_discovery_audit"
RUN_DIR = SCENARIO_DIR / "runs" / DISCOVERY_RUN_NAME / DATASET_ID
BN_OUTPUT_DIR = RUN_DIR / "bayesian_networks"

BN_PARTITION_SELECTIONS = {partitions!r}

CONFIG_PATH = SCENARIO_DIR / "config.yaml"
config = resolve_config_paths(
    select_dataset_config(load_yaml_config(CONFIG_PATH), DATASET_ID),
    CONFIG_PATH,
)
config["bayesian_networks"]["min_theme_support_count"] = 20
config["bayesian_networks"]["d_max"] = 2
config["bayesian_networks"]["latent_states"] = list(range(2, 9))
config["bayesian_networks"]["n_initializations"] = 20
config["bayesian_networks"]["alpha"] = 0.5

print("Dataset:", DATASET_ID)
print("Partitions:", BN_PARTITION_SELECTIONS)
        """),
        markdown("## 2. Matrice accident × facteurs figés"),
        code("""
units, _ = load_units(config)
analysis = run_frozen_bn_analysis(
    config=config,
    run_dir=RUN_DIR,
    partition_selections=BN_PARTITION_SELECTIONS,
    output_dir=BN_OUTPUT_DIR,
    units=units,
)

matrix = analysis["matrix"]
theme_dictionary = analysis["theme_dictionary"]
excluded_themes = analysis["excluded_themes"]
print("Accidents:", len(matrix), "Variables BN:", len(theme_dictionary))
display(theme_dictionary)
display(excluded_themes)
        """),
        markdown("## 3. Sélection de K par BIC"),
        code("""
k_selection = analysis["selection"]
display(k_selection.sort_values(["K", "bic"]))
selected_result = analysis["result"]
print("K sélectionné:", selected_result.n_states)
        """),
        code("""
fig, axis = plt.subplots(figsize=(8, 5))
summary = k_selection.groupby("K", as_index=False)["bic"].min()
axis.plot(summary["K"], summary["bic"], marker="o")
axis.set(xlabel="Nombre de familles latentes K", ylabel="BIC", title="Sélection de K par BIC")
axis.grid(alpha=0.25)
fig.tight_layout()
display(fig)
        """),
        markdown("## 4. Familles latentes, profils et scénarios"),
        code("""
display(analysis["profiles"].head(50))
display(analysis["scenarios"])
display(analysis["supports"])
display(analysis["prototypes"])
        """),
        code("""
profiles = analysis["profiles"].pivot(index="variable_name", columns="family_id", values="probability")
fig, axis = plt.subplots(figsize=(10, max(5, len(profiles) * 0.22)))
image = axis.imshow(profiles.fillna(0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
axis.set(yticks=np.arange(len(profiles)), yticklabels=profiles.index, xlabel="Famille latente", title="Profils factoriels P(X=1 | Z)")
fig.colorbar(image, ax=axis, label="Probabilité")
fig.tight_layout()
display(fig)
        """),
        markdown("""
## Fichiers produits

Les résultats sont écrits dans `RUN_DIR / "bayesian_networks"`, notamment la
matrice multi-hot, les responsabilités postérieures, les profils familiaux,
les CPT finales, les arêtes apprises, les supports et les prototypes observés.
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
    for dataset_id in ("caou", "metallurgie"):
        path = NOTEBOOK_DIR / f"recurrent_scenarios_bn_analysis_{dataset_id}.ipynb"
        path.write_text(json.dumps(build_notebook(dataset_id), indent=1, ensure_ascii=False), encoding="utf-8")
    old_path = NOTEBOOK_DIR / "recurrent_scenarios_bn_analysis.ipynb"
    if old_path.exists():
        old_path.unlink()


if __name__ == "__main__":
    main()
