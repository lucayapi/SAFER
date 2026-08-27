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


def build_notebook(dataset_id: str) -> dict:
    cells = [
        markdown(f"""
# Analyse BN des scénarios récurrents — {dataset_id}

Cette analyse commence après la sélection automatique S_R du job discovery
et le notebook `topic_modeling_results_{dataset_id}.ipynb`. Elle ne relance ni
UMAP, ni HDBSCAN, ni le resampling. Les partitions figées sont lues depuis
`selected_configurations.csv`, puis un réseau bayésien latent est ajusté par
Structural EM.
        """),
        code("""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

SCENARIO_DIR = None
for base_dir in (Path.cwd(), *Path.cwd().parents):
    candidates = (
        base_dir / "text" / "recurrent_scenarios",
        base_dir,
    )
    for candidate in candidates:
        if (candidate / "scenario_pipeline.py").is_file():
            SCENARIO_DIR = candidate
            break
    if SCENARIO_DIR is not None:
        break
if SCENARIO_DIR is None:
    raise FileNotFoundError(
        "Impossible de trouver text/recurrent_scenarios/scenario_pipeline.py "
        "depuis le répertoire courant."
    )
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

from scenario_pipeline import (
    build_frozen_bn_inputs,
    extract_latent_bn_scenarios,
    finalize_latent_bn,
    fit_latent_bn_analysis,
    load_selected_configurations,
    load_yaml_config,
    select_dataset_config,
    resolve_config_paths,
    load_units,
    write_final_bn_outputs,
)
from scenario_figures import generate_scenario_figures
        """),
        markdown("""
## 1. Partitions BN figées

Les identifiants viennent de `selected_configurations.csv` (job discovery).
Un override manuel reste possible en éditant `BN_PARTITION_SELECTIONS` après chargement.
        """),
        code(f"""
DATASET_ID = {dataset_id!r}
DISCOVERY_RUN_NAME = "theme_discovery_audit"
RUN_DIR = SCENARIO_DIR / "runs" / DISCOVERY_RUN_NAME / DATASET_ID
BN_OUTPUT_DIR = RUN_DIR / "bayesian_networks"

BN_PARTITION_SELECTIONS = load_selected_configurations(RUN_DIR)

CONFIG_PATH = SCENARIO_DIR / "config.yaml"
config = resolve_config_paths(
    select_dataset_config(load_yaml_config(CONFIG_PATH), DATASET_ID),
    CONFIG_PATH,
)
config["bayesian_networks"]["min_theme_support_count"] = 20
config["bayesian_networks"]["d_max"] = 2
config["bayesian_networks"]["alpha"] = 0.5
config["bayesian_networks"]["show_progress"] = True

print("Dataset:", DATASET_ID)
print("Partitions:", BN_PARTITION_SELECTIONS)
        """),
        markdown("## 2. Matrice accident × facteurs figés"),
        code("""
units, _ = load_units(config)
BN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
matrix, theme_dictionary, excluded_themes, roles = build_frozen_bn_inputs(
    units,
    RUN_DIR,
    BN_PARTITION_SELECTIONS,
    config,
    BN_OUTPUT_DIR,
)
factor_columns = list(theme_dictionary["variable_name"])
print("Dimensions de la matrice :", matrix.shape)
print("Accidents :", len(matrix), "Facteurs inclus :", len(factor_columns))
display(pd.DataFrame({"role": list(roles.values())}).value_counts().rename("n_facteurs").reset_index())
display(theme_dictionary)
display(excluded_themes)
        """),
        markdown("### Visualisation de la matrice multi-hot"),
        code("""
ordered_columns = [
    column for role in ("A0", "A1", "B", "C")
    for column in factor_columns
    if roles[column] == role
]
prevalence = matrix[ordered_columns].mean(axis=0).to_numpy(dtype=float)
heatmap_data = matrix[ordered_columns].to_numpy(dtype=np.int8)
row_order = np.argsort(heatmap_data.sum(axis=1), kind="stable")
max_heatmap_rows = 600
if len(row_order) > max_heatmap_rows:
    selected_rows = np.linspace(0, len(row_order) - 1, max_heatmap_rows, dtype=int)
    row_order = row_order[selected_rows]
heatmap_data = heatmap_data[row_order]

figure_width = max(12, len(ordered_columns) * 0.18)
fig, (prevalence_axis, heatmap_axis) = plt.subplots(
    2,
    1,
    figsize=(figure_width, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [1.4, 6]},
)
factor_positions = np.arange(len(ordered_columns))
prevalence_axis.bar(factor_positions, prevalence * 100, color="#4C78A8", width=0.82)
prevalence_axis.set_ylabel("Prevalence (%)")
prevalence_axis.set_title(f"Accident-level prevalence — {matrix.shape[0]} accidents")
prevalence_axis.set_ylim(0, max(5, float(prevalence.max() * 100 * 1.15)))
prevalence_axis.grid(alpha=0.2, axis="y")
prevalence_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

heatmap_axis.imshow(heatmap_data, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
heatmap_axis.set_xlabel("BN factors (same columns as prevalence barplot)")
heatmap_axis.set_ylabel("Accidents (sampled rows sorted by number of active factors)")
heatmap_axis.set_title(f"Accident × factor matrix — {matrix.shape[0]} × {len(ordered_columns)}")
heatmap_axis.set_xticks(factor_positions)
heatmap_axis.set_xticklabels(ordered_columns, rotation=90, fontsize=7)
heatmap_axis.set_yticks([])
offset = 0
for role in ("A0", "A1", "B", "C"):
    count = sum(roles[column] == role for column in ordered_columns)
    for current_axis in (prevalence_axis, heatmap_axis):
        current_axis.axvline(offset - 0.5, color="tab:red", linewidth=0.8)
    prevalence_axis.text(offset + max(count - 1, 0) / 2, 1.05, role, transform=prevalence_axis.get_xaxis_transform(), ha="center", va="bottom", fontweight="bold")
    offset += count
for current_axis in (prevalence_axis, heatmap_axis):
    current_axis.axvline(len(ordered_columns) - 0.5, color="tab:red", linewidth=0.8)
fig.tight_layout()
fig.savefig(BN_OUTPUT_DIR / "accident_factor_matrix_heatmap.png", dpi=220, bbox_inches="tight")
display(fig)
        """),
        markdown("## 3. Sélection de K par BIC"),
        code("""
selected_result, k_selection = fit_latent_bn_analysis(
    matrix,
    roles,
    config,
    BN_OUTPUT_DIR,
)
display(k_selection.sort_values(["K", "bic"]))
print("K sélectionné:", selected_result.n_states)
        """),
        code("""
plot_data = k_selection.copy()
plot_data["init_rank"] = plot_data.groupby("K").cumcount()
plot_data["n_initializations_K"] = plot_data.groupby("K")["K"].transform("size")
candidate_k = sorted(plot_data["K"].unique())
k_positions = {value: index for index, value in enumerate(candidate_k)}
plot_data["x_position"] = plot_data["K"].map(k_positions)
plot_data["x_plot"] = plot_data["x_position"] + (
    plot_data["init_rank"] - (plot_data["n_initializations_K"] - 1) / 2
) * 0.035

fig, (all_axis, best_axis) = plt.subplots(1, 2, figsize=(15, 6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
admissible = plot_data[plot_data["admissible"]]
inadmissible = plot_data[~plot_data["admissible"]]
all_axis.scatter(inadmissible["x_plot"], inadmissible["bic"], s=20, color="#D9D9D9", alpha=0.35, label="Non-admissible")
all_axis.scatter(admissible["x_plot"], admissible["bic"], s=28, color="#4C78A8", alpha=0.55, label="Admissible initialisations")

best_per_k = plot_data[plot_data["selected_for_K"]].sort_values("K")
all_axis.plot(best_per_k["x_position"], best_per_k["bic"], color="#1F4E79", linewidth=2.5, marker="o", markersize=7, label="Best admissible BIC")
best_axis.plot(best_per_k["x_position"], best_per_k["bic"], color="#1F4E79", linewidth=3, marker="o", markersize=9, label="Best admissible BIC")

selected_k = plot_data[plot_data["selected_final"]].copy()
selected_position = int(k_positions[selected_result.n_states])
selected_k["K"] = selected_k["K"].map(k_positions)
for current_axis in (all_axis, best_axis):
    current_axis.axvspan(selected_position - 0.18, selected_position + 0.18, color="#D62728", alpha=0.08, zorder=0)
    current_axis.set_xticks(range(len(candidate_k)))
    current_axis.set_xticklabels([str(value) for value in candidate_k])
    current_axis.set_xlabel("Latent cardinality K")
    current_axis.grid(alpha=0.25, axis="y")
best_axis.scatter(selected_k["K"], selected_k["bic"], s=180, marker="*", color="#D62728", edgecolor="black", linewidth=0.7, zorder=5, label=f"K⋆ = {selected_result.n_states}")
all_axis.set_ylabel("Observed-data BIC")
all_axis.set_title("All Structural EM initializations")
best_axis.set_title("Best admissible solution by K")
all_axis.legend(loc="best", frameon=False)
best_axis.legend(loc="best", frameon=False)
best_axis.annotate("Selected K", (selected_position, float(selected_k["bic"].iloc[0])), xytext=(8, 14), textcoords="offset points", color="#B22222", fontweight="bold")
for _, row in best_per_k.iterrows():
    best_axis.annotate(f"{row['bic']:.1f}", (row["x_position"], row["bic"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
fig.suptitle("Observed-data BIC across candidate latent cardinalities", fontsize=15, fontweight="bold")
fig.text(0.5, 0.01, "Small points: individual initialisations. Large connected points: best admissible solution for each K. Lower BIC is preferred.", ha="center", fontsize=9, color="#555555")
fig.tight_layout()
fig.subplots_adjust(bottom=0.13, top=0.86)
fig.savefig(BN_OUTPUT_DIR / "bic_by_k_initializations.png", dpi=220, bbox_inches="tight")
display(fig)
        """),
        markdown("## 4. Familles latentes, profils et scénarios"),
        code("""
final_result = finalize_latent_bn(selected_result, matrix, roles, config)
write_final_bn_outputs(final_result, BN_OUTPUT_DIR, theme_dictionary)
scenarios, supports, prototypes, profiles = extract_latent_bn_scenarios(
    final_result,
    matrix,
    roles,
    units,
    config,
    BN_OUTPUT_DIR,
    theme_dictionary,
)
analysis = {
    "matrix": matrix,
    "theme_dictionary": theme_dictionary,
    "excluded_themes": excluded_themes,
    "selection": k_selection,
    "result": final_result,
    "scenarios": scenarios,
    "supports": supports,
    "prototypes": prototypes,
    "profiles": profiles,
}
print("BN finalise :", len(final_result.nodes), "facteurs,", len(final_result.edges), "arcs")
display(analysis["profiles"].head(50))
display(analysis["scenarios"])
display(analysis["supports"])
display(analysis["prototypes"])
        """),
        code("""
scenario_figure_table = scenarios.copy()
scenario_figure_table["scenario_id"] = scenario_figure_table["family_id"].map(lambda value: f"S{int(value)}")
scenario_figure_table["latent_family"] = scenario_figure_table["family_id"].map(lambda value: str(int(value)))
scenario_figure_table["heading"] = scenario_figure_table.apply(
    lambda row: " → ".join(
        str(row[column]) for column in ("A0_labels", "A1_labels", "B_labels", "C_labels")
        if str(row.get(column, "")).strip()
    ),
    axis=1,
)
scenario_figure_table["factor_codes"] = scenario_figure_table.apply(
    lambda row: ";".join(
        str(row[column]) for column in ("A0_factors", "A1_factors", "B_factors", "C_factors")
        if str(row.get(column, "")).strip()
    ),
    axis=1,
)
scenario_figure_table = scenario_figure_table.rename(columns={
    "A0_labels": "A0_label",
    "A1_labels": "A1_label",
    "B_labels": "B_label",
    "C_labels": "C_label",
    "family_positive_support": "family_support",
    "global_positive_support": "global_support",
})
scenario_figure_dir = BN_OUTPUT_DIR / "scenario_figures"
generate_scenario_figures(scenario_figure_table, scenario_figure_dir, learned_edges=final_result.edges)
print("Figures individuelles :", scenario_figure_dir)
display(scenario_figure_table[["scenario_id", "latent_family", "family_support", "global_support", "heading", "factor_codes"]])
        """),
        code("""
from IPython.display import Image
for figure_name in (
    "conceptual_bn_architecture.png",
    "learned_bn_simplified.png",
    "recurrent_scenarios_graph.png",
):
    figure_path = BN_OUTPUT_DIR / figure_name
    if figure_path.exists():
        display(Image(filename=str(figure_path)))
        """),
        code("""
profiles = analysis["profiles"].pivot(index="variable_name", columns="family_id", values="probability")
fig, axis = plt.subplots(figsize=(10, max(5, len(profiles) * 0.22)))
image = axis.imshow(profiles.fillna(0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
axis.set(yticks=np.arange(len(profiles)), yticklabels=profiles.index, xlabel="Latent family", title=r"Factor profiles $P(X=1 \\mid Z)$")
fig.colorbar(image, ax=axis, label="Probability")
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
    for dataset_id in ("caou", "btp", "metallurgie"):
        path = NOTEBOOK_DIR / f"recurrent_scenarios_bn_analysis_{dataset_id}.ipynb"
        path.write_text(json.dumps(build_notebook(dataset_id), indent=1, ensure_ascii=False), encoding="utf-8")
    old_path = NOTEBOOK_DIR / "recurrent_scenarios_bn_analysis.ipynb"
    if old_path.exists():
        old_path.unlink()


if __name__ == "__main__":
    main()
