"""Generate result notebooks for the exact global BN analysis."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
BTP_FAMILY_DATASETS = (
    "btp_building_construction_and_masonry",
    "btp_building_finishing_and_insulation",
    "btp_carpentry_and_joinery",
    "btp_civil_engineering_and_networks",
    "btp_construction_equipment_with_operator",
    "btp_earthworks_foundations_and_demolition",
    "btp_electrical_installation",
    "btp_engineering_and_project_support",
    "btp_metal_construction_and_assembly",
    "btp_plumbing_heating_and_hvac",
    "btp_roofing_and_waterproofing",
)

DATASETS = (
    "caou",
    "btp",
    "metallurgie",
    "caou_plastics_manufacturing",
    "metallurgie_metal_forming_and_fabrication",
    *BTP_FAMILY_DATASETS,
)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


def build_notebook(dataset_id: str) -> dict:
    cells = [
        markdown(f"""
# Exact role-constrained Bayesian network — {dataset_id}

This notebook reports the exact A0 → A1 → B → C Bayesian network and the
empirical recurrent-scenario analysis. Scenario selection remains independent
of the Bayesian network. Edge directions follow the predefined role ordering
and are not causal directions inferred from the data.
        """),
        code("""
from pathlib import Path
import sys
import warnings
import pandas as pd
from IPython.display import Image, display

warnings.filterwarnings(
    "ignore",
    message=r"(?s).*support for the `google[.]generativeai` package has ended.*",
    category=FutureWarning,
)
SCENARIO_DIR = next(
    candidate
    for base in (Path.cwd(), *Path.cwd().parents)
    for candidate in (base / "text" / "recurrent_scenarios", base)
    if (candidate / "scenario_pipeline.py").is_file()
)
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))
from scenario_analysis import _output_paths, run_global_bn_scenario_mining
from scenario_pipeline import load_bn_analysis_config, load_selected_configurations
        """),
        code(f"""
DATASET_ID = {dataset_id!r}
RUN_DIR = SCENARIO_DIR / "runs" / "theme_discovery_audit" / DATASET_ID
CONFIG_PATH = SCENARIO_DIR / "config.yaml"
RUN_ANALYSIS = False  # Use the Slurm job for the 500-replicate bootstrap.
config = load_bn_analysis_config(CONFIG_PATH, DATASET_ID, RUN_DIR)
selections = load_selected_configurations(RUN_DIR)
paths = _output_paths(RUN_DIR)
print("Dataset:", DATASET_ID)
print("Output:", paths["root"])
print("Structure optimization: exact within the role-constrained graph class")
print("Bootstrap resamples:", config["bayesian_networks"]["bn_structure_bootstrap"]["n_resamples"])
print("Selected partitions:", selections)
        """),
        markdown("## 1. Accident-level matrix and exact structure"),
        code("""
if RUN_ANALYSIS:
    analysis = run_global_bn_scenario_mining(config, RUN_DIR, selections)
if not (paths["network"] / "global_bn_summary.csv").is_file():
    raise FileNotFoundError("Exact BN outputs are absent. Launch jobs/run_recurrent_scenarios_bn_exact.sh first.")
summary = pd.read_csv(paths["network"] / "global_bn_summary.csv")
local_scores = pd.read_csv(paths["network"] / "bn_local_scores.csv")
selected_parent_sets = pd.read_csv(paths["network"] / "bn_selected_parent_sets.csv")
display(summary)
display(selected_parent_sets)
summary_row = summary.iloc[0]
print("Variables:", int(summary_row["n_variables"]))
print("Local parent sets evaluated:", int(summary_row["n_local_parent_sets_evaluated"]))
print("Selected edges:", int(summary_row["n_edges"]))
print("Parameters:", int(summary_row["n_parameters"]))
print("Log-likelihood:", float(summary_row["log_likelihood"]))
print("BIC:", float(summary_row["BIC"]))
        """),
        markdown("## 2. Assess support of selected conditional-probability tables"),
        code("""
display(local_scores.sort_values(["child_factor", "rank"]).head(30))
support_diagnostics = pd.read_csv(paths["network"] / "bn_cpt_support_diagnostics.csv")
diagnostic = support_diagnostics.iloc[0]
print("Selected BN:")
print("CPT rows:", diagnostic["n_cpt_rows"])
print("Empty rows (N = 0):", diagnostic["n_rows_N_eq_0"])
print("Rows with N <= 5:", diagnostic["n_rows_N_le_5"])
print("Rows with N <= 10:", diagnostic.get("n_rows_N_le_10", "available after the next exact-BN run"))
print("Rows with N <= 20:", diagnostic.get("n_rows_N_le_20", "available after the next exact-BN run"))
print("MLE estimates equal to 0:", diagnostic["n_MLE_equal_0"])
print("MLE estimates equal to 1:", diagnostic["n_MLE_equal_1"])
print("Minimum observed cell count:", diagnostic["minimum_observed_cell_count"])
print("Median observed cell count:", diagnostic["median_observed_cell_count"])
print("Final CPT estimator:", diagnostic["final_CPT_estimation"])
display(support_diagnostics)
final_cpts = pd.read_csv(paths["network"] / "bn_final_cpts.csv")
display(final_cpts.head(30))
if diagnostic["final_CPT_estimation"] == "Jeffreys":
    print("Jeffreys regularization was used because the configured support criterion was not met.")
    display(pd.read_csv(paths["network"] / "bn_mle_cpts.csv").head(30))
        """),
        markdown("## 3. Structural bootstrap and conditional contrasts"),
        code("""
bootstrap = pd.read_csv(paths["network"] / "bn_bootstrap_edges.csv")
edges = pd.read_csv(paths["network"] / "bn_edges_full.csv")
contrasts = pd.read_csv(paths["network"] / "bn_conditional_contrasts.csv")
stable_threshold = config["bayesian_networks"]["bn_display_bootstrap_threshold"]
stable = edges.loc[edges["bootstrap_frequency"] >= stable_threshold].copy()
print(f"{len(edges)} full-sample edges; {len(stable)} with f >= {stable_threshold:.2f}")
print("Stable edges by transition:")
display(stable["transition"].value_counts().rename_axis("transition").reset_index(name="n_edges"))
main_columns = [
    "parent_label", "child_label", "transition", "bootstrap_frequency",
    "conditional_contrast_weighted", "conditional_contrast_min",
    "conditional_contrast_max", "conditional_contrast_pattern",
]
display(stable[[column for column in main_columns if column in stable.columns]].sort_values("bootstrap_frequency", ascending=False))
display(contrasts.head(30))
for name in ("global_bn_stable_dependencies.png", "bn_stability_vs_conditional_contrast.png"):
    path = paths["figures"] / name
    if path.is_file():
        display(Image(filename=str(path)))
        """),
        markdown("## 4. Empirical recurrent scenarios"),
        code("""
candidates = pd.read_csv(paths["scenarios"] / "scenario_candidates_all.csv")
recurrent = pd.read_csv(paths["scenarios"] / "recurrent_scenarios_all.csv")
thresholds = pd.read_csv(paths["scenarios"] / "scenario_threshold_summary.csv")
print("Admissible configurations:", len(candidates))
print("Closed recurrent scenarios:", len(recurrent))
display(thresholds)
display(recurrent[["scenario_id", "upstream_labels", "B_label", "C_label", "scenario_accident_count", "scenario_support", "confidence", "lift"]].head(20))
        """),
        markdown("## 5. Recurrence relative to the Bayesian-network background"),
        code("""
discrepancy = pd.read_csv(paths["scenarios"] / "scenario_bn_discrepancy.csv")
sensitivity = pd.read_csv(paths["scenarios"] / "scenario_bn_cpt_sensitivity.csv")
article = pd.read_csv(paths["scenarios"] / "scenarios_article_table.csv")
signed = discrepancy["BN_support_discrepancy"].dropna()
print("Scenarios with D_BN > 0:", int((signed > 0).sum()))
print("Scenarios with D_BN < 0:", int((signed < 0).sum()))
print("Median |D_BN| (pp):", 100.0 * float(signed.abs().median()))
print("Maximum |D_BN| (pp):", 100.0 * float(signed.abs().max()))
display(discrepancy.sort_values("BN_support_interestingness", ascending=False).head(20))
display(sensitivity[["scenario_id", "BN_CPT_estimation", "BN_MLE_support", "BN_Jeffreys_support", "BN_MLE_minus_Jeffreys_pp"]].head(20))
display(article)
path = paths["figures"] / "observed_vs_bn_implied_recurrence.png"
if path.is_file():
    display(Image(filename=str(path)))
        """),
        markdown("## 6. Files for the manuscript"),
        code("""
for folder in (paths["network"], paths["scenarios"], paths["figures"]):
    print(folder)
    for path in sorted(folder.glob("*")):
        if path.is_file():
            print("  ", path.name)
        """),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id in DATASETS:
        path = NOTEBOOK_DIR / f"recurrent_scenarios_bn_analysis_{dataset_id}.ipynb"
        path.write_text(json.dumps(build_notebook(dataset_id), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
