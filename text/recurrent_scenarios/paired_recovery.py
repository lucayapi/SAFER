"""Paired accident subsampling for fixed and reconstructed semantic factors.

Run from text/.  Every replicate uses the same distinct accidents in both
branches.  The script deliberately does not repeat hyperparameter selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from global_bn import _exact_structure_learning
from scenario_pipeline import (
    ROLES,
    _fit_cluster_with_embedding,
    load_bn_analysis_config,
    load_embeddings,
    load_selected_configurations,
    load_units,
    sha256_file,
)

DEFAULT_DATASET = "btp_carpentry_and_joinery"
REFERENCE_MIN_COUNT = 5
PARAMETER_POLICIES = ("fixed", "scaled_cluster_size", "scaled_density")
ARTICLE_SCENARIOS = {"S1": "SC_00197", "S2": "SC_00221",
                     "S3": "SC_04076", "S4": "SC_00211"}
ARTICLE_DESCRIPTIONS = {
    "S1": "Woodworking machine tasks; wood kickback; finger or hand amputation",
    "S2": "Missing or ineffective machine guards; wood kickback; finger or hand amputation",
    "S3": "Roof-covering work; fragile roofing material; fall from height; fatal consequence",
    "S4": "Incomplete or unsecured scaffold; fall from height; multiple fractures",
}


def _factor_name(role: str, topic_id: str) -> str:
    label = int(str(topic_id).rsplit("_", 1)[1])
    return f"{role}__T{label + 1:02d}"


def _selected_parameters(run_dir: Path) -> dict[str, dict]:
    selected = load_selected_configurations(run_dir)
    table = pd.read_csv(run_dir / "selected_configurations_summary.csv")
    result = {}
    keys = ("umap_n_neighbors", "umap_n_components", "umap_min_dist",
            "hdbscan_min_cluster_size", "hdbscan_min_samples",
            "hdbscan_cluster_selection_method")
    for role in ROLES:
        rows = table.loc[table["role"].eq(role) & table["configuration_id"].eq(selected[role])]
        if len(rows) != 1:
            raise ValueError(f"Exactly one selected configuration is required for {role}")
        row = rows.iloc[0]
        result[role] = {key: row[key].item() if isinstance(row[key], np.generic) else row[key]
                        for key in keys}
    return result


def paired_recovery_settings(config: dict) -> dict:
    """Read and validate the prespecified paired-recovery design."""
    settings = dict(config.get("paired_recovery", {}))
    n_replicates = int(settings.get("n_replicates", 500))
    fraction = float(settings.get("resampling_fraction", 0.80))
    seed = int(settings.get("random_state", 2026))
    thresholds = tuple(float(value) for value in settings.get("matching_thresholds", [0.40, 0.50, 0.60]))
    main_threshold = float(settings.get("main_matching_threshold", 0.50))
    workers = settings.get("n_workers", "auto")
    if n_replicates < 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid paired_recovery n_replicates or fraction")
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("matching_thresholds must contain values in (0, 1]")
    if len(set(thresholds)) != len(thresholds) or main_threshold not in thresholds:
        raise ValueError("matching thresholds must be distinct and include main_matching_threshold")
    return {"n_replicates": n_replicates, "fraction": fraction, "seed": seed,
            "thresholds": thresholds, "main_threshold": main_threshold,
            "n_workers": workers}


def resolve_worker_count(settings: dict, requested: int | None = None) -> int:
    """Use the Slurm CPU allocation unless a worker count is explicitly set."""
    value = settings["n_workers"] if requested is None else requested
    if value in (None, "", "auto"):
        value = os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1)
    workers = int(value)
    if workers < 1:
        raise ValueError("n_workers must be at least one")
    return workers


def _reference_assignments(units: pd.DataFrame, run_dir: Path, matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for role in ROLES:
        path = run_dir / "discovery" / role / "selected" / "topic_assignments.csv"
        assignments = pd.read_csv(path, dtype={"fact_id": str, "accident_id": str})
        if assignments["fact_id"].duplicated().any():
            raise ValueError(f"Duplicate fact_id in {path}")
        role_units = units.loc[units["_role"].eq(role), ["_fact_id", "_accident_id"]].copy()
        role_units = role_units.merge(assignments[["fact_id", "topic_id"]],
                                      left_on="_fact_id", right_on="fact_id", validate="one_to_one")
        if len(role_units) != int(units["_role"].eq(role).sum()):
            raise ValueError(f"Reference assignments do not cover role {role}")
        role_units["reference_factor"] = role_units["topic_id"].map(
            lambda value: _factor_name(role, value) if isinstance(value, str) and value else "")
        for factor in role_units["reference_factor"].unique():
            if factor and factor not in matrix.columns:
                raise ValueError(f"Reference factor absent from matrix: {factor}")
        result[role] = role_units[["_fact_id", "_accident_id", "reference_factor"]]
    # Check that the archived assignments reproduce the archived binary matrix.
    for role, frame in result.items():
        for factor, subset in frame.loc[frame["reference_factor"].ne("")].groupby("reference_factor"):
            found = set(subset["_accident_id"])
            archived = set(matrix.loc[matrix[factor].eq(1), "accident_id"].astype(str))
            if found != archived:
                raise ValueError(f"Selected assignments disagree with reference matrix: {role}/{factor}")
    return result


def match_factors(reference: pd.DataFrame, candidate_labels: np.ndarray, role: str,
                  reference_names: list[str]) -> dict[str, dict]:
    """One-to-one maximum-total-Jaccard matching, restricted to sampled units."""
    labels = np.asarray(candidate_labels, dtype=int)
    if len(reference) != len(labels):
        raise ValueError("Reference/candidate unit count mismatch")
    candidates = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    ref_values = reference["reference_factor"].to_numpy()
    weights = np.zeros((len(reference_names), len(candidates)), dtype=float)
    observable = {name: int(np.sum(ref_values == name)) for name in reference_names}
    for i, name in enumerate(reference_names):
        ref_mask = ref_values == name
        for j, label in enumerate(candidates):
            candidate_mask = labels == label
            union = int(np.sum(ref_mask | candidate_mask))
            weights[i, j] = float(np.sum(ref_mask & candidate_mask) / union) if union else 0.0
    matched = {}
    if weights.size:
        row_ids, column_ids = linear_sum_assignment(weights, maximize=True)
        matched = {int(i): int(j) for i, j in zip(row_ids, column_ids)}
    return {
        name: {
            "candidate": f"{role}__R{candidates[matched[i]]:03d}" if i in matched and weights[i, matched[i]] > 0 else None,
            "jaccard": float(weights[i, matched[i]]) if i in matched else 0.0,
            "n_reference_units": observable[name],
        }
        for i, name in enumerate(reference_names)
    }


def _matrix_for_candidates(units: pd.DataFrame, role_labels: dict[str, np.ndarray],
                           accident_ids: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    matrix = pd.DataFrame({"accident_id": accident_ids})
    roles = {}
    for role in ROLES:
        role_units = units.loc[units["_role"].eq(role)].reset_index(drop=True)
        labels = role_labels[role]
        for label in sorted(int(x) for x in np.unique(labels) if int(x) >= 0):
            name = f"{role}__R{label:03d}"
            present = set(role_units.loc[labels == label, "_accident_id"])
            matrix[name] = matrix["accident_id"].isin(present).astype(np.int8)
            roles[name] = role
    return matrix, roles


def _edges(matrix: pd.DataFrame, roles: dict[str, str], d_max: int) -> set[tuple[str, str]]:
    nodes = [name for name in matrix.columns if name != "accident_id"]
    if not nodes:
        return set()
    data = matrix[nodes].to_numpy(dtype=np.int8)
    learned, _, _ = _exact_structure_learning(data, nodes, roles, d_max)
    return set(learned)


def _scenario_count(matrix: pd.DataFrame, factors: list[str]) -> int:
    if not factors or any(name not in matrix.columns for name in factors):
        return 0
    return int(matrix[factors].to_numpy(dtype=np.int8).all(axis=1).sum())


def _scenarios(path: Path, reference_names: set[str]) -> list[dict]:
    table = pd.read_csv(path)
    if "is_closed_pattern" in table.columns:
        table = table.loc[table["is_closed_pattern"].astype(str).str.lower().eq("true")]
    table = table.loc[table["scenario_accident_count"].ge(REFERENCE_MIN_COUNT)]
    result = []
    for row in table.itertuples(index=False):
        upstream = [part.strip() for part in str(row.upstream_factor_ids).split("|") if part.strip()]
        factors = upstream + [str(row.B_factor_id), str(row.C_factor_id)]
        if not set(factors).issubset(reference_names):
            raise ValueError(f"Unknown factor in scenario {row.scenario_id}")
        result.append({"id": str(row.scenario_id), "factors": factors,
                       "full_count": int(row.scenario_accident_count),
                       "confidence": float(row.confidence), "lift": float(row.lift),
                       "label": " | ".join([str(row.upstream_labels), str(row.B_label), str(row.C_label)])})
    if not result or len({item["id"] for item in result}) != len(result):
        raise ValueError("Reference scenario IDs must be nonempty and unique")
    return result


def reconstruction_parameters(parameters: dict, n_subset: int, n_full: int, policy: str) -> dict:
    """Change absolute HDBSCAN sizes using the retained role-specific unit fraction.

    scaled_cluster_size changes only the minimum cluster size; scaled_density
    additionally changes min_samples. UMAP settings and reference fits stay fixed.
    """
    if policy not in PARAMETER_POLICIES or n_full < 1 or not 0 <= n_subset <= n_full:
        raise ValueError("Invalid reconstruction parameter policy or unit counts")
    result = dict(parameters)
    if policy != "fixed":
        fraction = n_subset / n_full
        result["hdbscan_min_cluster_size"] = max(2, int(np.floor(
            float(parameters["hdbscan_min_cluster_size"]) * fraction + 0.5)))
        if policy == "scaled_density":
            value = parameters.get("hdbscan_min_samples")
            if value is None:
                raise ValueError("scaled_density requires explicit selected min_samples")
            result["hdbscan_min_samples"] = max(1, int(np.floor(float(value) * fraction + 0.5)))
    return result


def _design(config_path: Path, run_dir: Path, config: dict, replicates: int,
            fraction: float, seed: int, thresholds: tuple[float, ...], main_threshold: float,
            parameter_policy: str = "fixed") -> dict:
    if parameter_policy not in PARAMETER_POLICIES:
        raise ValueError(f"Unknown parameter policy: {parameter_policy}")
    paths = {
        "config": config_path,
        "selected": run_dir / "selected_configurations_summary.csv",
        "selected_ids": run_dir / "selected_configurations.csv",
        "resolved_config": run_dir / "config_resolved.yaml",
        "reference_matrix": run_dir / "bn_results_exact/matrix/accident_factor_matrix.parquet",
        "reference_roles": run_dir / "bn_results_exact/matrix/variable_roles.json",
        "reference_scenarios": run_dir / "bn_results_exact/scenarios/recurrent_scenarios_all.csv",
        "reference_edges": run_dir / "bn_results_exact/network/bn_edges_full.csv",
        "units": Path(config["data"]["units_path"]),
        "embeddings": Path(config["data"]["embeddings_path"]),
        **{f"assignments_{role}": run_dir / "discovery" / role / "selected/topic_assignments.csv"
           for role in ROLES},
    }
    result = {"replicates": replicates, "fraction": fraction, "seed": seed,
            "matching_thresholds": list(thresholds), "main_matching_threshold": main_threshold,
            "reference_min_count": REFERENCE_MIN_COUNT,
            "d_max": int(config["bayesian_networks"]["d_max"]),
            "files_sha256": {key: sha256_file(path) for key, path in paths.items()}}
    # Preserve the design representation of historical fixed-parameter runs.
    if parameter_policy != "fixed":
        result["parameter_policy"] = parameter_policy
        result["parameter_scaling"] = "retained role units / full role units; round half up"
    return result


def _fingerprint(design: dict) -> str:
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_RECOVERY_CONTEXT: dict | None = None


def _run_one_replicate(replicate: int) -> tuple[int, int]:
    """Run one independent replicate using the forked read-only worker state."""
    if _RECOVERY_CONTEXT is None:
        raise RuntimeError("Paired-recovery worker context is unavailable")
    context = _RECOVERY_CONTEXT
    output = context["output"]
    fingerprint = context["fingerprint"]
    path = output / "replicates" / f"replicate_{replicate:04d}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("design_hash") != fingerprint:
            raise ValueError(f"Stale replicate output: {path}")
        return replicate, -1
    rng = np.random.default_rng(np.random.SeedSequence([context["seed"], replicate]))
    sampled = sorted(str(x) for x in rng.choice(context["accident_ids"], size=context["n_sample"], replace=False))
    sampled_set = set(sampled)
    units = context["units"]
    subset_mask = units["_accident_id"].isin(sampled_set).to_numpy()
    subset_units = units.loc[subset_mask].reset_index(drop=True)
    subset_embeddings = context["embeddings"][subset_mask]
    matrix = context["matrix"]
    roles = context["roles"]
    d_max = context["d_max"]
    fixed_matrix = matrix.set_index("accident_id").loc[sampled].reset_index()
    fixed_edges = _edges(fixed_matrix, roles, d_max)
    refit_labels = {}
    matching = {}
    fitted_parameters = {}
    for role in ROLES:
        role_mask = subset_units["_role"].eq(role).to_numpy()
        role_units = subset_units.loc[role_mask].reset_index(drop=True)
        fitted_parameters[role] = reconstruction_parameters(
            context["parameters"][role], len(role_units), len(context["assignments"][role]),
            context.get("parameter_policy", "fixed"))
        labels, _ = _fit_cluster_with_embedding(
            role_units["_text"].tolist(), subset_embeddings[role_mask], fitted_parameters[role],
            context["umap_seed"], context["config"],
        )
        refit_labels[role] = labels
        reference = context["assignments"][role].set_index("_fact_id").loc[role_units["_fact_id"]].reset_index()
        reference_names = [name for name, assigned_role in roles.items() if assigned_role == role]
        matching.update(match_factors(reference, labels, role, reference_names))
    reconstructed_matrix, reconstructed_roles = _matrix_for_candidates(subset_units, refit_labels, sampled)
    reconstructed_edges = _edges(reconstructed_matrix, reconstructed_roles, d_max)
    scenario_results = {}
    for scenario in context["scenarios"]:
        fixed_count = _scenario_count(fixed_matrix, scenario["factors"])
        reconstructed_counts = {}
        for tau in context["thresholds"]:
            names = [matching[name]["candidate"] if matching[name]["jaccard"] >= tau else None
                     for name in scenario["factors"]]
            reconstructed_counts[str(tau)] = _scenario_count(reconstructed_matrix, names)
        scenario_results[scenario["id"]] = {
            "fixed_count": fixed_count, "reconstructed_counts": reconstructed_counts}
    record = {"replicate": replicate, "design_hash": fingerprint, "accident_ids": sampled,
              "parameter_policy": context.get("parameter_policy", "fixed"),
              "fitted_parameters": fitted_parameters, "umap_seed": context["umap_seed"],
              "sample_size": context["n_sample"], "matches": matching,
              "fixed_edges": sorted([list(x) for x in fixed_edges]),
              "reconstructed_edges": sorted([list(x) for x in reconstructed_edges]),
              "n_reconstructed_factors": {role: sum(value == role for value in reconstructed_roles.values())
                                          for role in ROLES},
              "scenarios": scenario_results}
    _write_json(path, record)
    return replicate, len(reconstructed_edges)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    config = load_bn_analysis_config(config_path, args.dataset, run_dir)
    settings = paired_recovery_settings(config)
    replicates = settings["n_replicates"] if args.replicates is None else args.replicates
    fraction = settings["fraction"] if args.fraction is None else args.fraction
    seed = settings["seed"] if args.seed is None else args.seed
    thresholds = settings["thresholds"]
    main_threshold = settings["main_threshold"]
    count = (replicates - args.start) if args.count is None else args.count
    workers = resolve_worker_count(settings, args.workers)
    if not 0 < fraction <= 1 or replicates < 1 or args.start < 0 or count < 1:
        raise ValueError("Invalid subsampling settings")
    if args.start >= replicates:
        return
    parameters = _selected_parameters(run_dir)
    units, _ = load_units(config)
    embeddings = load_embeddings(config, units, output / "embedding_cache")
    matrix = pd.read_parquet(run_dir / "bn_results_exact/matrix/accident_factor_matrix.parquet")
    matrix["accident_id"] = matrix["accident_id"].astype(str)
    accident_ids = sorted(units["_accident_id"].unique().tolist())
    if len(matrix) != len(accident_ids) or set(matrix["accident_id"]) != set(accident_ids):
        raise ValueError("Reference matrix and source units contain different accidents")
    matrix = matrix.set_index("accident_id").loc[accident_ids].reset_index()
    roles = json.loads((run_dir / "bn_results_exact/matrix/variable_roles.json").read_text(encoding="utf-8"))
    reference_names = set(roles)
    assignments = _reference_assignments(units, run_dir, matrix)
    scenarios = _scenarios(run_dir / "bn_results_exact/scenarios/recurrent_scenarios_all.csv", reference_names)
    edges_file = pd.read_csv(run_dir / "bn_results_exact/network/bn_edges_full.csv")
    reference_edges = sorted((str(row.parent_factor), str(row.child_factor))
                             for row in edges_file.itertuples(index=False)
                             if bool(row.in_full_sample_BN))
    if set(reference_edges) != _edges(matrix, roles, int(config["bayesian_networks"]["d_max"])):
        raise ValueError("Recomputed reference BN differs from archived reference edges")
    for scenario in scenarios:
        archived = _scenario_count(matrix, scenario["factors"])
        if archived != scenario["full_count"]:
            raise ValueError(f"Archived scenario count differs for {scenario['id']}")
    parameter_policy = getattr(args, "parameter_policy", "fixed")
    design = _design(config_path, run_dir, config, replicates, fraction, seed, thresholds, main_threshold,
                     parameter_policy)
    fingerprint = _fingerprint(design)
    manifest_path = output / "design.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _fingerprint(previous) != fingerprint:
            raise ValueError(f"Output directory contains a different design: {output}")
    else:
        _write_json(manifest_path, design)
    requested = list(range(args.start, min(args.start + count, replicates)))
    global _RECOVERY_CONTEXT
    _RECOVERY_CONTEXT = {
        "output": output, "fingerprint": fingerprint, "seed": seed,
        "n_sample": max(1, min(len(accident_ids), round(fraction * len(accident_ids)))),
        "accident_ids": accident_ids, "units": units, "embeddings": embeddings,
        "matrix": matrix, "roles": roles, "d_max": int(config["bayesian_networks"]["d_max"]),
        "parameters": parameters, "umap_seed": int(config.get("validation", {}).get(
            "random_state", config.get("random_state", 42))),
        "config": config, "assignments": assignments, "scenarios": scenarios,
        "thresholds": thresholds, "parameter_policy": parameter_policy,
    }
    print(f"Running {len(requested)} replicates with {min(workers, len(requested))} local workers", flush=True)
    if workers == 1 or len(requested) == 1:
        results = (_run_one_replicate(replicate) for replicate in requested)
        for replicate, n_edges in results:
            message = f"Skipping completed replicate {replicate}" if n_edges < 0 else \
                f"Completed replicate {replicate}: {n_edges} reconstructed edges"
            print(message, flush=True)
        return
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("Parallel paired recovery requires a POSIX fork-capable environment")
    with ProcessPoolExecutor(max_workers=min(workers, len(requested)),
                             mp_context=multiprocessing.get_context("fork")) as executor:
        futures = [executor.submit(_run_one_replicate, replicate) for replicate in requested]
        for future in as_completed(futures):
            replicate, n_edges = future.result()
            message = f"Skipping completed replicate {replicate}" if n_edges < 0 else \
                f"Completed replicate {replicate}: {n_edges} reconstructed edges"
            print(message, flush=True)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _mcse(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def _plot_edges(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, zorder=1)
    ax.scatter(table["fixed_recovery"], table["reconstructed_recovery"],
               c="#147d85", s=36, zorder=2)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Fixed factors", ylabel="Reconstructed factors",
           title="Recovery of reference network edges")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "edge_recovery_real.pdf")
    plt.close(fig)


def _plot_article_scenarios(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = table["article_label"].tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hlines(y, table["fixed_recovery"], table["reconstructed_recovery"],
              color="0.72", linewidth=1.5, zorder=1)
    ax.scatter(table["fixed_recovery"], y, color="#147d85", s=48,
               label="Fixed factors", zorder=2)
    ax.scatter(table["reconstructed_recovery"], y, color="#bd633c", s=48,
               label="Reconstructed factors", zorder=2)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.02),
           xlabel="Scenario recovery frequency",
           title="Recovery of selected reference scenarios")
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "scenario_recovery_real.pdf")
    plt.close(fig)


def summarise(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    design = json.loads((output / "design.json").read_text(encoding="utf-8"))
    fingerprint = _fingerprint(design)
    m = int(design["replicates"])
    records = []
    for replicate in range(m):
        path = output / "replicates" / f"replicate_{replicate:04d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing replicate {replicate}; summary requires all {m} files")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["design_hash"] != fingerprint or record["replicate"] != replicate:
            raise ValueError(f"Mismatched replicate: {path}")
        records.append(record)
    run_dir = args.run_dir.resolve()
    config = load_bn_analysis_config(args.config.resolve(), args.dataset, run_dir)
    planned_replicates = paired_recovery_settings(config)["n_replicates"]
    thresholds = tuple(float(value) for value in design["matching_thresholds"])
    main_threshold = float(design["main_matching_threshold"])
    current = _design(args.config.resolve(), run_dir, config, m, float(design["fraction"]),
                      int(design["seed"]), thresholds, main_threshold,
                      design.get("parameter_policy", "fixed"))
    if _fingerprint(current) != fingerprint:
        raise ValueError("Source files or analysis settings changed after the jobs were run")
    roles = json.loads((run_dir / "bn_results_exact/matrix/variable_roles.json").read_text(encoding="utf-8"))
    edges_file = pd.read_csv(run_dir / "bn_results_exact/network/bn_edges_full.csv")
    reference_edges = [(str(row.parent_factor), str(row.child_factor))
                       for row in edges_file.itertuples(index=False) if bool(row.in_full_sample_BN)]
    scenarios = _scenarios(run_dir / "bn_results_exact/scenarios/recurrent_scenarios_all.csv", set(roles))
    scenario_by_id = {item["id"]: item for item in scenarios}
    q0 = REFERENCE_MIN_COUNT / len(pd.read_parquet(
        run_dir / "bn_results_exact/matrix/accident_factor_matrix.parquet"))
    for tau in thresholds:
        factor_rows = []
        for name, role in roles.items():
            recovered = [int(r["matches"][name]["candidate"] is not None
                             and r["matches"][name]["jaccard"] >= tau) for r in records]
            factor_rows.append({"factor": name, "role": role, "matching_threshold": tau,
                                "recovery": _mean(recovered), "mcse": _mcse(recovered),
                                "mean_jaccard": _mean([r["matches"][name]["jaccard"] for r in records]),
                                "observable_frequency": _mean([int(r["matches"][name]["n_reference_units"] > 0)
                                                               for r in records])})
        factor_table = pd.DataFrame(factor_rows)
        factor_table.to_csv(output / f"factor_recovery_tau_{tau:.1f}.csv", index=False)
        edge_rows = []
        for parent, child in reference_edges:
            fixed, joint, reconstructed = [], [], []
            for r in records:
                fixed.append(int([parent, child] in r["fixed_edges"]))
                a, b = r["matches"][parent], r["matches"][child]
                available = a["candidate"] is not None and b["candidate"] is not None \
                    and a["jaccard"] >= tau and b["jaccard"] >= tau
                joint.append(int(available))
                reconstructed.append(int(available and [a["candidate"], b["candidate"]]
                                         in r["reconstructed_edges"]))
            conditional = float(sum(reconstructed) / sum(joint)) if sum(joint) else float("nan")
            differences = [x-y for x, y in zip(fixed, reconstructed)]
            edge_rows.append({"parent": parent, "child": child, "matching_threshold": tau,
                              "fixed_recovery": _mean(fixed), "reconstructed_recovery": _mean(reconstructed),
                              "joint_factor_recovery": _mean(joint),
                              "edge_given_factors": conditional, "paired_difference": _mean(differences),
                              "paired_difference_mcse": _mcse(differences),
                              "reconstructed_mcse": _mcse(reconstructed),
                              "n_joint_factor_recoveries": sum(joint)})
        edge_table = pd.DataFrame(edge_rows)
        edge_table.to_csv(output / f"edge_recovery_tau_{tau:.1f}.csv", index=False)
        scenario_rows = []
        for scenario_id, scenario in scenario_by_id.items():
            fixed, joint, reconstructed = [], [], []
            fixed_counts, reconstructed_counts = [], []
            for r in records:
                counts = r["scenarios"][scenario_id]
                fixed_count = int(counts["fixed_count"])
                rec_count = int(counts["reconstructed_counts"][str(tau)])
                available = all(r["matches"][name]["candidate"] is not None
                                and r["matches"][name]["jaccard"] >= tau
                                for name in scenario["factors"])
                fixed_counts.append(fixed_count)
                reconstructed_counts.append(rec_count)
                joint.append(int(available))
                fixed.append(int(fixed_count / r["sample_size"] >= q0))
                reconstructed.append(int(available and rec_count / r["sample_size"] >= q0))
            differences = [x-y for x, y in zip(fixed, reconstructed)]
            scenario_rows.append({"scenario_id": scenario_id, "description": scenario["label"],
                                  "reference_count": scenario["full_count"], "matching_threshold": tau,
                                  "reference_confidence": scenario["confidence"],
                                  "reference_lift": scenario["lift"],
                                  "fixed_recovery": _mean(fixed),
                                  "reconstructed_recovery": _mean(reconstructed),
                                  "all_factor_recovery": _mean(joint),
                                  "scenario_given_factors": sum(reconstructed)/sum(joint) if sum(joint) else float("nan"),
                                  "paired_difference": _mean(differences),
                                  "paired_difference_mcse": _mcse(differences),
                                  "mean_fixed_count": _mean(fixed_counts),
                                  "mean_reconstructed_count": _mean(reconstructed_counts)})
        scenario_table = pd.DataFrame(scenario_rows)
        scenario_table.to_csv(output / f"scenario_recovery_tau_{tau:.1f}.csv", index=False)
        if tau == main_threshold:
            _plot_edges(edge_table, output)
            if args.dataset == DEFAULT_DATASET:
                article = scenario_table.set_index("scenario_id").loc[list(ARTICLE_SCENARIOS.values())].reset_index()
                article.insert(0, "article_label", list(ARTICLE_SCENARIOS))
                article.insert(1, "plain_description", [ARTICLE_DESCRIPTIONS[label] for label in ARTICLE_SCENARIOS])
                expected = [(23, .9583, 4.7574), (7, .5385, 2.6731),
                            (5, .8333, 4.4551), (5, .4167, 2.2862)]
                for row, (count, confidence, lift) in zip(article.itertuples(index=False), expected):
                    if row.reference_count != count or abs(row.reference_confidence-confidence) > .001 \
                            or abs(row.reference_lift-lift) > .001:
                        raise ValueError(f"Manuscript scenario {row.article_label} no longer matches the archive")
                article.to_csv(output / "article_scenario_recovery.csv", index=False)
                _plot_article_scenarios(article, output)
            role_summary = factor_table.groupby("role", sort=False).agg(
                n_factors=("factor", "size"), mean_recovery=("recovery", "mean"),
                median_recovery=("recovery", "median"), mean_jaccard=("mean_jaccard", "mean"))
            role_summary.to_csv(output / "factor_recovery_by_role.csv")
            median_fixed = float(edge_table["fixed_recovery"].median())
            median_rec = float(edge_table["reconstructed_recovery"].median())
            lower = int((edge_table["paired_difference"] > 0).sum())
            display_scenarios = (article if args.dataset == DEFAULT_DATASET else scenario_table.sort_values(
                ["reference_count", "scenario_id"], ascending=[False, True]).head(6))
            lines = [
                "# Observed paired recovery results" if m >= 100 else "# Pilot quality check: do not cite these frequencies",
                "",
                f"Design: {m} paired samples, {design['fraction']:.0%} distinct accidents, "
                f"Jaccard matching threshold {main_threshold:.2f}; sensitivity files cover "
                + ", ".join(f"{value:.2f}" for value in thresholds if value != main_threshold) + ".",
                f"This pilot is for execution and output checks only; run the prespecified {planned_replicates} replicates "
                "before interpreting or citing recovery frequencies." if m < 100 else "",
                f"Reference objects: {len(roles)} factors, {len(reference_edges)} network edges, "
                f"{len(scenarios)} closed recurrent scenarios.",
                "",
                "## Factors",
                "",
                *[f"- {role}: mean factor recovery {role_summary.loc[role, 'mean_recovery']:.3f} "
                  f"across {int(role_summary.loc[role, 'n_factors'])} reference factors."
                  for role in ROLES],
                "",
                "## Network edges",
                "",
                f"Median edge recovery across the {len(reference_edges)} reference edges: "
                f"{median_fixed:.3f} with fixed factors and {median_rec:.3f} "
                f"with reconstructed factors. {lower} edges have lower reconstruction recovery.",
                f"Median joint endpoint recovery: {edge_table['joint_factor_recovery'].median():.3f}; "
                f"median conditional edge recovery among edges with defined conditional frequency: "
                f"{edge_table['edge_given_factors'].median():.3f}.",
                "",
                "## Selected reference scenarios",
                "",
                *[f"- {getattr(r, 'article_label', r.scenario_id)} / {r.scenario_id} "
                  f"({r.reference_count} accidents): fixed {r.fixed_recovery:.3f}; "
                  f"reconstructed {r.reconstructed_recovery:.3f}; all factors found "
                  f"{r.all_factor_recovery:.3f}. {getattr(r, 'plain_description', r.description)}"
                  for r in display_scenarios.itertuples(index=False)],
                "",
                "## Interpretation and limits",
                "",
                "The paired difference describes how often the original conclusion is lost or gained "
                "when the factors are reconstructed on the same accidents. Joint factor recovery "
                "and conditional edge recovery distinguish factor loss from graph-selection variation; "
                "they are descriptive frequencies, not causal effects or posterior probabilities.",
                "Scenario recovery here means that all factors are matched and the empirical support "
                "reaches 5/417 on the subsample. It does not require that a pattern remain closed "
                "within that subsample. Fixed and reconstructed results share the same accidents; "
                "the older with-replacement bootstrap is a different experiment.",
                "Monte Carlo standard errors for each paired difference are in the CSV tables. "
                "They quantify simulation precision only, not uncertainty from annotation, "
                "embedding choice, hyperparameter selection or population sampling.",
            ]
            (output / "observed_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote observed results to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "summarise", "settings"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=Path("recurrent_scenarios/config.yaml"))
        p.add_argument("--dataset", default=DEFAULT_DATASET)
        p.add_argument("--run-dir", type=Path,
                       default=Path(f"recurrent_scenarios/runs/theme_discovery_audit/{DEFAULT_DATASET}"))
        p.add_argument("--output", type=Path,
                       default=Path(f"recurrent_scenarios/runs/theme_discovery_audit/{DEFAULT_DATASET}/paired_recovery"))
        if command == "run":
            p.add_argument("--parameter-policy", choices=PARAMETER_POLICIES, default="fixed")
            p.add_argument("--replicates", type=int)
            p.add_argument("--fraction", type=float)
            p.add_argument("--seed", type=int)
            p.add_argument("--workers", type=int,
                           help="Override paired_recovery.n_workers for this run")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--count", type=int,
                           help="Run this many replicates; defaults to all remaining replicates")
    args = parser.parse_args()
    if args.command == "settings":
        settings = paired_recovery_settings(load_bn_analysis_config(args.config.resolve(), args.dataset, args.run_dir.resolve()))
        print("\n".join(str(settings[name]) for name in ("n_replicates", "fraction", "seed", "n_workers")))
    else:
        (run if args.command == "run" else summarise)(args)


if __name__ == "__main__":
    main()
