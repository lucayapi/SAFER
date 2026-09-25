"""Exact role-constrained observed-factor Bayesian network.

The primary analysis has no latent variable. Because all allowed arcs follow
the fixed A0 -> A1 -> B -> C order, each node's admissible parent set can be
optimized independently with the decomposable BIC score.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scenario_pipeline import (
    ROLES,
    StructuralEMResult,
    _allowed_bn_edges,
    _bn_config,
    _bn_parent_map,
    _theme_label_map,
)


TIE_ATOL = 1e-12


def _assert_bn_edge_constraints(edges: Sequence[tuple[str, str]], roles: Mapping[str, str], d_max: int) -> None:
    allowed = set(_allowed_bn_edges(roles))
    parents = _bn_parent_map(list(roles), edges)
    for parent, child in edges:
        if (parent, child) not in allowed:
            raise AssertionError(f"Forbidden BN arc: {parent} ({roles[parent]}) -> {child} ({roles[child]})")
    for node, node_parents in parents.items():
        if len(node_parents) > d_max:
            raise AssertionError(f"In-degree > {d_max} for {node}: {node_parents}")


def _admissible_parent_sets(node: str, roles: Mapping[str, str], d_max: int) -> list[tuple[str, ...]]:
    allowed = sorted(parent for parent, child in _allowed_bn_edges(roles) if child == node)
    return [
        tuple(parents)
        for size in range(min(d_max, len(allowed)) + 1)
        for parents in itertools.combinations(allowed, size)
    ]


def _local_mle_bic(data: np.ndarray, node_index: Mapping[str, int], child: str, parents: Sequence[str]) -> tuple[float, float, int]:
    """Return maximized local log likelihood, BIC and nominal parameter count."""

    n_accidents = len(data)
    if n_accidents < 1:
        raise ValueError("The exact BN requires at least one accident.")
    parent_positions = [node_index[parent] for parent in parents]
    child_values = data[:, node_index[child]]
    log_likelihood = 0.0
    for values in itertools.product((0, 1), repeat=len(parent_positions)):
        mask = np.ones(n_accidents, dtype=bool)
        for position, value in zip(parent_positions, values):
            mask &= data[:, position] == value
        total = int(mask.sum())
        if total == 0:
            continue
        positive = int(child_values[mask].sum())
        negative = total - positive
        if positive:
            log_likelihood += positive * math.log(positive / total)
        if negative:
            log_likelihood += negative * math.log(negative / total)
    n_parameters = 2 ** len(parent_positions)
    bic = -2.0 * log_likelihood + n_parameters * math.log(n_accidents)
    return float(log_likelihood), float(bic), int(n_parameters)


def _exact_structure_learning(
    data: np.ndarray,
    nodes: Sequence[str],
    roles: Mapping[str, str],
    d_max: int,
) -> tuple[list[tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    """Enumerate all local parent sets and select the exact constrained BN."""

    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    edges: list[tuple[str, str]] = []
    for child in nodes:
        candidates = []
        for parents in _admissible_parent_sets(child, roles, d_max):
            log_likelihood, bic, n_parameters = _local_mle_bic(data, node_index, child, parents)
            candidates.append({
                "child_factor": child,
                "child_role": roles[child],
                "parent_factor_ids": " | ".join(parents),
                "parent_tuple": parents,
                "n_parents": len(parents),
                "log_likelihood": log_likelihood,
                "n_parameters": n_parameters,
                "BIC": bic,
            })
        minimum_bic = min(float(row["BIC"]) for row in candidates)
        ties = [row for row in candidates if np.isclose(float(row["BIC"]), minimum_bic, rtol=0.0, atol=TIE_ATOL)]
        chosen = min(ties, key=lambda row: (int(row["n_parents"]), tuple(row["parent_tuple"])))
        chosen_parents = tuple(chosen["parent_tuple"])
        tied_parent_sets = {tuple(row["parent_tuple"]) for row in ties}
        ordered = sorted(candidates, key=lambda row: (float(row["BIC"]), int(row["n_parents"]), tuple(row["parent_tuple"])))
        rank_lookup = {tuple(row["parent_tuple"]): rank for rank, row in enumerate(ordered, start=1)}
        for row in candidates:
            key = tuple(row["parent_tuple"])
            row["BIC_tied_for_minimum"] = key in tied_parent_sets
            row["selected"] = key == chosen_parents
            row["rank"] = rank_lookup[key]
            row.pop("parent_tuple")
            rows.append(row)
        selected_rows.append({
            "child_factor": child,
            "child_role": roles[child],
            "parent_factor_ids": " | ".join(chosen_parents),
            "n_parents": chosen["n_parents"],
            "log_likelihood": chosen["log_likelihood"],
            "n_parameters": chosen["n_parameters"],
            "BIC": chosen["BIC"],
            "n_BIC_minimum_ties": len(ties),
            "tie_break": "not_required" if len(ties) == 1 else "fewest_parents_then_factor_id_order",
        })
        edges.extend((parent, child) for parent in chosen_parents)
    return sorted(edges), pd.DataFrame(rows), pd.DataFrame(selected_rows)


def _postselection_cpt_counts(
    data: np.ndarray,
    nodes: Sequence[str],
    roles: Mapping[str, str],
    edges: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Return MLE and Jeffreys estimates for every selected CPT row.

    This function is deliberately separate from BIC scoring.  BIC uses the
    MLE likelihood of *observed* configurations only; the Jeffreys values
    below are available solely for post-selection inference when support
    diagnostics show that a fully specified MLE network is unavailable or
    insufficiently supported according to the configured threshold.
    """

    parents = _bn_parent_map(list(nodes), edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[dict[str, Any]] = []
    for child in nodes:
        parent_nodes = parents[child]
        parent_positions = [node_index[parent] for parent in parent_nodes]
        child_values = data[:, node_index[child]]
        for parent_values in itertools.product((0, 1), repeat=len(parent_positions)):
            mask = np.ones(len(data), dtype=bool)
            for position, value in zip(parent_positions, parent_values):
                mask &= data[:, position] == value
            total = int(mask.sum())
            positive = int(child_values[mask].sum())
            mle_probability = positive / total if total else np.nan
            jeffreys_probability = (positive + 0.5) / (total + 1.0)
            rows.append({
                "child_factor": child,
                "child_role": roles[child],
                "parent_factor_ids": " | ".join(parent_nodes),
                "parent_values": " | ".join(f"{parent}={value}" for parent, value in zip(parent_nodes, parent_values)),
                "n_parent_configuration": total,
                "n_child_1": positive,
                "n_child_0": total - positive,
                "parent_configuration_observed": total > 0,
                "MLE_P_child_1": mle_probability,
                "MLE_P_child_0": 1.0 - mle_probability if total else np.nan,
                "Jeffreys_P_child_1": jeffreys_probability,
                "Jeffreys_P_child_0": 1.0 - jeffreys_probability,
            })
    return pd.DataFrame(rows)


def _cpt_probabilities(cpts: pd.DataFrame, probability_column: str) -> dict[tuple[str, tuple[int, ...]], float]:
    """Convert a CPT export to pgmpy-compatible probability lookup values."""

    probabilities: dict[tuple[str, tuple[int, ...]], float] = {}
    for _, row in cpts.iterrows():
        parent_values = tuple(
            int(fragment.rsplit("=", 1)[1])
            for fragment in str(row["parent_values"]).split(" | ")
            if fragment
        )
        probability = float(row[probability_column])
        if not np.isfinite(probability):
            raise ValueError(
                "MLE CPTs cannot define a full joint distribution because at least one selected "
                "parent configuration is unobserved."
            )
        probabilities[(str(row["child_factor"]), parent_values)] = probability
    return probabilities


def _cpt_support_diagnostics(cpts: pd.DataFrame, min_observed_count: int) -> pd.DataFrame:
    """Summarize support before selecting MLE or post-selection regularization."""

    if min_observed_count < 0:
        raise ValueError("cpt_regularization_min_observed_count must be non-negative.")
    observed = cpts.loc[cpts["parent_configuration_observed"]].copy()
    counts = cpts["n_parent_configuration"]
    requires_regularization = bool(counts.le(min_observed_count).any())
    return pd.DataFrame([{
        "n_cpt_rows": int(len(cpts)),
        "n_rows_N_eq_0": int(counts.eq(0).sum()),
        "n_rows_N_le_2": int(counts.le(2).sum()),
        "n_rows_N_le_5": int(counts.le(5).sum()),
        "n_MLE_equal_0": int(observed["MLE_P_child_1"].eq(0.0).sum()),
        "n_MLE_equal_1": int(observed["MLE_P_child_1"].eq(1.0).sum()),
        "minimum_observed_cell_count": int(observed["n_parent_configuration"].min()) if not observed.empty else np.nan,
        "median_observed_cell_count": float(observed["n_parent_configuration"].median()) if not observed.empty else np.nan,
        "regularization_min_observed_count": int(min_observed_count),
        "n_rows_at_or_below_regularization_threshold": int(counts.le(min_observed_count).sum()),
        "final_CPT_estimation": "Jeffreys" if requires_regularization else "MLE",
        "regularization_required": requires_regularization,
    }])


def _jeffreys_cpts(
    data: np.ndarray,
    nodes: Sequence[str],
    roles: Mapping[str, str],
    edges: Sequence[tuple[str, str]],
) -> tuple[dict[tuple[str, tuple[int, ...]], float], pd.DataFrame]:
    """Compatibility wrapper returning all post-selection Jeffreys CPTs."""

    cpts = _postselection_cpt_counts(data, nodes, roles, edges)
    return _cpt_probabilities(cpts, "Jeffreys_P_child_1"), cpts


def _build_exact_pgmpy_model(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    probabilities: Mapping[tuple[str, tuple[int, ...]], float],
) -> Any:
    try:
        from pgmpy.factors.discrete import TabularCPD
        try:
            from pgmpy.models import DiscreteBayesianNetwork
        except ImportError:
            from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
    except ImportError as error:
        raise ImportError("pgmpy is required for exact BN inference.") from error

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    parents = _bn_parent_map(list(nodes), edges)
    cpds = []
    for child in nodes:
        parent_nodes = parents[child]
        columns = list(itertools.product((0, 1), repeat=len(parent_nodes)))
        values = np.zeros((2, len(columns)), dtype=float)
        for column, parent_values in enumerate(columns):
            probability = float(probabilities[(child, tuple(parent_values))])
            values[:, column] = [1.0 - probability, probability]
        cpds.append(TabularCPD(
            child, 2, values.tolist(), evidence=parent_nodes or None,
            evidence_card=[2] * len(parent_nodes) if parent_nodes else None,
        ))
    model.add_cpds(*cpds)
    model.check_model()
    return model


def fit_global_bn(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    initialization: str = "exact",
    progress_callback: Callable[[int, float, int], None] | None = None,
    build_inference_model: bool = True,
) -> StructuralEMResult:
    """Fit the exact BIC-optimal BN in the role-constrained graph class."""

    del seed, initialization
    cfg = _bn_config(config)
    nodes = [column for column in matrix.columns if column != "accident_id"]
    data = matrix[nodes].to_numpy(dtype=np.int8)
    edges, local_scores, selected_parent_sets = _exact_structure_learning(data, nodes, roles, int(cfg["d_max"]))
    _assert_bn_edge_constraints(edges, roles, int(cfg["d_max"]))
    cpt_rows = _postselection_cpt_counts(data, nodes, roles, edges)
    support_threshold = int(cfg.get("cpt_regularization_min_observed_count", 0))
    cpt_diagnostics = _cpt_support_diagnostics(cpt_rows, support_threshold)
    final_method = str(cpt_diagnostics.loc[0, "final_CPT_estimation"])
    mle_probabilities: dict[tuple[str, tuple[int, ...]], float] | None
    try:
        mle_probabilities = _cpt_probabilities(cpt_rows, "MLE_P_child_1")
    except ValueError:
        mle_probabilities = None
    jeffreys_probabilities = _cpt_probabilities(cpt_rows, "Jeffreys_P_child_1")
    probabilities = mle_probabilities if final_method == "MLE" else jeffreys_probabilities
    if probabilities is None:
        raise AssertionError("An MLE final CPT selection requires every selected row to be observed.")
    final_cpts = cpt_rows.copy()
    final_cpts["CPT_estimation"] = final_method
    final_cpts["P_child_1"] = final_cpts[f"{final_method}_P_child_1"]
    final_cpts["P_child_0"] = final_cpts[f"{final_method}_P_child_0"]
    final_model = _build_exact_pgmpy_model(nodes, edges, probabilities) if build_inference_model else None
    mle_model = (
        final_model if final_method == "MLE" else
        _build_exact_pgmpy_model(nodes, edges, mle_probabilities)
        if build_inference_model and mle_probabilities is not None else None
    )
    jeffreys_model = (
        final_model if final_method == "Jeffreys" else
        _build_exact_pgmpy_model(nodes, edges, jeffreys_probabilities)
        if build_inference_model else None
    )
    log_likelihood = float(selected_parent_sets["log_likelihood"].sum())
    bic = float(selected_parent_sets["BIC"].sum())
    result = StructuralEMResult(
        nodes=list(nodes), roles=dict(roles), edges=list(edges), n_states=1,
        weights=np.array([1.0]), responsibilities=np.ones((len(data), 1), dtype=float),
        upstream_probabilities={}, downstream_probabilities=probabilities,
        log_likelihood=log_likelihood, bic=bic, n_iter=1, converged=True,
        seed=0, initialization="exact_local_BIC", model=final_model,
        iteration_history=[], last_loglik_delta=0.0, relative_loglik_delta=0.0,
        same_graph=True, edges_added_last=len(edges), edges_removed_last=0,
        observed_data=np.asarray(data, dtype=np.int8).copy(),
    )
    result.exact_local_scores = local_scores
    result.exact_selected_parent_sets = selected_parent_sets
    result.mle_cpts = cpt_rows.copy()
    result.jeffreys_cpts = cpt_rows.copy()
    result.final_cpts = final_cpts
    result.cpt_support_diagnostics = cpt_diagnostics
    result.cpt_estimation_method = final_method
    result.mle_model = mle_model
    result.jeffreys_model = jeffreys_model
    if progress_callback is not None:
        progress_callback(1, log_likelihood, len(edges))
    return result


def _max_observed_parents(edges: Sequence[tuple[str, str]], nodes: Sequence[str]) -> int:
    parents = _bn_parent_map(list(nodes), edges)
    return max((len(parents[node]) for node in nodes), default=0)


def write_global_bn_summary(result: StructuralEMResult, matrix: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    selected = getattr(result, "exact_selected_parent_sets", pd.DataFrame())
    local_scores = getattr(result, "exact_local_scores", pd.DataFrame())
    counts = {role: sum(1 for node in result.nodes if result.roles[node] == role) for role in ROLES}
    transitions = {
        f"n_{left}_to_{right}": sum(result.roles[parent] == left and result.roles[child] == right for parent, child in result.edges)
        for left, right in (("A0", "A1"), ("A0", "B"), ("A1", "B"), ("B", "C"))
    }
    summary = pd.DataFrame([{
        "structure_optimization": "exact_role_constrained_local_BIC",
        "n_accidents": int(len(matrix)), "n_variables": len(result.nodes),
        **{f"n_{role}": counts[role] for role in ROLES},
        "n_local_parent_sets_evaluated": int(len(local_scores)), "n_edges": len(result.edges),
        **transitions,
        "n_parameters": int(selected["n_parameters"].sum()) if not selected.empty else np.nan,
        "log_likelihood": float(result.log_likelihood), "BIC": float(result.bic),
        "final_CPT_estimation": getattr(result, "cpt_estimation_method", "unknown"),
        "max_observed_parents": int(_max_observed_parents(result.edges, result.nodes)), "completed": True,
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "global_bn_summary.csv", index=False)
    local_scores.to_csv(output_dir / "bn_local_scores.csv", index=False)
    selected.to_csv(output_dir / "bn_selected_parent_sets.csv", index=False)
    getattr(result, "mle_cpts", pd.DataFrame()).to_csv(output_dir / "bn_mle_cpts.csv", index=False)
    getattr(result, "jeffreys_cpts", pd.DataFrame()).to_csv(output_dir / "bn_jeffreys_cpts.csv", index=False)
    getattr(result, "final_cpts", pd.DataFrame()).to_csv(output_dir / "bn_final_cpts.csv", index=False)
    getattr(result, "cpt_support_diagnostics", pd.DataFrame()).to_csv(
        output_dir / "bn_cpt_support_diagnostics.csv", index=False,
    )
    return summary


def _edge_contrast_strata(result: StructuralEMResult, parent: str, child: str) -> pd.DataFrame:
    if result.observed_data is None:
        raise ValueError("Conditional contrasts require the observed matrix.")
    data = np.asarray(result.observed_data, dtype=np.int8)
    parents = result.parent_map[child]
    if parent not in parents:
        raise ValueError(f"{parent} is not a parent of {child}.")
    index = {node: position for position, node in enumerate(result.nodes)}
    parent_position = parents.index(parent)
    other_positions = [position for position in range(len(parents)) if position != parent_position]
    other_parents = [parents[position] for position in other_positions]
    rows = []
    for other_values in itertools.product((0, 1), repeat=len(other_parents)):
        counts, probabilities = [], []
        for parent_value in (0, 1):
            values = [0] * len(parents)
            values[parent_position] = parent_value
            for position, value in zip(other_positions, other_values):
                values[position] = value
            mask = np.ones(len(data), dtype=bool)
            for parent_node, value in zip(parents, values):
                mask &= data[:, index[parent_node]] == value
            counts.append(int(mask.sum()))
            probabilities.append(float(result.downstream_probabilities[(child, tuple(values))]))
        estimable = counts[0] > 0 and counts[1] > 0
        rows.append({
            "parent_factor": parent, "child_factor": child,
            "other_parent_configuration": " | ".join(f"{node}={value}" for node, value in zip(other_parents, other_values)),
            "n_parent_0": counts[0], "n_parent_1": counts[1], "n_context": counts[0] + counts[1], "estimable": estimable,
            "CPT_estimation": getattr(result, "cpt_estimation_method", "external"),
            "P_child_1_parent_0": probabilities[0] if estimable else np.nan,
            "P_child_1_parent_1": probabilities[1] if estimable else np.nan,
            "conditional_contrast": probabilities[1] - probabilities[0] if estimable else np.nan,
        })
    return pd.DataFrame(rows)


def _contrast_pattern(values: pd.Series) -> str:
    if values.empty:
        return "not_estimable"
    if bool(values.ge(0.0).all()) and bool(values.gt(0.0).any()):
        return "monotonically_positive"
    if bool(values.le(0.0).all()) and bool(values.lt(0.0).any()):
        return "monotonically_negative"
    return "non_monotone"


def write_global_bn_edges(
    result: StructuralEMResult, matrix: pd.DataFrame, label_map: Mapping[str, str], roles: Mapping[str, str],
    bootstrap: pd.DataFrame | None, output_dir: Path, *, stable_threshold: float = 0.60,
) -> pd.DataFrame:
    freq_lookup = {} if bootstrap is None or bootstrap.empty else {
        (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
        for _, row in bootstrap.iterrows()
    }
    data = matrix[result.nodes].to_numpy(dtype=np.int8)
    prevalence = {node: float(data[:, position].mean()) for position, node in enumerate(result.nodes)}
    rows, all_strata = [], []
    for parent, child in result.edges:
        strata = _edge_contrast_strata(result, parent, child)
        all_strata.append(strata)
        estimable = strata.loc[strata["estimable"]].copy()
        if estimable.empty:
            delta = delta_min = delta_max = float("nan")
        else:
            weights = estimable["n_context"] / estimable["n_context"].sum()
            delta = float((weights * estimable["conditional_contrast"]).sum())
            delta_min, delta_max = float(estimable["conditional_contrast"].min()), float(estimable["conditional_contrast"].max())
        frequency = freq_lookup.get((parent, child), np.nan)
        n_resamples = (
            int(bootstrap["n_resamples"].iloc[0])
            if bootstrap is not None and not bootstrap.empty and "n_resamples" in bootstrap.columns
            else 0
        )
        rows.append({
            "parent_factor": parent, "parent_label": label_map.get(parent, parent), "parent_role": roles[parent],
            "child_factor": child, "child_label": label_map.get(child, child), "child_role": roles[child],
            "transition": f"{roles[parent]}->{roles[child]}", "in_full_sample_BN": True,
            "bootstrap_frequency": frequency,
            "bootstrap_MCSE": math.sqrt(frequency * (1.0 - frequency) / n_resamples) if np.isfinite(frequency) and n_resamples else np.nan,
            "conditional_contrast_weighted": delta, "conditional_contrast_min": delta_min, "conditional_contrast_max": delta_max,
            "conditional_contrast_pattern": _contrast_pattern(estimable["conditional_contrast"]),
            "conditional_contrast_n_estimable_strata": int(len(estimable)), "conditional_contrast_n_total_strata": int(len(strata)),
            "conditional_contrast_signed": delta, "conditional_contrast_abs": abs(delta) if np.isfinite(delta) else np.nan,
            "parent_child_observed_count": int((data[:, result.nodes.index(parent)] & data[:, result.nodes.index(child)]).sum()),
            "parent_prevalence": prevalence[parent], "child_prevalence": prevalence[child],
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("bootstrap_frequency", ascending=False, na_position="last").reset_index(drop=True)
    strata_frame = pd.concat(all_strata, ignore_index=True) if all_strata else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "bn_edges_full.csv", index=False)
    frame.to_csv(output_dir / "global_bn_edges.csv", index=False)
    strata_frame.to_csv(output_dir / "bn_conditional_contrasts.csv", index=False)
    strata_frame.to_csv(output_dir / "global_bn_edge_contrast_strata.csv", index=False)
    if not frame.empty:
        frame.loc[frame["bootstrap_frequency"].ge(stable_threshold) & frame["conditional_contrast_pattern"].eq("monotonically_negative")].to_csv(output_dir / "stable_negative_edges.csv", index=False)
    return frame


def run_global_bn_bootstrap(
    matrix: pd.DataFrame, roles: Mapping[str, str], config: Mapping[str, Any],
    reference: StructuralEMResult, output_dir: Path,
) -> pd.DataFrame:
    """Accident-level nonparametric bootstrap of the exact learner."""

    bootstrap_cfg = _bn_config(config).get("bn_structure_bootstrap", {})
    if not bool(bootstrap_cfg.get("enabled", False)):
        return pd.DataFrame()
    n_resamples = int(bootstrap_cfg.get("n_resamples", 500))
    if n_resamples < 1:
        raise ValueError("bn_structure_bootstrap.n_resamples must be positive.")
    nodes, data = list(roles), matrix[list(roles)].to_numpy(dtype=np.int8)
    # A child seed is allocated to every replicate.  This keeps the sampling
    # stream reproducible even if the loop is later parallelised or resumed.
    root_seed = int(bootstrap_cfg.get("random_state", config.get("random_state", 42)))
    replicate_seed_sequences = np.random.SeedSequence(root_seed).spawn(n_resamples)
    edge_counts: dict[tuple[str, str], int] = {}
    report_every = max(1, n_resamples // 20)
    for index, seed_sequence in enumerate(replicate_seed_sequences):
        replicate_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(replicate_seed)
        sampled = rng.choice(len(data), size=len(data), replace=True)
        sample_matrix = pd.DataFrame(data[sampled], columns=nodes)
        sample_matrix.insert(0, "accident_id", sampled.astype(str))
        for edge in fit_global_bn(sample_matrix, roles, config, build_inference_model=False).edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if (index + 1) % report_every == 0 or index + 1 == n_resamples:
            print(f"[exact-bn] bootstrap {index + 1}/{n_resamples}", flush=True, file=sys.stdout)
    labels, reference_edges = _theme_label_map(None), set(reference.edges)
    rows = []
    for parent, child in sorted(set(edge_counts) | reference_edges):
        frequency = edge_counts.get((parent, child), 0) / n_resamples
        rows.append({
            "parent": parent, "child": child, "parent_role": roles[parent], "child_role": roles[child],
            "parent_label": labels.get(parent, parent), "child_label": labels.get(child, child),
            "selection_count": edge_counts.get((parent, child), 0), "selection_frequency": frequency,
            "selection_frequency_MCSE": math.sqrt(frequency * (1.0 - frequency) / n_resamples),
            "n_resamples": n_resamples, "bootstrap_sample_size": len(data), "bootstrap_root_seed": root_seed,
            "in_reference_bn": (parent, child) in reference_edges,
        })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "bn_bootstrap_edges.csv", index=False)
    frame.to_csv(output_dir / "global_bn_bootstrap_edges.csv", index=False)
    return frame


def _scenario_implied_support(model: Any, scenario_nodes: Sequence[str]) -> float:
    """Return P(all scenario nodes = 1), marginalizing every other node."""

    try:
        from pgmpy.inference import VariableElimination
    except ImportError as error:
        raise ImportError("pgmpy is required for scenario BN inference.") from error
    query = VariableElimination(model).query(variables=list(scenario_nodes), show_progress=False)
    try:
        return float(query.get_value(**{node: 1 for node in scenario_nodes}))
    except AttributeError:
        return float(query.values[tuple(1 for _ in scenario_nodes)])


def add_scenario_bn_discrepancy(
    scenarios: pd.DataFrame,
    result: StructuralEMResult,
    n_accidents: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Attach marginal BN support to recurrent closed scenarios.

    Only the scenario factors are queried. All other factor variables are
    marginalized by variable elimination, matching containment-based scenario
    recurrence in the observed accident matrix.
    """

    if scenarios.empty:
        empty = pd.DataFrame(columns=[
            "scenario_id", "BN_implied_support", "BN_expected_accident_count",
            "BN_support_discrepancy", "BN_support_discrepancy_pp", "BN_support_interestingness",
            "BN_CPT_estimation", "BN_MLE_support", "BN_Jeffreys_support", "BN_MLE_minus_Jeffreys_pp",
            "internal_BN_edges", "n_internal_BN_edges",
        ])
        output_dir.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_dir / "scenario_bn_discrepancy.csv", index=False)
        empty.to_csv(output_dir / "scenario_bn_cpt_sensitivity.csv", index=False)
        return scenarios.copy()
    if result.model is None:
        raise ValueError("Scenario support requires the fitted exact BN model.")
    rows = []
    for _, row in scenarios.iterrows():
        upstream = [part for part in str(row["upstream_factor_ids"]).split(" | ") if part]
        scenario_nodes = [*upstream, str(row["B_factor_id"]), str(row["C_factor_id"])]
        implied_support = _scenario_implied_support(result.model, scenario_nodes)
        mle_support = (
            _scenario_implied_support(result.mle_model, scenario_nodes)
            if getattr(result, "mle_model", None) is not None else np.nan
        )
        jeffreys_support = (
            _scenario_implied_support(result.jeffreys_model, scenario_nodes)
            if getattr(result, "jeffreys_model", None) is not None else np.nan
        )
        observed_support = float(row["scenario_support"])
        discrepancy = observed_support - implied_support
        internal_edges = [
            (parent, child) for parent, child in result.edges
            if parent in scenario_nodes and child in scenario_nodes
        ]
        rows.append({
            "scenario_id": row["scenario_id"],
            "BN_implied_support": implied_support,
            "BN_expected_accident_count": n_accidents * implied_support,
            "BN_support_discrepancy": discrepancy,
            "BN_support_discrepancy_pp": 100.0 * discrepancy,
            "BN_support_interestingness": abs(discrepancy),
            "BN_CPT_estimation": getattr(result, "cpt_estimation_method", "unknown"),
            "BN_MLE_support": mle_support,
            "BN_Jeffreys_support": jeffreys_support,
            "BN_MLE_minus_Jeffreys_pp": 100.0 * (mle_support - jeffreys_support)
            if np.isfinite(mle_support) and np.isfinite(jeffreys_support) else np.nan,
            "internal_BN_edges": " | ".join(f"{parent}->{child}" for parent, child in internal_edges),
            "n_internal_BN_edges": len(internal_edges),
        })
    discrepancy_frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    discrepancy_frame.to_csv(output_dir / "scenario_bn_discrepancy.csv", index=False)
    discrepancy_frame.to_csv(output_dir / "scenario_bn_cpt_sensitivity.csv", index=False)
    return scenarios.merge(discrepancy_frame, on="scenario_id", how="left", validate="one_to_one")


def assert_expected_inventory(matrix: pd.DataFrame, roles: Mapping[str, str], config: Mapping[str, Any]) -> None:
    nodes = [column for column in matrix.columns if column != "accident_id"]
    dataset_id = str(config.get("data", {}).get("dataset_id", config.get("dataset_id", "")))
    expected = config.get("expected_inventory", {}).get(dataset_id, {})
    if not expected:
        return
    if len(matrix) != int(expected["n_accidents"]):
        raise AssertionError(f"Unexpected accident inventory for {dataset_id}: {len(matrix)} != {expected['n_accidents']}")
    if len(nodes) != int(expected["n_factors"]):
        raise AssertionError(f"Unexpected factor inventory for {dataset_id}: {len(nodes)} != {expected['n_factors']}")
    counts = {role: sum(1 for node in nodes if roles[node] == role) for role in ROLES}
    for role, expected_count in expected.get("role_counts", {}).items():
        if counts[role] != int(expected_count):
            raise AssertionError(f"Unexpected {role} factor inventory for {dataset_id}: {counts[role]} != {expected_count}")


def assert_global_bn_outputs(
    result: StructuralEMResult, matrix: pd.DataFrame, roles: Mapping[str, str], bootstrap: pd.DataFrame,
    edges: pd.DataFrame, config: Mapping[str, Any],
) -> None:
    _assert_bn_edge_constraints(result.edges, roles, int(_bn_config(config)["d_max"]))
    assert_expected_inventory(matrix, roles, config)
    if result.model is None or "Z" in result.model.nodes():
        raise AssertionError("The exact global BN must not contain a latent Z node.")
    if not bootstrap.empty:
        assert bootstrap["selection_frequency"].between(0.0, 1.0).all()
        assert bootstrap["bootstrap_sample_size"].eq(len(matrix)).all()
    if not edges.empty:
        assert edges["conditional_contrast_n_estimable_strata"].le(edges["conditional_contrast_n_total_strata"]).all()
