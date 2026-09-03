"""Global observed-factor Bayesian network (no latent Z) and bootstrap stability."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scenario_pipeline import (
    BN_ROLE_ARCS,
    ROLES,
    StructuralEMResult,
    _allowed_bn_edges,
    _bn_config,
    _bn_mle_parameters,
    _bn_parameter_count,
    _bn_parent_map,
    _edge_conditional_contrast_signed,
    _edge_conditional_strength,
    _fit_bn_k1,
    _latent_scope,
    _theme_label_map,
    finalize_latent_bn,
)


def _assert_bn_edge_constraints(edges: Sequence[tuple[str, str]], roles: Mapping[str, str], d_max: int) -> None:
    allowed = set(_allowed_bn_edges(roles))
    parents = _bn_parent_map(list(roles), edges)
    for parent, child in edges:
        if (parent, child) not in allowed:
            raise AssertionError(f"Arc BN interdit: {parent} ({roles[parent]}) -> {child} ({roles[child]})")
        if roles[parent] == roles[child]:
            raise AssertionError(f"Arc intra-rôle interdit: {parent} -> {child}")
    for node, node_parents in parents.items():
        if len(node_parents) > d_max:
            raise AssertionError(f"In-degree > {d_max} pour {node}: {node_parents}")


def fit_global_bn(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    initialization: str = "empty",
    progress_callback: Callable[[int, float, int], None] | None = None,
) -> StructuralEMResult:
    """Learn a role-constrained BN on observed factors only (equivalent to K=1, no Z)."""

    cfg = _bn_config(config)
    nodes = [column for column in matrix.columns if column != "accident_id"]
    data = matrix[nodes].to_numpy(dtype=np.int8)
    fit_seed = int(seed if seed is not None else cfg.get("random_state", config.get("random_state", 42)))
    result = _fit_bn_k1(data, nodes, dict(roles), fit_seed, initialization, config, progress_callback)
    if result.n_states != 1:
        raise AssertionError("Le BN global doit avoir n_states=1 (pas de variable latente Z).")
    _assert_bn_edge_constraints(result.edges, roles, int(cfg["d_max"]))
    return finalize_latent_bn(result, matrix, roles, config)


def _max_observed_parents(edges: Sequence[tuple[str, str]], nodes: Sequence[str]) -> int:
    parents = _bn_parent_map(list(nodes), edges)
    return max((len(parents[node]) for node in nodes), default=0)


def write_global_bn_summary(result: StructuralEMResult, matrix: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    n_accidents = len(matrix)
    summary = pd.DataFrame([{
        "n_accidents": int(n_accidents),
        "n_variables": len(result.nodes),
        "n_edges": len(result.edges),
        "n_parameters": int(_bn_parameter_count(result.nodes, result.roles, result.edges, 1, "upstream_only")),
        "log_likelihood": float(result.log_likelihood),
        "BIC": float(result.bic),
        "max_observed_parents": int(_max_observed_parents(result.edges, result.nodes)),
        "converged_or_completed": bool(result.converged),
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "global_bn_summary.csv", index=False)
    return summary


def write_global_bn_edges(
    result: StructuralEMResult,
    matrix: pd.DataFrame,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    bootstrap: pd.DataFrame | None,
    output_dir: Path,
    *,
    stable_threshold: float = 0.60,
) -> pd.DataFrame:
    nodes = result.nodes
    data = matrix[nodes].to_numpy(dtype=np.int8)
    freq_lookup: dict[tuple[str, str], float] = {}
    if bootstrap is not None and not bootstrap.empty:
        freq_lookup = {
            (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
            for _, row in bootstrap.iterrows()
        }
    prevalence = {node: float(data[:, index].mean()) for index, node in enumerate(nodes)}
    rows = []
    for parent, child in result.edges:
        parent_index = nodes.index(parent)
        child_index = nodes.index(child)
        joint = int((data[:, parent_index] & data[:, child_index]).sum())
        signed = _edge_conditional_contrast_signed(result, parent, child)
        rows.append({
            "parent_factor": parent,
            "parent_label": label_map.get(parent, parent),
            "parent_role": roles[parent],
            "child_factor": child,
            "child_label": label_map.get(child, child),
            "child_role": roles[child],
            "bootstrap_frequency": freq_lookup.get((parent, child), np.nan),
            "conditional_contrast_signed": signed,
            "conditional_contrast_abs": abs(signed),
            "parent_child_observed_count": joint,
            "parent_prevalence": prevalence[parent],
            "child_prevalence": prevalence[child],
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("bootstrap_frequency", ascending=False, na_position="last")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "global_bn_edges.csv", index=False)
    if not frame.empty and "bootstrap_frequency" in frame.columns:
        stable_negative = frame[
            frame["bootstrap_frequency"].ge(stable_threshold)
            & frame["conditional_contrast_signed"].lt(0)
        ].copy()
        stable_negative.to_csv(output_dir / "stable_negative_edges.csv", index=False)
    return frame


def run_global_bn_bootstrap(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    reference: StructuralEMResult,
    output_dir: Path,
) -> pd.DataFrame:
    cfg = _bn_config(config)
    bootstrap_cfg = cfg.get("bn_structure_bootstrap", {})
    if not bool(bootstrap_cfg.get("enabled", False)):
        return pd.DataFrame()
    label_map = _theme_label_map(None)
    nodes = list(roles)
    data = matrix[nodes].to_numpy(dtype=np.int8)
    n_resamples = int(bootstrap_cfg.get("n_resamples", 30))
    fraction = float(bootstrap_cfg.get("sample_fraction", bootstrap_cfg.get("fraction", 0.8)))
    n_inits = int(bootstrap_cfg.get("n_initializations_per_resample", 3))
    seed = int(bootstrap_cfg.get("random_state", config.get("random_state", 42)))
    rng = np.random.default_rng(seed)
    edge_counts: dict[tuple[str, str], int] = {}
    n_accidents = len(data)
    for resample_index in range(n_resamples):
        sample_indices = rng.choice(n_accidents, size=max(1, int(math.ceil(n_accidents * fraction))), replace=True)
        sample_matrix = pd.DataFrame(data[sample_indices], columns=nodes)
        sample_matrix.insert(0, "accident_id", sample_indices.astype(str))
        best_bic = math.inf
        best_edges: list[tuple[str, str]] = []
        for init_index in range(n_inits):
            init_seed = seed + resample_index * 1000 + init_index
            candidate = fit_global_bn(
                sample_matrix,
                roles,
                config,
                seed=init_seed,
                initialization="empty" if init_index == 0 else "random",
            )
            if candidate.bic < best_bic:
                best_bic = candidate.bic
                best_edges = list(candidate.edges)
        for edge in best_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    rows = []
    reference_edges = set(reference.edges)
    all_edges = sorted(set(edge_counts) | reference_edges)
    for parent, child in all_edges:
        rows.append({
            "parent": parent,
            "child": child,
            "parent_role": roles[parent],
            "child_role": roles[child],
            "parent_label": label_map.get(parent, parent),
            "child_label": label_map.get(child, child),
            "selection_frequency": edge_counts.get((parent, child), 0) / n_resamples,
            "in_reference_bn": (parent, child) in reference_edges,
        })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "global_bn_bootstrap_edges.csv", index=False)
    return frame


def assert_global_bn_outputs(
    result: StructuralEMResult,
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    bootstrap: pd.DataFrame,
    edges: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    cfg = _bn_config(config)
    d_max = int(cfg["d_max"])
    stable_threshold = float(cfg.get("bn_display_bootstrap_threshold", 0.60))

    assert result.n_states == 1
    assert "Z" not in result.nodes
    _assert_bn_edge_constraints(result.edges, roles, d_max)

    nodes = [column for column in matrix.columns if column != "accident_id"]
    n_accidents = len(matrix)
    expected = config.get("expected_inventory", {}).get(
        str(config.get("data", {}).get("dataset_id", config.get("dataset_id", ""))), {}
    )
    if expected:
        assert n_accidents == int(expected["n_accidents"])
        assert len(nodes) == int(expected["n_factors"])
        role_counts = {role: sum(1 for node in nodes if roles[node] == role) for role in ROLES}
        for role, count in expected.get("role_counts", {}).items():
            assert role_counts[role] == int(count)

    if not bootstrap.empty:
        assert bootstrap["selection_frequency"].between(0.0, 1.0).all()

    if not edges.empty:
        assert (edges["parent_child_observed_count"] >= 0).all()
        assert (edges["conditional_contrast_abs"] >= 0).all()
        signed = edges["conditional_contrast_signed"]
        assert ((edges["conditional_contrast_abs"] - signed.abs()) < 1e-12).all()
