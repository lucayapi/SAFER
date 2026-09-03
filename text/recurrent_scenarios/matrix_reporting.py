"""Exports CSV and figures for the frozen accident × factor matrix."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manuscript_reporting import ROLE_COLORS, ROLE_NODE_FILL, save_manuscript_figure
from scenario_pipeline import BN_ROLE_ARCS, ROLES, _theme_label_map


def write_matrix_derived_tables(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    theme_dictionary: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write supplementary CSV tables derived from the frozen matrix."""

    output_dir.mkdir(parents=True, exist_ok=True)
    factor_columns = [column for column in matrix.columns if column != "accident_id"]
    data = matrix[factor_columns].to_numpy(dtype=np.int8)
    n_accidents = len(matrix)
    label_map = _theme_label_map(theme_dictionary)

    profile_rows = []
    for _, row in audit.iterrows():
        profile_rows.append({
            "accident_id": row["accident_id"],
            "included_in_bn": bool(row["included_in_bn"]),
            "n_observed_factors": int(row["n_observed_factors"]),
            "n_A0": int(row["n_A0"]),
            "n_A1": int(row["n_A1"]),
            "n_B": int(row["n_B"]),
            "n_C": int(row["n_C"]),
            "has_all_roles": int(row["n_A0"] > 0 and row["n_A1"] > 0 and row["n_B"] > 0 and row["n_C"] > 0),
        })
    profile_path = output_dir / "accident_factor_profile_summary.csv"
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False)

    cooccurrence_rows = []
    for left_index, left in enumerate(factor_columns):
        left_role = roles[left]
        left_count = int(data[:, left_index].sum())
        for right_index in range(left_index + 1, len(factor_columns)):
            right = factor_columns[right_index]
            right_role = roles[right]
            joint = int((data[:, left_index] & data[:, right_index]).sum())
            if joint == 0:
                continue
            right_count = int(data[:, right_index].sum())
            union = left_count + right_count - joint
            cooccurrence_rows.append({
                "factor_a": left,
                "factor_a_label": label_map.get(left, left),
                "factor_a_role": left_role,
                "factor_b": right,
                "factor_b_label": label_map.get(right, right),
                "factor_b_role": right_role,
                "role_pair": f"{left_role}-{right_role}",
                "allowed_role_pair": (left_role, right_role) in BN_ROLE_ARCS or (right_role, left_role) in BN_ROLE_ARCS,
                "cooccurrence_count": joint,
                "cooccurrence_rate": joint / n_accidents,
                "jaccard": joint / union if union else 0.0,
            })
    cooccurrence_path = output_dir / "factor_pair_cooccurrence.csv"
    cooccurrence = pd.DataFrame(cooccurrence_rows)
    if not cooccurrence.empty:
        cooccurrence = cooccurrence.sort_values(["cooccurrence_count", "jaccard"], ascending=False)
    cooccurrence.to_csv(cooccurrence_path, index=False)

    role_summary_rows = []
    for role in ROLES:
        role_columns = [column for column in factor_columns if roles.get(column) == role]
        role_data = data[:, [factor_columns.index(column) for column in role_columns]]
        observed_per_accident = role_data.sum(axis=1) if role_columns else np.zeros(n_accidents)
        role_summary_rows.append({
            "role": role,
            "n_factors": len(role_columns),
            "mean_factors_per_accident": float(observed_per_accident.mean()) if len(role_columns) else 0.0,
            "median_factors_per_accident": float(np.median(observed_per_accident)) if len(role_columns) else 0.0,
            "max_factors_per_accident": int(observed_per_accident.max()) if len(role_columns) else 0,
            "accidents_with_any_factor": int((observed_per_accident > 0).sum()) if len(role_columns) else 0,
        })
    role_summary_path = output_dir / "role_factor_summary.csv"
    pd.DataFrame(role_summary_rows).to_csv(role_summary_path, index=False)

    n_cells = n_accidents * len(factor_columns)
    density_path = output_dir / "matrix_density_summary.json"
    density_payload = {
        "n_accidents": n_accidents,
        "n_factors": len(factor_columns),
        "n_observed_cells": int(data.sum()),
        "n_cells": int(n_cells),
        "matrix_density": float(data.sum() / n_cells) if n_cells else 0.0,
        "accidents_with_all_roles": int(pd.DataFrame(profile_rows)["has_all_roles"].sum()),
        "role_counts": {role: int(sum(roles[column] == role for column in factor_columns)) for role in ROLES},
    }
    density_path.write_text(json.dumps(density_payload, indent=2), encoding="utf-8")

    return {
        "accident_factor_profile_summary": profile_path,
        "factor_pair_cooccurrence": cooccurrence_path,
        "role_factor_summary": role_summary_path,
        "matrix_density_summary": density_path,
    }


def _short_label(text: str, width: int = 28) -> str:
    wrapped = textwrap.wrap(str(text), width=width)
    return "\n".join(wrapped[:2])


def render_conceptual_role_architecture(output_path: Path) -> None:
    """Role-level DAG for the global BN (no latent Z)."""

    try:
        import networkx as nx
    except ImportError:
        return

    graph = nx.DiGraph(list(BN_ROLE_ARCS))
    positions = {"A0": (0.0, 0.5), "A1": (1.0, 1.0), "B": (2.0, 0.5), "C": (3.0, 0.5)}
    labels = {
        "A0": "A0\nContexte",
        "A1": "A1\nCondition adverse",
        "B": "B\nÉvénement",
        "C": "C\nConséquence",
    }
    figure, axis = plt.subplots(figsize=(10, 4))
    nx.draw_networkx(
        graph,
        positions,
        labels=labels,
        ax=axis,
        node_color=[ROLE_NODE_FILL[role] for role in graph.nodes],
        edgecolors=[ROLE_COLORS[role] for role in graph.nodes],
        linewidths=1.2,
        node_size=3200,
        font_size=9,
        arrows=True,
        arrowsize=22,
        width=2.0,
        edge_color="#555555",
    )
    axis.set_title("Architecture role-contrainte du BN global (sans Z)")
    axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_factor_prevalence_figure(
    theme_dictionary: pd.DataFrame,
    prevalence: pd.DataFrame,
    output_path: Path,
) -> None:
    """Horizontal bar chart of factor observation prevalence, colored by role."""

    if prevalence.empty:
        return
    frame = prevalence.merge(
        theme_dictionary[["variable_name", "topic_label", "role"]],
        on=["variable_name", "role"],
        how="left",
        suffixes=("", "_dict"),
    )
    frame = frame.sort_values(["role", "observation_prevalence"], ascending=[True, False])
    frame["display_label"] = frame["topic_label"].fillna(frame["variable_name"]).map(lambda value: _short_label(value, 34))

    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.28 * len(frame) + 1.2)))
    y_positions = np.arange(len(frame))
    colors = [ROLE_COLORS.get(role, "#888888") for role in frame["role"]]
    axis.barh(y_positions, 100.0 * frame["observation_prevalence"], color=colors, alpha=0.85)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(frame["display_label"], fontsize=7)
    axis.set_xlabel("Prevalence d'observation (%)")
    axis.set_title("Prevalence des facteurs retenus dans la matrice")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_accident_factor_count_figure(audit: pd.DataFrame, output_path: Path) -> None:
    """Boxplots of observed factor counts per accident, by role."""

    included = audit[audit["included_in_bn"]].copy()
    if included.empty:
        return
    count_columns = [("n_A0", "A0"), ("n_A1", "A1"), ("n_B", "B"), ("n_C", "C")]
    data = [included[column].to_numpy(dtype=float) for column, _ in count_columns]
    labels = [role for _, role in count_columns]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    box = axis.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, role in zip(box["boxes"], labels):
        patch.set_facecolor(ROLE_NODE_FILL[role])
        patch.set_edgecolor(ROLE_COLORS[role])
    axis.set_ylabel("Nombre de facteurs observés par accident")
    axis.set_title("Distribution du nombre de facteurs observés par rôle")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_cooccurrence_heatmap(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    theme_dictionary: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 24,
) -> None:
    """Heatmap of pairwise co-occurrence for the most prevalent factors."""

    factor_columns = [column for column in matrix.columns if column != "accident_id"]
    if len(factor_columns) < 2:
        return
    prevalence = matrix[factor_columns].mean().sort_values(ascending=False)
    selected = prevalence.head(min(top_n, len(factor_columns))).index.tolist()
    submatrix = matrix[selected].to_numpy(dtype=np.int8)
    co_matrix = submatrix.T @ submatrix
    label_map = _theme_label_map(theme_dictionary)
    tick_labels = [_short_label(label_map.get(name, name), 18) for name in selected]
    role_colors = [ROLE_COLORS.get(roles[name], "#888888") for name in selected]

    figure, axis = plt.subplots(figsize=(max(8, 0.45 * len(selected)), max(7, 0.45 * len(selected))))
    image = axis.imshow(co_matrix, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(selected)))
    axis.set_yticks(range(len(selected)))
    axis.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    axis.set_yticklabels(tick_labels, fontsize=6)
    for tick, color in zip(axis.get_xticklabels(), role_colors):
        tick.set_color(color)
    for tick, color in zip(axis.get_yticklabels(), role_colors):
        tick.set_color(color)
    axis.set_title(f"Absolute co-occurrence counts — top {len(selected)} factors")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_cooccurrence_network(
    cooccurrence: pd.DataFrame,
    roles: Mapping[str, str],
    label_map: Mapping[str, str],
    output_path: Path,
    *,
    min_count: int = 5,
    max_edges: int = 40,
) -> None:
    """Network of frequent factor co-occurrences (allowed role pairs only)."""

    if cooccurrence.empty:
        return
    try:
        import networkx as nx
    except ImportError:
        return

    edges_frame = cooccurrence[
        cooccurrence["allowed_role_pair"]
        & (cooccurrence["cooccurrence_count"] >= min_count)
    ].head(max_edges)
    if edges_frame.empty:
        return

    graph = nx.Graph()
    for _, row in edges_frame.iterrows():
        graph.add_edge(
            str(row["factor_a"]),
            str(row["factor_b"]),
            weight=float(row["cooccurrence_count"]),
        )

    figure, axis = plt.subplots(figsize=(14, max(8, 0.25 * graph.number_of_nodes() + 4)))
    positions = nx.spring_layout(graph, seed=42, k=1.4 / max(graph.number_of_nodes(), 1))
    node_colors = [ROLE_NODE_FILL.get(roles.get(node, ""), "#DDDDDD") for node in graph.nodes]
    edge_widths = [
        0.5 + 3.0 * graph[u][v]["weight"] / edges_frame["cooccurrence_count"].max()
        for u, v in graph.edges
    ]
    labels = {node: _short_label(label_map.get(node, node), 16) for node in graph.nodes}

    nx.draw_networkx_nodes(graph, positions, ax=axis, node_color=node_colors, node_size=900, edgecolors="#444444", linewidths=0.6)
    nx.draw_networkx_edges(graph, positions, ax=axis, width=edge_widths, edge_color="#666666", alpha=0.7)
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=6)
    axis.set_title(f"Réseau de co-occurrence (>= {min_count} accidents, paires de rôles admises)")
    axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_matrix_reporting(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    theme_dictionary: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path,
    *,
    cooccurrence: pd.DataFrame | None = None,
    include_diagnostic_figures: bool = False,
) -> dict[str, Path]:
    """Write matrix figures under ``output_dir/figures``."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    prevalence = pd.read_csv(output_dir / "factor_prevalence.csv")
    if cooccurrence is None:
        cooccurrence_path = output_dir / "factor_pair_cooccurrence.csv"
        cooccurrence = pd.read_csv(cooccurrence_path) if cooccurrence_path.is_file() else pd.DataFrame()

    outputs = {
        "conceptual_role_architecture": figures_dir / "conceptual_role_architecture.png",
        "factor_prevalence_by_role": figures_dir / "factor_prevalence_by_role.png",
        "accident_factor_count_by_role": figures_dir / "accident_factor_count_by_role.png",
    }
    render_conceptual_role_architecture(outputs["conceptual_role_architecture"])
    render_factor_prevalence_figure(theme_dictionary, prevalence, outputs["factor_prevalence_by_role"])
    render_accident_factor_count_figure(audit, outputs["accident_factor_count_by_role"])
    if include_diagnostic_figures:
        outputs["factor_cooccurrence_heatmap"] = figures_dir / "factor_cooccurrence_heatmap.png"
        outputs["factor_cooccurrence_network"] = figures_dir / "factor_cooccurrence_network.png"
        render_cooccurrence_heatmap(matrix, roles, theme_dictionary, outputs["factor_cooccurrence_heatmap"])
        label_map = _theme_label_map(theme_dictionary)
        render_cooccurrence_network(cooccurrence, roles, label_map, outputs["factor_cooccurrence_network"])
    return outputs


def export_matrix_artifacts(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    theme_dictionary: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path,
    *,
    include_diagnostic_figures: bool = False,
) -> dict[str, Path]:
    """Convenience wrapper: derived CSV tables + matrix figures."""

    tables = write_matrix_derived_tables(matrix, roles, theme_dictionary, audit, output_dir)
    cooccurrence = pd.read_csv(tables["factor_pair_cooccurrence"])
    if not cooccurrence.empty:
        assert (cooccurrence["cooccurrence_count"] >= 0).all()
    figures = render_matrix_reporting(
        matrix, roles, theme_dictionary, audit, output_dir,
        cooccurrence=cooccurrence, include_diagnostic_figures=include_diagnostic_figures,
    )
    return {**tables, **figures}
