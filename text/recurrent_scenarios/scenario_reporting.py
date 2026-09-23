"""Manuscript figures for global BN and empirical scenario mining."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scenario_pipeline import ROLES, StructuralEMResult, _edge_conditional_contrast_signed, _theme_label_map

ROLE_SUBTITLES = {
    "A0": "Work context",
    "A1": "Adverse condition",
    "B": "Event/deviation",
    "C": "Consequence",
}


def _count_layer_crossings(
    left_order: Sequence[str],
    right_order: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> int:
    left_pos = {node: index for index, node in enumerate(left_order)}
    right_pos = {node: index for index, node in enumerate(right_order)}
    pairs = [
        (left_pos[parent], right_pos[child])
        for parent, child in edges
        if parent in left_pos and child in right_pos
    ]
    crossings = 0
    for index, (left_a, right_a) in enumerate(pairs):
        for left_b, right_b in pairs[index + 1 :]:
            if (left_a - left_b) * (right_a - right_b) < 0:
                crossings += 1
    return crossings


def _barycenter_reorder(
    nodes: Sequence[str],
    neighbor_map: Mapping[str, Sequence[str]],
    neighbor_order: Sequence[str],
) -> list[str]:
    """Reorder ``nodes`` by mean index of their neighbors in ``neighbor_order``."""
    if len(nodes) <= 1:
        return list(nodes)
    neighbor_pos = {node: index for index, node in enumerate(neighbor_order)}
    scored: list[tuple[float, int, str]] = []
    for fallback_index, node in enumerate(nodes):
        neighbors = [neighbor_pos[item] for item in neighbor_map.get(node, ()) if item in neighbor_pos]
        score = float(np.mean(neighbors)) if neighbors else float(fallback_index)
        scored.append((score, fallback_index, node))
    scored.sort()
    return [node for _, _, node in scored]


def _order_roles_minimize_crossings(
    nodes_by_role: Mapping[str, Sequence[str]],
    edges: Sequence[tuple[str, str]],
    *,
    sweeps: int = 10,
) -> dict[str, list[str]]:
    """Barycenter heuristic over A0–A1–B–C to reduce edge crossings, especially B→C."""
    order = {role: list(nodes_by_role.get(role, [])) for role in ROLES}
    undirected_neighbors: dict[str, set[str]] = {node: set() for role in ROLES for node in order[role]}
    for parent, child in edges:
        if parent in undirected_neighbors and child in undirected_neighbors:
            undirected_neighbors[parent].add(child)
            undirected_neighbors[child].add(parent)

    def total_crossings(candidate: Mapping[str, Sequence[str]]) -> int:
        return (
            _count_layer_crossings(candidate["A0"], candidate["A1"], edges)
            + _count_layer_crossings(candidate["A0"], candidate["B"], edges)
            + _count_layer_crossings(candidate["A1"], candidate["B"], edges)
            + _count_layer_crossings(candidate["B"], candidate["C"], edges)
        )

    best = {role: list(order[role]) for role in ROLES}
    best_score = total_crossings(best)

    for _ in range(sweeps):
        # Forward sweep: children follow parents (priority: B then C).
        order["A1"] = _barycenter_reorder(
            order["A1"],
            {node: [parent for parent, child in edges if child == node] for node in order["A1"]},
            order["A0"],
        )
        order["B"] = _barycenter_reorder(
            order["B"],
            {
                node: [parent for parent, child in edges if child == node]
                for node in order["B"]
            },
            order["A0"] + order["A1"],
        )
        order["C"] = _barycenter_reorder(
            order["C"],
            {node: [parent for parent, child in edges if child == node] for node in order["C"]},
            order["B"],
        )
        # Backward sweep: parents follow children (priority: B from C).
        order["B"] = _barycenter_reorder(
            order["B"],
            {node: [child for parent, child in edges if parent == node] for node in order["B"]},
            order["C"],
        )
        order["A1"] = _barycenter_reorder(
            order["A1"],
            {node: [child for parent, child in edges if parent == node] for node in order["A1"]},
            order["B"],
        )
        order["A0"] = _barycenter_reorder(
            order["A0"],
            {node: list(undirected_neighbors.get(node, ())) for node in order["A0"]},
            order["A1"] + order["B"],
        )
        score = total_crossings(order)
        if score < best_score:
            best_score = score
            best = {role: list(order[role]) for role in ROLES}
    return best


def _display_factor_id(
    node: str,
    *,
    topic_id_lookup: Mapping[str, str] | None = None,
) -> str:
    """Return the readable discovery ``topic_id`` (``A0_001``) for a BN node.

    BN edges / scenarios use ``variable_name`` (``A0__T02`` = 1-indexed).
    Figures show ``topic_id`` (``A0_001`` = 0-indexed cluster label), which matches
    the retained-factor tables and is easier to read.
    """
    text = str(node)
    if topic_id_lookup and text in topic_id_lookup:
        return str(topic_id_lookup[text])
    if "__T" in text:
        role, _, topic = text.partition("__T")
        try:
            return f"{role}_{int(topic) - 1:03d}"
        except ValueError:
            return text
    return text


def _topic_id_lookup_from_frame(frame: pd.DataFrame | None) -> dict[str, str]:
    if frame is None or frame.empty:
        return {}
    if {"variable_name", "topic_id"}.issubset(frame.columns):
        return {
            str(row["variable_name"]): str(row["topic_id"])
            for _, row in frame.iterrows()
        }
    return {}


def _wrapped_factor_label(label: str, *, width: int = 26, max_lines: int = 3) -> str:
    cleaned = " ".join(str(label).split())
    lines = textwrap.wrap(cleaned, width=width, break_long_words=False, break_on_hyphens=True)
    if not lines:
        return cleaned
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # Prefer complete words on the last kept line; avoid mid-word truncation.
    return "\n".join(lines[:max_lines])


def render_global_bn_stable_dependencies(
    result: StructuralEMResult,
    bootstrap: pd.DataFrame | None,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_path: Path,
    *,
    edges_frame: pd.DataFrame | None = None,
    topic_id_lookup: Mapping[str, str] | None = None,
) -> None:
    """Role-layered BN figure: width = bootstrap freq, linestyle = sign(Δ̂)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from manuscript_reporting import ROLE_COLORS, ROLE_NODE_FILL, save_manuscript_figure

    threshold = float(config.get("bayesian_networks", {}).get("bn_display_bootstrap_threshold", 0.60))
    signed_lookup: dict[tuple[str, str], float] = {}
    freq_lookup: dict[tuple[str, str], float] = {}
    label_lookup = dict(label_map)
    id_lookup = dict(topic_id_lookup or {})
    id_lookup.update(_topic_id_lookup_from_frame(edges_frame))

    if edges_frame is not None and not edges_frame.empty:
        parent_col = "parent_factor" if "parent_factor" in edges_frame.columns else "parent"
        child_col = "child_factor" if "child_factor" in edges_frame.columns else "child"
        freq_col = "bootstrap_frequency" if "bootstrap_frequency" in edges_frame.columns else "selection_frequency"
        for _, row in edges_frame.iterrows():
            parent = str(row[parent_col])
            child = str(row[child_col])
            if freq_col in edges_frame.columns and pd.notna(row[freq_col]):
                freq_lookup[(parent, child)] = float(row[freq_col])
            if "conditional_contrast_signed" in edges_frame.columns and pd.notna(row["conditional_contrast_signed"]):
                signed_lookup[(parent, child)] = float(row["conditional_contrast_signed"])
            if "parent_label" in edges_frame.columns and pd.notna(row.get("parent_label")):
                label_lookup.setdefault(parent, str(row["parent_label"]))
            if "child_label" in edges_frame.columns and pd.notna(row.get("child_label")):
                label_lookup.setdefault(child, str(row["child_label"]))

    if bootstrap is not None and not bootstrap.empty:
        for _, row in bootstrap.iterrows():
            key = (str(row["parent"]), str(row["child"]))
            freq_lookup[key] = float(row["selection_frequency"])

    reference_edges = set(result.edges)
    for parent, child in reference_edges:
        if (parent, child) not in signed_lookup:
            signed_lookup[(parent, child)] = _edge_conditional_contrast_signed(result, parent, child)

    edges = [
        ((parent, child), frequency, signed_lookup[(parent, child)])
        for (parent, child), frequency in freq_lookup.items()
        if (
            frequency >= threshold
            and (parent, child) in reference_edges
            and np.isfinite(signed_lookup.get((parent, child), np.nan))
        )
    ]
    edges.sort(key=lambda item: item[1], reverse=True)

    active_nodes: set[str] = set()
    for (parent, child), _, _ in edges:
        active_nodes.add(parent)
        active_nodes.add(child)

    nodes_by_role = {
        role: sorted(node for node in active_nodes if roles.get(node) == role)
        for role in ROLES
    }
    edge_pairs = [(parent, child) for (parent, child), _, _ in edges]
    nodes_by_role = _order_roles_minimize_crossings(nodes_by_role, edge_pairs)

    node_gap = 1.08
    column_heights = {
        role: node_gap * max(len(nodes_by_role[role]) - 1, 0)
        for role in ROLES
    }
    vertical_span = max(max(column_heights.values(), default=0.0), 2.0)
    box_half_width = 0.42
    # Slightly tighter horizontal columns for A4.
    role_x = {role: 0.95 * index for index, role in enumerate(ROLES)}
    positions: dict[str, tuple[float, float]] = {}
    for role in ROLES:
        role_nodes = nodes_by_role[role]
        if not role_nodes:
            continue
        height = column_heights[role]
        if len(role_nodes) == 1:
            y_values = [0.0]
        else:
            y_values = list(np.linspace(height / 2, -height / 2, len(role_nodes)))
        for node, y_pos in zip(role_nodes, y_values):
            positions[node] = (float(role_x[role]), float(y_pos))

    fig_height = max(6.4, vertical_span + 1.55)
    figure, axis = plt.subplots(figsize=(10.8, fig_height))
    header_y = vertical_span / 2 + 0.52
    axis.set_xlim(-0.55, role_x[ROLES[-1]] + 0.55)
    axis.set_ylim(-vertical_span / 2 - 0.48, header_y + 0.14)
    axis.axis("off")

    for role in ROLES:
        x_pos = role_x[role]
        axis.text(
            x_pos,
            header_y,
            role,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=ROLE_COLORS[role],
        )
        axis.text(
            x_pos,
            header_y - 0.22,
            ROLE_SUBTITLES[role],
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
        )
        if not nodes_by_role[role]:
            axis.text(
                x_pos,
                0.0,
                f"No edge ≥ {threshold:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
                style="italic",
            )

    for node, (x_pos, y_pos) in positions.items():
        role = roles[node]
        wrapped = _wrapped_factor_label(label_lookup.get(node, node), width=28, max_lines=3)
        line_count = wrapped.count("\n") + 1
        box_height = 0.46 + 0.19 * line_count
        box = FancyBboxPatch(
            (x_pos - box_half_width, y_pos - box_height / 2),
            2 * box_half_width,
            box_height,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            facecolor=ROLE_NODE_FILL[role],
            edgecolor=ROLE_COLORS[role],
            linewidth=1.0,
            alpha=0.98,
            zorder=3,
        )
        axis.add_patch(box)
        axis.text(
            x_pos,
            y_pos + box_height * 0.30,
            _display_factor_id(node, topic_id_lookup=id_lookup),
            ha="center",
            va="center",
            fontsize=7.0,
            color="#555555",
            zorder=4,
        )
        axis.text(
            x_pos,
            y_pos - box_height * 0.05,
            wrapped,
            ha="center",
            va="center",
            fontsize=8.6,
            color="#1a1a1a",
            zorder=4,
        )

    outgoing: dict[str, list[tuple[tuple[str, str], float, float]]] = {}
    incoming: dict[str, list[tuple[tuple[str, str], float, float]]] = {}
    for edge in edges:
        (parent, child), _, _ = edge
        outgoing.setdefault(parent, []).append(edge)
        incoming.setdefault(child, []).append(edge)
    for node in outgoing:
        outgoing[node].sort(key=lambda item: positions[item[0][1]][1], reverse=True)
    for node in incoming:
        incoming[node].sort(key=lambda item: positions[item[0][0]][1], reverse=True)

    # Draw weaker edges first so strong bootstrap arcs stay readable on top.
    for (parent, child), frequency, signed in sorted(edges, key=lambda item: item[1]):
        start_x, start_y = positions[parent]
        end_x, end_y = positions[child]
        out_group = outgoing[parent]
        in_group = incoming[child]
        out_index = next(i for i, item in enumerate(out_group) if item[0] == (parent, child))
        in_index = next(i for i, item in enumerate(in_group) if item[0] == (parent, child))
        start_y = start_y + (out_index - (len(out_group) - 1) / 2) * 0.06
        end_y = end_y + (in_index - (len(in_group) - 1) / 2) * 0.06
        dy = end_y - start_y
        rad = float(np.clip(0.08 * np.tanh(dy / max(vertical_span, 1e-6)), -0.10, 0.10))
        linestyle = "solid" if signed > 0 else (0, (3.2, 2.0))
        axis.annotate(
            "",
            xy=(end_x - box_half_width, end_y),
            xytext=(start_x + box_half_width, start_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#333333",
                linewidth=0.85 + 3.1 * frequency,
                alpha=0.50 + 0.45 * frequency,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=18,
                connectionstyle=f"arc3,rad={rad}",
                linestyle=linestyle,
            ),
            zorder=2,
        )

    legend_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.6, linestyle="solid", label="Positive conditional association"),
        Line2D([0], [0], color="#333333", linewidth=1.6, linestyle=(0, (3.2, 2.0)), label="Negative conditional association"),
        Line2D([0], [0], color="#333333", linewidth=2.8, linestyle="solid", label="Edge width = bootstrap frequency"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handlelength=2.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    if not edges:
        axis.text(
            0.5,
            0.5,
            f"No stable edge with estimable contrast ≥ {threshold:.2f}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
        )

    figure.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_global_bn_stable_dependencies_from_edges(
    edges_frame: pd.DataFrame,
    config: Mapping[str, Any],
    output_path: Path,
    *,
    theme_dictionary: pd.DataFrame | None = None,
) -> None:
    """Regenerate the stable-dependencies figure from ``global_bn_edges.csv``."""
    if edges_frame.empty:
        return
    parent_col = "parent_factor" if "parent_factor" in edges_frame.columns else "parent"
    child_col = "child_factor" if "child_factor" in edges_frame.columns else "child"
    roles = {}
    label_map = {}
    reference_edges: list[tuple[str, str]] = []
    for _, row in edges_frame.iterrows():
        parent = str(row[parent_col])
        child = str(row[child_col])
        reference_edges.append((parent, child))
        if "parent_role" in edges_frame.columns:
            roles[parent] = str(row["parent_role"])
        if "child_role" in edges_frame.columns:
            roles[child] = str(row["child_role"])
        if "parent_label" in edges_frame.columns and pd.notna(row.get("parent_label")):
            label_map[parent] = str(row["parent_label"])
        if "child_label" in edges_frame.columns and pd.notna(row.get("child_label")):
            label_map[child] = str(row["child_label"])

    class _EdgeOnlyResult:
        pass

    result = _EdgeOnlyResult()
    result.edges = reference_edges
    result.nodes = sorted(roles)
    result.roles = roles

    bootstrap = pd.DataFrame(
        {
            "parent": [parent for parent, _ in reference_edges],
            "child": [child for _, child in reference_edges],
            "selection_frequency": edges_frame.get(
                "bootstrap_frequency",
                edges_frame.get("selection_frequency", 1.0),
            ),
        }
    )
    render_global_bn_stable_dependencies(
        result,  # type: ignore[arg-type]
        bootstrap,
        label_map,
        roles,
        config,
        output_path,
        edges_frame=edges_frame,
        topic_id_lookup=_topic_id_lookup_from_frame(theme_dictionary),
    )


def render_support_lift_figure(
    candidates: pd.DataFrame,
    article: pd.DataFrame,
    output_path: Path,
    *,
    min_accident_count: int = 5,
) -> None:
    """Descriptive support--lift scatter. Not used for scenario selection."""
    from manuscript_reporting import K_SELECTION_SELECTED_COLOR, save_manuscript_figure

    closed = candidates[
        candidates["is_closed_pattern"]
        & (candidates["scenario_accident_count"] >= min_accident_count)
    ].copy()
    if closed.empty:
        return
    article_ids = set(article["scenario_id"]) if not article.empty else set()

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.scatter(
        100.0 * closed["scenario_support"],
        closed["lift"],
        s=22,
        color="#D9D9D9",
        alpha=0.7,
        label=f"Closed recurrent (n>={min_accident_count})",
    )
    if not article.empty:
        article_rows = closed[closed["scenario_id"].isin(article_ids)]
        axis.scatter(
            100.0 * article_rows["scenario_support"],
            article_rows["lift"],
            s=120,
            marker="*",
            color=K_SELECTION_SELECTED_COLOR,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
            label="Highest-recurrence scenarios in main text",
        )
        for _, row in article_rows.iterrows():
            axis.annotate(
                str(row["scenario_id"]),
                (100.0 * float(row["scenario_support"]), float(row["lift"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    axis.set_title("Recurrence and consequence enrichment of retained scenarios", fontsize=11)
    axis.set_xlabel("Scenario support (%)")
    axis.set_ylabel("Lift of (upstream + B) -> C")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=False, fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_scenario_reduction_figure(
    *,
    n_admissible: int,
    n_observed: int,
    n_recurrent: int,
    n_closed: int,
    n_displayed: int,
    min_accident_count: int,
    output_path: Path,
) -> None:
    """Funnel: admissible -> observed -> recurrent -> closed -> main-text display."""
    from manuscript_reporting import save_manuscript_figure

    stages = [
        (f"{n_admissible:,}", "Admissible\nrole-complete\nconfigurations"),
        (f"{n_observed:,}", "Observed\nat least once"),
        (f"{n_recurrent:,}", f"Recurrent\nn >= {min_accident_count}"),
        (f"{n_closed:,}", "Closed recurrent\npatterns"),
        (f"{n_displayed}", f"{n_displayed} scenarios\ndisplayed in\nmain text"),
    ]
    figure, axis = plt.subplots(figsize=(11.5, 2.8))
    axis.set_xlim(0, len(stages) - 0.2)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Reduction of the recurrent-scenario search space", fontsize=11, pad=10)
    for index, (value, label) in enumerate(stages):
        x = float(index)
        axis.add_patch(
            plt.Rectangle((x - 0.35, 0.28), 0.7, 0.44, facecolor="#EEF3F8", edgecolor="#4C78A8", linewidth=1.0)
        )
        axis.text(x, 0.58, value, ha="center", va="center", fontsize=12, fontweight="bold", color="#1F4E79")
        axis.text(x, 0.12, label, ha="center", va="top", fontsize=7.5, color="#333333")
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(x + 0.55, 0.50),
                xytext=(x + 0.40, 0.50),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2),
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_learned_global_bn_graph(
    result: StructuralEMResult,
    label_map: Mapping[str, str],
    output_path: Path,
) -> None:
    """Full learned BN graph (all fitted arcs, no latent Z)."""

    try:
        import networkx as nx
    except ImportError:
        return
    from manuscript_reporting import ROLE_COLORS, ROLE_NODE_FILL, save_manuscript_figure
    from scenario_pipeline import _edge_conditional_strength

    graph = nx.DiGraph()
    graph.add_nodes_from(result.nodes)
    graph.add_edges_from(result.edges)
    if graph.number_of_edges() == 0:
        return

    strengths = {(parent, child): _edge_conditional_strength(result, parent, child) for parent, child in result.edges}
    max_strength = max(strengths.values(), default=1.0)
    positions: dict[str, tuple[float, float]] = {}
    for role_index, role in enumerate(ROLES):
        role_nodes = sorted(node for node in result.nodes if result.roles[node] == role)
        center = (len(role_nodes) - 1) / 2
        for node_index, node in enumerate(role_nodes):
            positions[node] = (role_index, center - node_index)

    labels = {node: "\n".join(textwrap.wrap(str(label_map.get(node, node)), width=24)[:2]) for node in result.nodes}
    figure, axis = plt.subplots(figsize=(max(10, 1.8 * len(ROLES)), max(6, len(result.nodes) * 0.22)))
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=[ROLE_NODE_FILL[result.roles[node]] for node in graph.nodes],
        node_size=1600,
        edgecolors=[ROLE_COLORS[result.roles[node]] for node in graph.nodes],
        linewidths=0.8,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=6)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        arrows=True,
        arrowsize=16,
        edge_color="#555555",
        width=[0.8 + 3.5 * strengths[edge] / max_strength for edge in graph.edges],
        connectionstyle="arc3,rad=0.04",
    )
    axis.set_title("Learned global Bayesian network — all fitted edges (appendix)")
    axis.text(0.5, -0.05, "All estimated edges shown | non-causal conditional dependencies", transform=axis.transAxes, ha="center", fontsize=8)
    axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def render_scenario_reporting(
    result: StructuralEMResult,
    bootstrap: pd.DataFrame | None,
    candidates: pd.DataFrame,
    article: pd.DataFrame,
    theme_dictionary: pd.DataFrame | None,
    config: Mapping[str, Any],
    figs_dir: Path,
    network_dir: Path | None = None,
    article_full: pd.DataFrame | None = None,
    *,
    recurrent_all: pd.DataFrame | None = None,
    n_admissible: int | None = None,
    n_observed: int | None = None,
) -> None:
    label_map = _theme_label_map(theme_dictionary)
    roles = result.roles
    min_count = int(config.get("scenario_mining", {}).get("scenario_min_accident_count", 5))
    render_global_bn_stable_dependencies(
        result,
        bootstrap,
        label_map,
        roles,
        config,
        figs_dir / "global_bn_stable_dependencies.png",
        topic_id_lookup=_topic_id_lookup_from_frame(theme_dictionary),
    )
    if network_dir is not None:
        network_figs = network_dir / "figures"
        network_figs.mkdir(parents=True, exist_ok=True)
        render_learned_global_bn_graph(result, label_map, network_figs / "learned_global_bn.png")

    source = recurrent_all if recurrent_all is not None else candidates
    render_support_lift_figure(
        source if "is_closed_pattern" in source.columns else candidates,
        article,
        figs_dir / "recurrent_scenarios_support_lift.png",
        min_accident_count=min_count,
    )

    n_recurrent = int((candidates["scenario_accident_count"] >= min_count).sum()) if not candidates.empty else 0
    n_closed = len(recurrent_all) if recurrent_all is not None else int(
        (
            candidates["is_closed_pattern"]
            & (candidates["scenario_accident_count"] >= min_count)
        ).sum()
    ) if not candidates.empty else 0
    render_scenario_reduction_figure(
        n_admissible=int(n_admissible if n_admissible is not None else len(candidates)),
        n_observed=int(n_observed if n_observed is not None else (candidates["scenario_accident_count"] > 0).sum()),
        n_recurrent=n_recurrent,
        n_closed=n_closed,
        n_displayed=len(article),
        min_accident_count=min_count,
        output_path=figs_dir / "recurrent_scenario_reduction.png",
    )

    figure_source = article_full if article_full is not None and not article_full.empty else article
    if not figure_source.empty and "upstream_factor_ids" in figure_source.columns:
        from scenario_figures import generate_empirical_scenario_figures
        generate_empirical_scenario_figures(figure_source, result, bootstrap, figs_dir / "recurrent_scenarios_compact.png")
