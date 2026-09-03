"""Manuscript figures for global BN and empirical scenario mining."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scenario_pipeline import ROLES, StructuralEMResult, _edge_conditional_contrast_signed, _theme_label_map


def render_global_bn_stable_dependencies(
    result: StructuralEMResult,
    bootstrap: pd.DataFrame | None,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_path: Path,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from manuscript_reporting import ROLE_COLORS, ROLE_NODE_FILL, format_bootstrap_frequency, save_manuscript_figure

    threshold = float(config.get("bayesian_networks", {}).get("bn_display_bootstrap_threshold", 0.60))
    freq_lookup: dict[tuple[str, str], float] = {}
    if bootstrap is not None and not bootstrap.empty:
        freq_lookup = {
            (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
            for _, row in bootstrap.iterrows()
        }
    signed_lookup = {
        (parent, child): _edge_conditional_contrast_signed(result, parent, child)
        for parent, child in result.edges
    }
    edges = [
        ((parent, child), frequency, signed_lookup[(parent, child)])
        for (parent, child), frequency in freq_lookup.items()
        if frequency >= threshold and (parent, child) in set(result.edges)
    ]
    edges.sort(key=lambda item: item[1], reverse=True)

    active_nodes: set[str] = set()
    for (parent, child), _, _ in edges:
        active_nodes.add(parent)
        active_nodes.add(child)

    nodes_by_role = {role: sorted(node for node in active_nodes if roles.get(node) == role) for role in ROLES}
    max_nodes = max((len(nodes_by_role[role]) for role in ROLES), default=1)
    vertical_spacing = 0.95
    positions: dict[str, tuple[float, float]] = {}
    for role_index, role in enumerate(ROLES):
        for node_index, node in enumerate(nodes_by_role[role]):
            start_y = (len(nodes_by_role[role]) - 1) / 2 * vertical_spacing
            positions[node] = (float(role_index), start_y - node_index * vertical_spacing)

    figure, axis = plt.subplots(figsize=(max(6.5, 1.7 * len(ROLES)), max(3.2, max_nodes * 0.82 + 1.0)))
    y_span = max(max_nodes - 0.5, 0.5) * vertical_spacing
    header_y = y_span + 0.45
    axis.set_xlim(-0.55, len(ROLES) - 0.45)
    axis.set_ylim(-y_span - 0.35, header_y + 0.55)
    axis.axis("off")
    axis.set_title("Stable dependencies in the learned Bayesian network", fontsize=11, fontweight="bold", pad=8)

    for role_index, role in enumerate(ROLES):
        axis.text(role_index, header_y, role, ha="center", va="center", fontsize=10, fontweight="bold", color=ROLE_COLORS[role])
        if not nodes_by_role[role]:
            axis.text(role_index, header_y - 0.28, f"No edge >= {threshold:.2f}", ha="center", va="top", fontsize=7, color="#888888", style="italic")

    for node, (x_pos, y_pos) in positions.items():
        role = roles[node]
        label_text = str(label_map.get(node, node))
        wrapped = "\n".join(textwrap.wrap(label_text, width=22, break_long_words=False)[:3])
        line_count = wrapped.count("\n") + 1
        box_height = 0.28 + 0.11 * line_count
        box = FancyBboxPatch(
            (x_pos - 0.38, y_pos - box_height / 2), 0.76, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=ROLE_NODE_FILL[role], edgecolor=ROLE_COLORS[role], linewidth=0.9, alpha=0.98,
        )
        axis.add_patch(box)
        axis.text(x_pos, y_pos, wrapped, ha="center", va="center", fontsize=6.4, color="#222222")

    for edge_index, ((parent, child), frequency, signed) in enumerate(edges):
        start_x, start_y = positions[parent]
        end_x, end_y = positions[child]
        rad = 0.12 if edge_index % 2 == 0 else -0.12
        linestyle = "solid" if signed > 0 else "dashed"
        axis.annotate(
            "",
            xy=(end_x - 0.38, end_y),
            xytext=(start_x + 0.38, start_y),
            arrowprops=dict(
                arrowstyle="-|>", color="#333333", linewidth=0.8 + 3.2 * frequency,
                alpha=0.55 + 0.45 * frequency, shrinkA=0, shrinkB=0,
                connectionstyle=f"arc3,rad={rad}", linestyle=linestyle,
            ),
        )
        mid_x = (start_x + end_x) / 2
        mid_y = max(start_y, end_y) + 0.18 + 0.04 * (edge_index % 3)
        axis.text(mid_x, mid_y, format_bootstrap_frequency(frequency), ha="center", va="bottom", fontsize=7, color="#333333")

    legend_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.5, linestyle="solid", label="Positive conditional association"),
        Line2D([0], [0], color="#333333", linewidth=1.5, linestyle="dashed", label="Negative conditional association"),
        Line2D([0], [0], color="#333333", linewidth=2.5, linestyle="solid", label="Edge width = bootstrap frequency"),
    ]
    axis.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7)

    if not edges:
        axis.text(0.5, 0.5, f"No stable edge >= {threshold:.2f}", transform=axis.transAxes, ha="center", va="center", fontsize=8, color="#666666")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    pdf_path = output_path.with_suffix(".pdf")
    save_manuscript_figure(figure, pdf_path, dpi=220)
    plt.close(figure)


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
        result, bootstrap, label_map, roles, config, figs_dir / "global_bn_stable_dependencies.png",
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
