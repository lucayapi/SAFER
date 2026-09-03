"""Compact, auditable figures for latent-family accident scenarios."""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from manuscript_reporting import ROLE_COLORS, ROLE_NODE_FILL


ROLE_ORDER = ("A0", "A1", "B", "C")
ROLE_TITLES = {
    "A0": "Work context",
    "A1": "Adverse condition",
    "B": "Event / deviation",
    "C": "Consequence",
}
ROLE_LABEL_COLUMNS = {role: f"{role}_label" for role in ROLE_ORDER}


def _split_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(";") if item.strip()]


def _as_fraction(value: Any) -> float:
    numeric = float(value)
    return numeric / 100.0 if numeric > 1.0 else numeric


def normalize_scenario_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the scenario table used by the renderer."""
    required = {
        "scenario_id", "latent_family", "family_support", "global_support",
        "heading", "factor_codes",
    }
    missing = sorted(required.difference(scenarios.columns))
    if missing:
        raise ValueError(f"Missing scenario-table columns: {missing}")
    normalized = scenarios.copy()
    for role, column in ROLE_LABEL_COLUMNS.items():
        if column not in normalized.columns:
            normalized[column] = ""
    for column in ("family_support", "global_support"):
        normalized[column] = normalized[column].map(_as_fraction)
        if not normalized[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be between 0 and 1 or expressed as a percentage.")
    for column in ("scenario_id", "latent_family", "heading", "factor_codes"):
        normalized[column] = normalized[column].fillna("").astype(str)
    for column in ("N_eff", "omega", "mpe_probability", "support_enrichment_ratio"):
        if column not in normalized.columns:
            normalized[column] = float("nan")
    if "prototype_exact_mpe_match" not in normalized.columns:
        normalized["prototype_exact_mpe_match"] = np.nan
    return normalized


def _scenario_filename(scenario_id: str, latent_family: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(scenario_id)).strip("_") or "scenario"
    safe_family = re.sub(r"[^A-Za-z0-9_-]+", "_", str(latent_family)).strip("_") or "family"
    return f"scenario_{safe_id}_family_{safe_family}"


def _role_from_code(code: str) -> str | None:
    for role in ROLE_ORDER:
        if code.startswith(f"{role}__") or code.startswith(f"{role}_"):
            return role
    return None


def _role_entries(row: pd.Series) -> list[tuple[str, str, str]]:
    """Return one entry per positive MPE factor, in process order."""
    codes = _split_values(row["factor_codes"])
    entries: list[tuple[str, str, str]] = []
    for role in ROLE_ORDER:
        role_codes = [code for code in codes if _role_from_code(code) == role]
        labels = _split_values(row[ROLE_LABEL_COLUMNS[role]])
        for index, code in enumerate(role_codes):
            label = labels[index] if index < len(labels) else code
            entries.append((role, label, code))
    return entries


def _edge_set(learned_edges: Iterable[tuple[str, str]] | pd.DataFrame | None) -> set[tuple[str, str]]:
    if learned_edges is None:
        return set()
    if isinstance(learned_edges, pd.DataFrame):
        if not {"parent", "child"}.issubset(learned_edges.columns):
            return set()
        return set(zip(learned_edges["parent"].astype(str), learned_edges["child"].astype(str)))
    return {(str(parent), str(child)) for parent, child in learned_edges}


def _draw_compact_path(axis, entries, learned_edges: set[tuple[str, str]], *, y: float = 0.5) -> None:
    if not entries:
        return
    count = len(entries)
    spacing = 0.82 / max(count - 1, 1)
    centers = [0.09 + index * spacing for index in range(count)] if count > 1 else [0.5]
    box_width = min(0.18, 0.76 / max(count, 1))
    box_height = 0.52
    for index, (role, label, code) in enumerate(entries):
        center_x = centers[index]
        left = center_x - box_width / 2
        axis.text(center_x, 0.92, role, ha="center", va="top", fontsize=8.5, fontweight="bold", color=ROLE_COLORS[role])
        box = FancyBboxPatch(
            (left, y - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=ROLE_NODE_FILL[role], edgecolor=ROLE_COLORS[role], linewidth=0.9,
            alpha=0.94, transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(center_x, y + 0.10, code, ha="center", va="center", fontsize=8.2, fontweight="bold", color="white", transform=axis.transAxes)
        axis.text(center_x, y - 0.08, "\n".join(textwrap.wrap(str(label), width=18)), ha="center", va="center", fontsize=7.8, color="white", transform=axis.transAxes)
        if index == 0:
            continue
        previous_code = entries[index - 1][2]
        start_x = centers[index - 1] + box_width / 2 + 0.01
        end_x = center_x - box_width / 2 - 0.01
        direct_edge = (previous_code, code) in learned_edges
        axis.add_patch(FancyArrowPatch(
            (start_x, y), (end_x, y), transform=axis.transAxes,
            arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
            linestyle="-" if direct_edge else "--",
            color="#333333" if direct_edge else "#777777",
        ))


def _metadata_text(row: pd.Series) -> str:
    n_eff = row.get("N_eff")
    omega = row.get("omega")
    if pd.notna(n_eff) and pd.notna(omega):
        return f"N_eff = {float(n_eff):.1f} accidents ({float(omega):.1%})"
    return "Effective family size not provided"


def render_scenario_figure(row: pd.Series, output_dir: Path, *, learned_edges=None, dpi: int = 300) -> tuple[Path, Path]:
    """Render one compact scenario to PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _role_entries(row)
    if not entries:
        raise ValueError(f"Scenario {row['scenario_id']} contains no positive factors.")
    figure, axis = plt.subplots(figsize=(11.0, 3.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("white")
    axis.text(0.5, 0.98, f"Scenario {row['scenario_id']} — Latent family {row['latent_family']}", ha="center", va="top", fontsize=15, fontweight="bold")
    axis.text(0.5, 0.88, _metadata_text(row), ha="center", va="top", fontsize=9.5, color="#444444")
    axis.text(0.5, 0.81, f"Family support: {float(row['family_support']):.2%}   |   Global support: {float(row['global_support']):.2%}", ha="center", va="top", fontsize=9.5, color="#444444")
    _draw_compact_path(axis, entries, _edge_set(learned_edges), y=0.48)
    axis.text(0.5, 0.08, textwrap.fill(str(row["heading"]), width=105), ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222222")
    figure.tight_layout()
    stem = _scenario_filename(row["scenario_id"], row["latent_family"])
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    from manuscript_reporting import save_manuscript_figure

    save_manuscript_figure(figure, png_path, dpi=dpi)
    save_manuscript_figure(figure, pdf_path, dpi=dpi)
    plt.close(figure)
    return png_path, pdf_path


def render_compact_scenarios_figure(scenarios: pd.DataFrame, output_dir: str | Path, *, learned_edges=None, stem: str = "recurrent_scenarios_compact", dpi: int = 300) -> tuple[Path, Path]:
    """Render one publication-style table row per selected latent family."""
    normalized = normalize_scenario_table(scenarios)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = list(normalized.iterrows())
    if not rows:
        raise ValueError("The scenario table is empty.")
    edge_set = _edge_set(learned_edges)
    figure = plt.figure(figsize=(17.0, max(3.3, 2.25 * len(rows) + 1.0)))
    grid = figure.add_gridspec(
        len(rows) + 1, 8,
        height_ratios=[0.34] + [1.0] * len(rows),
        width_ratios=[0.85, 1.0, 1.35, 1.15, 1.15, 1.05, 0.85, 3.8],
        hspace=0.18, wspace=0.06,
    )
    headers = ("Scenario", "Weight / N_eff", "A0/A1", "B", "C", "Supports", "Enrichment", "MPE path")
    for column, header in enumerate(headers):
        header_axis = figure.add_subplot(grid[0, column])
        header_axis.axis("off")
        header_axis.text(0.5, 0.5, header, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#333333")
    for row_index, (_, row) in enumerate(rows, start=1):
        upstream_labels = []
        for role in ("A0", "A1"):
            column = ROLE_LABEL_COLUMNS[role]
            labels = _split_values(row.get(column, ""))
            if labels:
                upstream_labels.extend(labels)
        upstream_text = "\n".join(
            "\n".join(textwrap.wrap(str(label), width=22)) for label in upstream_labels
        ) if upstream_labels else "—"
        role_labels = [upstream_text]
        for role in ("B", "C"):
            column = ROLE_LABEL_COLUMNS[role]
            labels = _split_values(row.get(column, ""))
            if labels:
                wrapped = ["\n".join(textwrap.wrap(str(label), width=22)) for label in labels]
                role_labels.append("\n".join(wrapped))
            else:
                role_labels.append("—")
        enrichment = row.get("support_enrichment_ratio")
        enrichment_text = f"{float(enrichment):.2f}×" if pd.notna(enrichment) else "—"
        exact_proto = row.get("prototype_exact_mpe_match")
        if pd.notna(exact_proto):
            proto_text = f"\nExact prototype: {'Yes' if bool(exact_proto) else 'No'}"
        else:
            proto_text = ""
        text_values = [
            f"{row['scenario_id']}\nFamily {row['latent_family']}",
            _metadata_text(row),
            *role_labels,
            f"Family {float(row['family_support']):.1%}\nGlobal {float(row['global_support']):.1%}{proto_text}",
            enrichment_text,
        ]
        for column, value in enumerate(text_values):
            cell_axis = figure.add_subplot(grid[row_index, column])
            cell_axis.axis("off")
            cell_axis.text(
                0.5, 0.5, value, ha="center", va="center", fontsize=8.0,
                fontweight="bold" if column == 0 else "normal", color="#222222",
            )
        diagram_axis = figure.add_subplot(grid[row_index, 7])
        diagram_axis.set_xlim(0, 1)
        diagram_axis.set_ylim(0, 1)
        diagram_axis.axis("off")
        _draw_compact_path(diagram_axis, _role_entries(row), edge_set, y=0.49)
    figure.text(
        0.5, 0.012,
        "Solid arrow: learned BN dependency. Dashed arrow: functional role order (not identified causality).",
        ha="center", fontsize=8.5, color="#666666",
    )
    figure.subplots_adjust(top=0.96, bottom=0.06, left=0.02, right=0.99)
    png_path = destination / f"{stem}.png"
    pdf_path = destination / f"{stem}.pdf"
    from manuscript_reporting import save_manuscript_figure

    save_manuscript_figure(figure, png_path, dpi=dpi)
    save_manuscript_figure(figure, pdf_path, dpi=dpi)
    plt.close(figure)
    return png_path, pdf_path


def generate_empirical_scenario_figures(
    article: pd.DataFrame,
    result,
    bootstrap: pd.DataFrame | None,
    output_path: Path,
    *,
    stable_threshold: float = 0.60,
) -> None:
    """Compact upstream -> B -> C panels for article scenarios ranked by recurrence."""

    from manuscript_reporting import save_manuscript_figure
    from scenario_pipeline import _edge_conditional_contrast_signed

    if article.empty:
        return
    article = article.sort_values(
        ["scenario_accident_count", "scenario_support", "lift", "confidence", "scenario_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    bootstrap_freq: dict[tuple[str, str], float] = {}
    if bootstrap is not None and not bootstrap.empty:
        bootstrap_freq = {
            (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
            for _, row in bootstrap.iterrows()
        }
    stable_positive = {
        (parent, child)
        for parent, child in result.edges
        if _edge_conditional_contrast_signed(result, parent, child) > 0
        and bootstrap_freq.get((parent, child), 0.0) >= stable_threshold
    }

    n_rows = len(article)
    figure, axes = plt.subplots(n_rows, 1, figsize=(10, max(2.2, 1.8 * n_rows)), squeeze=False)
    for axis, (_, row) in zip(axes.flat, article.iterrows()):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        upstream = [part for part in str(row["upstream_factor_ids"]).split(" | ") if part]
        b_factor = str(row["B_factor_id"])
        c_factor = str(row["C_factor_id"])
        ordered = upstream + [b_factor, c_factor]
        labels = {
            node: (
                str(row["upstream_labels"]) if node in upstream else
                str(row["B_label"]) if node == b_factor else str(row["C_label"])
            )
            for node in ordered
        }
        if len(upstream) == 2:
            x_positions = [0.12, 0.32, 0.62, 0.88]
        elif len(upstream) == 1:
            x_positions = [0.18, 0.50, 0.82]
        else:
            x_positions = [0.25, 0.55, 0.85]
        y = 0.58
        for x_pos, node in zip(x_positions, ordered):
            role = result.roles[node]
            box = FancyBboxPatch(
                (x_pos - 0.08, y - 0.12), 0.16, 0.24,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                transform=axis.transAxes,
                facecolor=ROLE_NODE_FILL[role], edgecolor=ROLE_COLORS[role], linewidth=0.9,
            )
            axis.add_patch(box)
            wrapped = "\n".join(textwrap.wrap(labels[node], width=16)[:2])
            axis.text(x_pos, y, wrapped, ha="center", va="center", fontsize=7, color="#222222", transform=axis.transAxes)
        for left, right in zip(ordered, ordered[1:]):
            left_x = x_positions[ordered.index(left)]
            right_x = x_positions[ordered.index(right)]
            linestyle = "-" if (left, right) in stable_positive else "--"
            axis.annotate(
                "",
                xy=(right_x - 0.08, y),
                xytext=(left_x + 0.08, y),
                xycoords=axis.transAxes,
                textcoords=axis.transAxes,
                arrowprops=dict(arrowstyle="-|>", linestyle=linestyle, color="#333333", linewidth=1.2),
            )
        axis.text(
            0.5, 0.12,
            f"n={int(row['scenario_accident_count'])}  support={100 * float(row['scenario_support']):.1f}%  "
            f"conf={100 * float(row['confidence']):.1f}%  lift={float(row['lift']):.1f}x",
            ha="center", va="center", fontsize=7.5, transform=axis.transAxes,
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, output_path, dpi=220)
    save_manuscript_figure(figure, output_path.with_suffix(".pdf"), dpi=220)
    plt.close(figure)


def generate_scenario_figures(scenarios: pd.DataFrame | str | Path, output_dir: str | Path, *, learned_edges=None) -> pd.DataFrame:
    """Write the compact summary and one PNG/PDF figure per scenario."""
    if isinstance(scenarios, (str, Path)):
        scenarios = pd.read_csv(scenarios)
    normalized = normalize_scenario_table(scenarios)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("scenario_*.png", "scenario_*.pdf"):
        for stale_file in destination.glob(pattern):
            stale_file.unlink()
    normalized.to_csv(destination / "scenarios_summary.csv", index=False)
    render_compact_scenarios_figure(normalized, destination, learned_edges=learned_edges)
    if destination.name == "scenario_figures":
        figures_dir = destination.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for name in ("recurrent_scenarios_compact.png", "recurrent_scenarios_compact.pdf"):
            source = destination / name
            if source.is_file():
                target = figures_dir / name
                target.write_bytes(source.read_bytes())
    for _, row in normalized.iterrows():
        render_scenario_figure(row, destination, learned_edges=learned_edges)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Normalized scenario CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    arguments = parser.parse_args()
    generate_scenario_figures(arguments.input, arguments.output_dir)


if __name__ == "__main__":
    main()
