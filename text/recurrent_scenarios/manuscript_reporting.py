"""Manuscript-oriented tables and figures for topic-discovery results notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROLES = ("A0", "A1", "B", "C")

ROLE_COLORS = {
    "A0": "#4C78A8",
    "A1": "#DE2D26",
    "B": "#31A354",
    "C": "#E45756",
    "Z": "#CAB2D6",
}

ROLE_NODE_FILL = {
    "A0": "#C6DBEF",
    "A1": "#FEE0D2",
    "B": "#E5F5E0",
    "C": "#FCAEA1",
    "Z": "#E7E0EC",
}

ROLE_BOXPLOT_PROPS = {
    "A0": {"facecolor": "#C6DBEF", "edgecolor": "#4C78A8", "mediancolor": "#D62728"},
    "A1": {"facecolor": "#FEE0D2", "edgecolor": "#DE2D26", "mediancolor": "#A50F15"},
    "B": {"facecolor": "#E5F5E0", "edgecolor": "#31A354", "mediancolor": "#006D2C"},
    "C": {"facecolor": "#FCAEA1", "edgecolor": "#E45756", "mediancolor": "#A50F15"},
}

K_SELECTION_ADMISSIBLE_COLOR = ROLE_COLORS["A0"]
K_SELECTION_BEST_LINE_COLOR = "#1F4E79"
K_SELECTION_SELECTED_COLOR = "#D62728"


def role_color(role: str) -> str:
    """Return the manuscript node fill color for one accident-process role."""
    return ROLE_NODE_FILL.get(role, "#DDDDDD")


def role_boxplot_kwargs(role: str) -> dict[str, dict[str, str | float]]:
    """Matplotlib boxplot styling kwargs for one role (topic-modeling palette)."""
    props = ROLE_BOXPLOT_PROPS.get(
        role,
        {"facecolor": "#DDDDDD", "edgecolor": "#333333", "mediancolor": "#333333"},
    )
    return {
        "boxprops": {"facecolor": props["facecolor"], "edgecolor": props["edgecolor"]},
        "medianprops": {"color": props["mediancolor"], "linewidth": 1.4},
    }


def format_bootstrap_frequency(frequency: float) -> str:
    """Compact bootstrap frequency label for arc annotations (e.g. 1.00, .83)."""
    if frequency >= 0.995:
        return "1.00"
    text = f"{frequency:.2f}"
    if text.startswith("0"):
        return text[1:]
    return text


ROLE_RETAINED_FACTOR_TITLES = {
    "A0": "A0 — Work-context factors",
    "A1": "A1 — Adverse-condition factors",
    "B": "B — Event/deviation factors",
    "C": "C — Consequence factors",
}

ROLE_RETAINED_FACTOR_TITLES_SHORT = {
    "A0": "A0 — Work context",
    "A1": "A1 — Adverse conditions",
    "B": "B — Events/deviations",
    "C": "C — Consequences",
}

# Panel titles for multi-role manuscript figures (lettered subplots).
ROLE_PANEL_TITLES = {
    "A0": "A0 – Work context",
    "A1": "A1 – Adverse condition",
    "B": "B – Event/deviation",
    "C": "C – Consequence",
}

ROLE_PANEL_LETTERS = ("a", "b", "c", "d")


def role_panel_title(role: str, *, index: int | None = None) -> str:
    """Return ``(a) A0 – Work context``-style panel title."""
    label = ROLE_PANEL_TITLES.get(role, str(role))
    if index is None:
        try:
            index = list(ROLES).index(role)
        except ValueError:
            index = 0
    letter = ROLE_PANEL_LETTERS[index] if 0 <= index < len(ROLE_PANEL_LETTERS) else chr(ord("a") + index)
    return f"({letter}) {label}"

FIGURE_STABILITY_LANDSCAPE = "stability_landscape_all_roles.png"
FIGURE_PARETO_NORMALIZED = "pareto_normalized_knee_all_roles.png"
FIGURE_FACTOR_RESAMPLING_A0 = "factor_resampling_A0.png"
FIGURE_FACTOR_RESAMPLING_A1_B_C = "factor_resampling_A1_B_C.png"
FIGURE_UMAP_SEED_SENSITIVITY = "umap_seed_sensitivity_all_roles.png"

MANUSCRIPT_FIGURE_PAD_INCHES = 0.02


def save_manuscript_figure(
    figure,
    output_path: Path | str,
    *,
    dpi: float = 250,
    facecolor: str = "white",
    pad_inches: float | None = None,
) -> None:
    """Save a figure with tight bounding box and minimal outer padding."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=MANUSCRIPT_FIGURE_PAD_INCHES if pad_inches is None else pad_inches,
        facecolor=facecolor,
    )
FIGURE_RETAINED_FACTORS_A0 = "retained_factors_A0.png"


def retained_factors_figure_name(role: str) -> str:
    """Manuscript filename for one role UMAP map."""
    return f"retained_factors_{role}.png"


def membership_strength_figure_name(role: str) -> str:
    """Appendix filename for one role membership-strength boxplot."""
    return f"membership_strength_{role}.png"

PARETO_OBJECTIVE_DECIMALS = 4

PARETO_TABLE_COLUMNS = [
    "Role",
    "Candidate ID",
    "kU",
    "dU",
    "m",
    "k",
    "SR",
    "DBCV",
    "K",
    "Noise",
]

KNEE_TABLE_COLUMNS = [
    "Role",
    "Candidate",
    "SR",
    "DBCV",
    "S_norm",
    "D_norm",
    "dK",
    "Selected",
]

RETAINED_FACTOR_COLUMNS = [
    "Role",
    "Factor",
    "Units",
    "Accidents",
    "S_cg",
    "Median membership strength",
    "Illustrative content",
]

_MISSING_TEXT = {"", "nan", "none", "null", "<na>"}


def sanitize_label_text(value) -> str:
    """Treat NaN/None/empty and literal ``nan`` strings as missing."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in _MISSING_TEXT:
        return ""
    return text


def coalesce_text(*values, default: str = "") -> str:
    """Return the first non-missing text value."""
    for value in values:
        text = sanitize_label_text(value)
        if text:
            return text
    return default


def coalesce_factor_label(
    label_row: pd.Series | None,
    *,
    topic_id: str,
) -> str:
    """Display label for manuscript tables: LLM only, else topic id."""
    if label_row is not None and isinstance(label_row, pd.Series):
        label = coalesce_text(label_row.get("llm_label"), default="")
        if label:
            return label
    return topic_id


def coalesce_illustrative_content(label_row: pd.Series | None) -> str:
    """Illustrative text: LLM evidence, else representative sentences (never top terms)."""
    if label_row is None or not isinstance(label_row, pd.Series):
        return ""
    evidence = coalesce_text(
        label_row.get("llm_evidence"),
        label_row.get("representative_sentences"),
        label_row.get("llm_description"),
    )
    if " || " in evidence:
        evidence = ", ".join(part.strip() for part in evidence.split(" || ")[:4])
    return evidence


def build_corpus_summary_table(units: pd.DataFrame) -> pd.DataFrame:
    """Role-level factual units, contributing accidents and units per accident."""
    rows = []
    for role in ROLES:
        subset = units.loc[units["_role"].astype(str).eq(role)]
        n_units = int(len(subset))
        n_accidents = int(subset["_accident_id"].nunique()) if n_units else 0
        units_per_accident = float(n_units / n_accidents) if n_accidents else np.nan
        rows.append({
            "Role": role,
            "Factual units": n_units,
            "Contributing accidents": n_accidents,
            "Units per accident": round(units_per_accident, 2) if np.isfinite(units_per_accident) else np.nan,
        })
    return pd.DataFrame(rows)


def build_corpus_summary_from_audit(summary: pd.DataFrame) -> pd.DataFrame:
    """Build corpus summary from ``audit_input_summary.csv`` long format."""
    lookup = summary.set_index("metric")["value"].to_dict() if "metric" in summary.columns else {}
    rows = []
    for role in ROLES:
        n_units = int(float(lookup.get(f"n_units_{role}", 0)))
        n_accidents = int(float(lookup.get(f"n_accidents_{role}", 0)))
        units_per_accident = round(n_units / n_accidents, 2) if n_accidents else np.nan
        rows.append({
            "Role": role,
            "Factual units": n_units,
            "Contributing accidents": n_accidents,
            "Units per accident": units_per_accident,
        })
    return pd.DataFrame(rows)


def _pareto_candidate_label(role: str, rank: int) -> str:
    return f"{role}-P{int(rank)}"


def _configuration_hyperparameter_row(row: pd.Series) -> dict:
    return {
        "kU": row.get("umap_n_neighbors"),
        "dU": row.get("umap_n_components"),
        "m": row.get("hdbscan_min_cluster_size"),
        "k": row.get("hdbscan_min_samples"),
        "SR": row.get("stability"),
        "DBCV": row.get("dbcv_umap"),
        "K": row.get("n_clusters"),
        "Noise": row.get("noise_fraction"),
    }


def build_pareto_front_summary_table(selection_table: pd.DataFrame, *, role: str) -> pd.DataFrame:
    """Pareto-optimal configurations sorted by increasing DBCV."""
    frame = selection_table.loc[selection_table["role"].astype(str).eq(role)].copy()
    if frame.empty:
        return pd.DataFrame(columns=PARETO_TABLE_COLUMNS)
    if "is_pareto" in frame.columns:
        frame = frame.loc[frame["is_pareto"].fillna(False).astype(bool)]
    elif "on_pareto" in frame.columns:
        frame = frame.loc[frame["on_pareto"].fillna(False).astype(bool)]
    frame = frame.sort_values(["dbcv_umap", "stability"], ascending=[True, False], na_position="last")
    rows = []
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        hyper = _configuration_hyperparameter_row(row)
        rows.append({
            "Role": role,
            "Candidate ID": _pareto_candidate_label(role, rank),
            **hyper,
            "_configuration_id": str(row.get("configuration_id", "")),
            "_candidate_rank": rank,
        })
    return pd.DataFrame(rows)


def build_knee_selection_table(selection_table: pd.DataFrame, *, role: str) -> pd.DataFrame:
    """Normalized Pareto objectives and knee distance for every Pareto point."""
    pareto = build_pareto_front_summary_table(selection_table, role=role)
    if pareto.empty:
        return pd.DataFrame(columns=KNEE_TABLE_COLUMNS)
    frame = selection_table.loc[selection_table["role"].astype(str).eq(role)].copy()
    if "is_pareto" in frame.columns:
        frame = frame.loc[frame["is_pareto"].fillna(False).astype(bool)]
    lookup = frame.set_index("configuration_id")
    rows = []
    for _, row in pareto.iterrows():
        configuration_id = str(row["_configuration_id"])
        source = lookup.loc[configuration_id] if configuration_id in lookup.index else row
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        selected = bool(source.get("is_selected_knee", source.get("selected", False)))
        rows.append({
            "Role": role,
            "Candidate": row["Candidate ID"],
            "SR": round(float(source.get("stability", row["SR"])), PARETO_OBJECTIVE_DECIMALS) if pd.notna(source.get("stability", row["SR"])) else np.nan,
            "DBCV": round(float(source.get("dbcv_umap", row["DBCV"])), PARETO_OBJECTIVE_DECIMALS) if pd.notna(source.get("dbcv_umap", row["DBCV"])) else np.nan,
            "S_norm": round(float(source.get("stability_normalized")), 2) if pd.notna(source.get("stability_normalized")) else np.nan,
            "D_norm": round(float(source.get("dbcv_normalized")), 2) if pd.notna(source.get("dbcv_normalized")) else np.nan,
            "dK": round(float(source.get("knee_distance")), 3) if pd.notna(source.get("knee_distance")) else np.nan,
            "Selected": "Yes" if selected else "No",
        })
    return pd.DataFrame(rows)


def build_selected_configuration_table(selection_table: pd.DataFrame, *, role: str) -> pd.DataFrame:
    """One-row summary for the knee-selected configuration."""
    frame = selection_table.loc[selection_table["role"].astype(str).eq(role)].copy()
    if frame.empty:
        return pd.DataFrame()
    if "is_selected_knee" in frame.columns:
        selected = frame.loc[frame["is_selected_knee"].fillna(False).astype(bool)]
    else:
        selected = frame.loc[frame.get("selected", False).fillna(False).astype(bool)] if "selected" in frame.columns else pd.DataFrame()
    if selected.empty:
        return pd.DataFrame()
    row = selected.iloc[0]
    pareto = build_pareto_front_summary_table(selection_table, role=role)
    candidate_id = ""
    if not pareto.empty and str(row["configuration_id"]) in set(pareto["_configuration_id"].astype(str)):
        match = pareto.loc[pareto["_configuration_id"].astype(str).eq(str(row["configuration_id"]))]
        if not match.empty:
            candidate_id = str(match.iloc[0]["Candidate ID"])
    hyper = _configuration_hyperparameter_row(row)
    return pd.DataFrame([{
        "Role": role,
        "Candidate ID": candidate_id or str(row["configuration_id"]),
        **hyper,
        "configuration_id": str(row["configuration_id"]),
        "selection_rule": row.get("selection_rule", ""),
    }])


def appendix_figure_path(run_dir: Path, filename: str) -> Path:
    """Path under ``figs_ch4/appendix/`` for supplementary figures."""
    path = Path(run_dir) / "figs_ch4" / "appendix" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def combine_figure_images(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    ncol: int = 3,
    dpi: float = 250.0,
) -> None:
    """Stitch saved PNG panels into one manuscript figure."""
    import matplotlib.pyplot as plt

    paths = [Path(path) for path in image_paths if Path(path).is_file()]
    if not paths:
        return
    nrow = int(np.ceil(len(paths) / ncol))
    figure, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 5.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for axis, path in zip(axes, paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    for axis in axes[len(paths):]:
        axis.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.2)
    save_manuscript_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


def _resampling_panel_data(
    theme_stability: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
) -> tuple[list, list, np.ndarray, pd.DataFrame] | None:
    frame = theme_stability[
        theme_stability["configuration_id"].astype(str).eq(str(configuration_id))
        & theme_stability["n_reference_units"].astype(int).gt(0)
    ].copy()
    if frame.empty:
        return None
    theme_summary = frame.drop_duplicates("cluster_label").set_index("cluster_label")
    order = (
        theme_summary["theme_stability"]
        .astype(float)
        .sort_values(ascending=True)
        .index.tolist()
    )
    data = [
        frame.loc[frame["cluster_label"].eq(label), "best_jaccard"].astype(float).to_numpy()
        for label in order
    ]
    positions = np.arange(1, len(order) + 1)
    return data, order, positions, theme_summary


def _draw_factor_resampling_panel(
    axis,
    theme_stability: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    show_legend: bool = False,
) -> bool:
    """Draw one resampling boxplot panel; return False when no data."""
    panel = _resampling_panel_data(theme_stability, role=role, configuration_id=configuration_id)
    if panel is None:
        axis.text(0.5, 0.5, "No resampling data", ha="center", va="center")
        axis.axis("off")
        return False
    data, order, positions, theme_summary = panel
    boxplot_style = role_boxplot_kwargs(role)
    axis.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.55,
        showfliers=True,
        patch_artist=True,
        **boxplot_style,
    )
    for position, label in zip(positions, order):
        if label not in theme_summary.index:
            continue
        row = theme_summary.loc[label]
        s_cg = float(row.get("theme_stability", np.nan))
        if np.isfinite(s_cg):
            axis.scatter(
                [s_cg],
                [position],
                color="#D62728",
                s=36,
                zorder=4,
                label=r"Mean $S_{cg}$" if show_legend and position == positions[0] else "",
            )
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{role}_{int(label):03d}" for label in order])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Best-match Jaccard across accident resamples")
    axis.grid(axis="x", alpha=0.2)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles[:1], labels[:1], loc="lower right", frameon=False, fontsize=8)
    return True


def plot_membership_strength_by_factor(
    partition_frame: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    output_path: Path | None = None,
    show_unit_annotations: bool = False,
) -> object:
    """Horizontal boxplots of membership strength per retained factor."""
    import matplotlib.pyplot as plt

    del configuration_id
    valid = partition_frame.loc[partition_frame["Topic"].astype(int).ge(0)].copy()
    if valid.empty:
        return None
    valid["Topic"] = valid["Topic"].astype(int)
    summaries = []
    for topic, subset in valid.groupby("Topic"):
        strengths = subset["membership_strength"].astype(float)
        summaries.append({
            "topic": int(topic),
            "median_strength": float(strengths.median()),
            "n_units": int(len(subset)),
            "n_accidents": int(subset["accident_id"].nunique()),
            "values": strengths.to_numpy(),
        })
    summaries.sort(key=lambda item: item["median_strength"])
    figure, axis = plt.subplots(figsize=(10, max(4.0, 0.45 * len(summaries))))
    positions = np.arange(1, len(summaries) + 1)
    axis.boxplot(
        [item["values"] for item in summaries],
        vert=False,
        positions=positions,
        widths=0.55,
        showfliers=True,
        patch_artist=True,
        **role_boxplot_kwargs(role),
    )
    if show_unit_annotations:
        for item, position in zip(summaries, positions):
            axis.text(
                1.02,
                position,
                f"n={item['n_units']}, accidents={item['n_accidents']}",
                va="center",
                fontsize=8,
                transform=axis.get_yaxis_transform(),
            )
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{role}_{item['topic']:03d}" for item in summaries])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("HDBSCAN membership strength")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    if output_path is not None:
        save_manuscript_figure(figure, output_path)
    return figure


def plot_factor_resampling_reproducibility(
    theme_stability: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    output_path: Path | None = None,
) -> object:
    """Factor-level best-match Jaccard distributions with mean $S_{cg}$."""
    import matplotlib.pyplot as plt

    panel = _resampling_panel_data(theme_stability, role=role, configuration_id=configuration_id)
    if panel is None:
        return None
    _, order, _, _ = panel
    figure, axis = plt.subplots(figsize=(10, max(4.0, 0.45 * len(order))))
    _draw_factor_resampling_panel(
        axis,
        theme_stability,
        role=role,
        configuration_id=configuration_id,
        show_legend=True,
    )
    figure.tight_layout()
    if output_path is not None:
        save_manuscript_figure(figure, output_path)
    return figure


def plot_factor_resampling_multi_panel(
    theme_stability_by_role: Mapping[str, pd.DataFrame],
    *,
    roles: Sequence[str],
    configuration_ids: Mapping[str, str],
    output_path: Path | None = None,
) -> object:
    """Three-panel resampling figure for A1, B and C."""
    import matplotlib.pyplot as plt

    plot_roles = [role for role in roles if role in theme_stability_by_role]
    if not plot_roles:
        return None
    max_factors = 1
    for role in plot_roles:
        panel = _resampling_panel_data(
            theme_stability_by_role[role],
            role=role,
            configuration_id=str(configuration_ids[role]),
        )
        if panel is not None:
            max_factors = max(max_factors, len(panel[1]))
    figure_height = max(4.5, 0.42 * max_factors)
    figure, axes = plt.subplots(1, len(plot_roles), figsize=(4.8 * len(plot_roles), figure_height))
    axes = np.atleast_1d(axes).ravel()
    for axis, role in zip(axes, plot_roles):
        _draw_factor_resampling_panel(
            axis,
            theme_stability_by_role[role],
            role=role,
            configuration_id=str(configuration_ids[role]),
            show_legend=role == plot_roles[0],
        )
    figure.tight_layout()
    if output_path is not None:
        save_manuscript_figure(figure, output_path)
    return figure


def build_seed_sensitivity_summary(
    seed_summary: pd.DataFrame,
    *,
    reference_row: Mapping[str, Any] | pd.Series | None = None,
) -> pd.DataFrame:
    r"""Complementary seed table: reference \(K\) plus ranges over alternative seeds.

    The primary seed \(s_0\) is the reference partition; alternative seeds are never
    selected. Ranges summarise diagnostic variation only.
    """
    if seed_summary.empty:
        return pd.DataFrame()
    if isinstance(reference_row, pd.Series):
        reference_row = reference_row.to_dict()
    reference_row = dict(reference_row or {})
    rows = []
    for role, subset in seed_summary.groupby("role"):
        jaccard = subset["seed_stability"].astype(float)
        k_values = subset["n_clusters"].astype(float)
        dbcv = subset["dbcv_umap"].astype(float)
        unassigned = subset["noise_fraction"].astype(float)
        ref_k = reference_row.get("n_clusters", reference_row.get("K"))
        rows.append({
            "Role": role,
            "configuration_id": str(subset["configuration_id"].iloc[0]),
            "Reference_K": int(ref_k) if ref_k is not None and pd.notna(ref_k) else pd.NA,
            "K_range": f"{int(k_values.min())}-{int(k_values.max())}",
            "DBCV_range": f"{float(dbcv.min()):.3f}-{float(dbcv.max()):.3f}",
            "Unassigned_fraction_range": f"{float(unassigned.min()):.3f}-{float(unassigned.max()):.3f}",
            "Mean_Jaccard_range": f"{float(jaccard.min()):.3f}-{float(jaccard.max()):.3f}",
            "Mean_Jaccard_min": float(jaccard.min()),
            "Mean_Jaccard_max": float(jaccard.max()),
            "n_alternative_seeds": int(len(subset)),
        })
    return pd.DataFrame(rows)


def build_seed_sensitivity_summary_all_roles(
    run_dir: Path,
    *,
    roles: Sequence[str] = ROLES,
) -> pd.DataFrame:
    """Build the manuscript seed-sensitivity table for all roles under ``run_dir``."""
    run_dir = Path(run_dir)
    selected_path = run_dir / "selected_configurations.csv"
    selected = pd.read_csv(selected_path) if selected_path.is_file() else pd.DataFrame()
    tables = []
    for role in roles:
        summary_path = run_dir / "discovery" / role / "seed_sensitivity" / "seed_summary.csv"
        if not summary_path.is_file():
            continue
        seed_summary = pd.read_csv(summary_path)
        if seed_summary.empty:
            continue
        reference = None
        if not selected.empty and "role" in selected.columns:
            match = selected.loc[selected["role"].astype(str).eq(role)]
            if not match.empty:
                reference = match.iloc[0]
        if reference is None:
            selection_path = run_dir / "discovery" / role / "selection_table.csv"
            if selection_path.is_file():
                selection = pd.read_csv(selection_path)
                if not selection.empty:
                    knee = selection.loc[
                        selection.get("is_selected_knee", pd.Series(False, index=selection.index))
                        .fillna(False)
                        .astype(bool)
                    ]
                    if knee.empty and "configuration_id" in seed_summary.columns:
                        cfg = str(seed_summary["configuration_id"].iloc[0])
                        knee = selection.loc[selection["configuration_id"].astype(str).eq(cfg)]
                    if not knee.empty:
                        reference = knee.iloc[0]
        role_table = build_seed_sensitivity_summary(seed_summary, reference_row=reference)
        if not role_table.empty:
            tables.append(role_table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_seed_sensitivity_factors(
    seed_factor_frame: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    output_path: Path | None = None,
) -> object:
    """Factor-level Jaccard distributions across alternative UMAP seeds."""
    import matplotlib.pyplot as plt

    del configuration_id
    frame = seed_factor_frame[seed_factor_frame["role"].astype(str).eq(role)].copy()
    if frame.empty:
        return None
    order = (
        frame.groupby("cluster_label")["best_jaccard"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    data = [frame.loc[frame["cluster_label"].eq(label), "best_jaccard"].astype(float).to_numpy() for label in order]
    figure, axis = plt.subplots(figsize=(10, max(4.0, 0.45 * len(order))))
    positions = np.arange(1, len(order) + 1)
    axis.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.55,
        showfliers=True,
        patch_artist=True,
        **role_boxplot_kwargs(role),
    )
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{role}_{int(label):03d}" for label in order])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Best-match Jaccard across UMAP seeds")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    if output_path is not None:
        save_manuscript_figure(figure, output_path)
    return figure


def plot_umap_seed_sensitivity_all_roles(
    run_dir: Path,
    *,
    roles: Sequence[str] = ROLES,
    output_path: Path | None = None,
) -> object:
    r"""Four-panel mean best-match Jaccard vs alternative UMAP seeds.

    Primary message only: membership stability relative to the \(s_0\) reference
    partition. \(K\), DBCV and unassigned fraction belong in the complementary table,
    not on a dual axis.
    """
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), squeeze=False, sharey=True)
    drew_any = False
    jaccard_color = ROLE_COLORS["A0"]
    for index, (axis, role) in enumerate(zip(axes.flat, roles)):
        summary_path = run_dir / "discovery" / role / "seed_sensitivity" / "seed_summary.csv"
        axis.set_title(role_panel_title(role, index=index), fontsize=11, pad=6)
        if not summary_path.is_file():
            axis.text(0.5, 0.5, "No seed data", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlim(0.5, 10.5)
            axis.set_ylim(0, 1.05)
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty:
            axis.text(0.5, 0.5, "No seed data", ha="center", va="center", transform=axis.transAxes)
            continue
        drew_any = True
        seeds = summary["seed"].astype(int)
        jaccard = summary["seed_stability"].astype(float)
        axis.plot(
            seeds,
            jaccard,
            marker="o",
            color=ROLE_COLORS.get(role, jaccard_color),
            linewidth=1.4,
            markersize=5.5,
        )
        axis.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle="--", alpha=0.7)
        axis.set_ylim(0, 1.05)
        axis.set_xlim(float(seeds.min()) - 0.5, float(seeds.max()) + 0.5)
        axis.set_xticks(sorted(seeds.unique().tolist()))
        axis.set_xlabel("Alternative UMAP seed")
        if index % 2 == 0:
            axis.set_ylabel("Mean best-match Jaccard vs $s_0$")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(roles):]:
        axis.remove()
    if not drew_any:
        plt.close(figure)
        return None
    figure.suptitle(
        "UMAP seed sensitivity: membership stability relative to the reference seed $s_0$",
        fontsize=12,
        y=1.01,
    )
    figure.tight_layout()
    if output_path is not None:
        save_manuscript_figure(figure, output_path)
        pdf_path = Path(output_path).with_suffix(".pdf")
        save_manuscript_figure(figure, pdf_path)
    return figure


def build_retained_factors_summary_table(
    theme_labels: pd.DataFrame,
    partition_frames: Mapping[tuple[str, str], pd.DataFrame],
    factor_stability: pd.DataFrame | None = None,
    *,
    partition_selection: Mapping[str, str],
) -> pd.DataFrame:
    """Summary of retained factors after LLM labelling."""
    rows = []
    for role in ROLES:
        configuration_id = str(partition_selection[role])
        frame = partition_frames.get((role, configuration_id))
        if frame is None:
            continue
        valid = frame.loc[frame["Topic"].astype(int).ge(0)].copy()
        role_labels = theme_labels[
            theme_labels["role"].astype(str).eq(role)
            & theme_labels["configuration_id"].astype(str).eq(configuration_id)
        ]
        label_lookup = role_labels.set_index("topic_id") if not role_labels.empty else pd.DataFrame()
        stability_lookup = {}
        if factor_stability is not None and not factor_stability.empty:
            subset = factor_stability[
                factor_stability["configuration_id"].astype(str).eq(configuration_id)
            ].drop_duplicates("cluster_label")
            stability_lookup = {
                int(row["cluster_label"]): float(row.get("theme_stability", np.nan))
                for _, row in subset.iterrows()
            }
        for topic in sorted(int(value) for value in valid["Topic"].unique()):
            subset = valid.loc[valid["Topic"].eq(topic)]
            topic_id = f"{role}_{topic:03d}"
            label_row = label_lookup.loc[topic_id] if topic_id in label_lookup.index else None
            factor_name = coalesce_factor_label(label_row, topic_id=topic_id)
            evidence = coalesce_illustrative_content(label_row)
            rows.append({
                "Role": role,
                "Factor": factor_name,
                "Units": int(len(subset)),
                "Accidents": int(subset["accident_id"].nunique()),
                "S_cg": round(stability_lookup.get(topic, np.nan), 2) if topic in stability_lookup else np.nan,
                "Median membership strength": round(float(subset["membership_strength"].median()), 2),
                "Illustrative content": evidence,
            })
    return pd.DataFrame(rows)


def format_manuscript_table(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    column_decimals: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Return a display-ready copy with manuscript column order."""
    if frame.empty:
        return frame.copy()
    ordered = [column for column in (columns or frame.columns) if column in frame.columns]
    display_frame = frame.loc[:, ordered].copy()
    decimals = {"SR": PARETO_OBJECTIVE_DECIMALS, "DBCV": PARETO_OBJECTIVE_DECIMALS}
    if column_decimals:
        decimals.update(column_decimals)

    def _format_cell(value, *, decimals: int = 3):
        if pd.isna(value):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            return int(numeric) if numeric.is_integer() and decimals == 0 else round(numeric, decimals)
        return value

    for column in display_frame.columns:
        if column in {"Role", "Factor", "Illustrative content", "Candidate", "Candidate ID"}:
            continue
        if display_frame[column].dtype.kind in "fi":
            column_precision = decimals.get(column, 3)
            display_frame[column] = display_frame[column].map(
                lambda value, precision=column_precision: _format_cell(value, decimals=precision)
            )
    return display_frame


def display_manuscript_table(frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Format and display a manuscript table (returns the formatted frame)."""
    from IPython.display import display

    formatted = format_manuscript_table(frame, columns)
    display(formatted)
    return formatted
