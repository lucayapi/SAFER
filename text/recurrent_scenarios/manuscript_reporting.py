"""Manuscript-oriented tables and figures for topic-discovery results notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROLES = ("A0", "A1", "B", "C")

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
    "S_ck",
    "Median membership strength",
    "Illustrative content",
]


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
            "SR": round(float(source.get("stability", row["SR"])), 2) if pd.notna(source.get("stability", row["SR"])) else np.nan,
            "DBCV": round(float(source.get("dbcv_umap", row["DBCV"])), 2) if pd.notna(source.get("dbcv_umap", row["DBCV"])) else np.nan,
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


def plot_membership_strength_by_factor(
    partition_frame: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    output_path: Path | None = None,
) -> object:
    """Horizontal boxplots of membership strength per retained factor."""
    import matplotlib.pyplot as plt

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
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D62728", "linewidth": 1.4},
    )
    ylabels = []
    for item, position in zip(summaries, positions):
        ylabels.append(f"{role}_{item['topic']:03d}")
        axis.text(
            1.02,
            position,
            f"n={item['n_units']}, accidents={item['n_accidents']}",
            va="center",
            fontsize=8,
            transform=axis.get_yaxis_transform(),
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(ylabels)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("HDBSCAN membership strength")
    axis.set_title(f"{role} — membership-strength distributions ({configuration_id})")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
    return figure


def plot_factor_resampling_reproducibility(
    theme_stability: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    output_path: Path | None = None,
) -> object:
    """Factor-level best-match Jaccard distributions with S_ck and B_ck/B."""
    import matplotlib.pyplot as plt

    frame = theme_stability[
        theme_stability["configuration_id"].astype(str).eq(str(configuration_id))
        & theme_stability["n_reference_units"].astype(int).gt(0)
    ].copy()
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
        boxprops={"facecolor": "#FEE0D2", "edgecolor": "#DE2D26"},
        medianprops={"color": "#A50F15", "linewidth": 1.4},
    )
    theme_summary = frame.drop_duplicates("cluster_label").set_index("cluster_label")
    for position, label in zip(positions, order):
        if label not in theme_summary.index:
            continue
        row = theme_summary.loc[label]
        s_ck = float(row.get("theme_stability", np.nan))
        observability = float(row.get("observability", np.nan))
        axis.scatter([s_ck], [position], color="#D62728", s=36, zorder=4, label="Mean $S_{ck}$" if position == positions[0] else "")
        axis.text(
            1.02,
            position,
            rf"$S_{{ck}}$={s_ck:.2f}, $B_{{ck}}/B$={observability:.2f}",
            va="center",
            fontsize=8,
            transform=axis.get_yaxis_transform(),
        )
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{role}_{int(label):03d}" for label in order])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Best-match Jaccard across accident resamples")
    axis.set_title(f"{role} — factor-level resampling reproducibility ({configuration_id})")
    axis.grid(axis="x", alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles[:1], labels[:1], loc="lower right", frameon=False)
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
    return figure


def build_seed_sensitivity_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DBCV, K and noise ranges across alternative UMAP seeds."""
    if seed_summary.empty:
        return pd.DataFrame()
    rows = []
    for role, subset in seed_summary.groupby("role"):
        rows.append({
            "Role": role,
            "configuration_id": str(subset["configuration_id"].iloc[0]),
            "DBCV_min": float(subset["dbcv_umap"].min()),
            "DBCV_max": float(subset["dbcv_umap"].max()),
            "K_min": int(subset["n_clusters"].min()),
            "K_max": int(subset["n_clusters"].max()),
            "Noise_min": float(subset["noise_fraction"].min()),
            "Noise_max": float(subset["noise_fraction"].max()),
            "n_seeds": int(len(subset)),
        })
    return pd.DataFrame(rows)


def plot_seed_sensitivity_factors(
    seed_factor_frame: pd.DataFrame,
    *,
    role: str,
    configuration_id: str,
    output_path: Path | None = None,
) -> object:
    """Factor-level Jaccard distributions across alternative UMAP seeds."""
    import matplotlib.pyplot as plt

    frame = seed_factor_frame[
        seed_factor_frame["role"].astype(str).eq(role)
        & seed_factor_frame["configuration_id"].astype(str).eq(str(configuration_id))
    ].copy()
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
        boxprops={"facecolor": "#E5F5E0", "edgecolor": "#31A354"},
        medianprops={"color": "#006D2C", "linewidth": 1.4},
    )
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{role}_{int(label):03d}" for label in order])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Best-match Jaccard across UMAP seeds")
    axis.set_title(f"{role} — UMAP seed sensitivity ({configuration_id})")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
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
            if label_row is not None and isinstance(label_row, pd.Series):
                factor_name = str(
                    label_row.get("plot_label")
                    or label_row.get("llm_label")
                    or label_row.get("top_terms")
                    or topic_id
                ).strip()
                evidence = str(label_row.get("llm_evidence") or "").strip()
                if not evidence:
                    evidence = str(label_row.get("representative_sentences") or "").strip()
                if " || " in evidence:
                    evidence = ", ".join(part.strip() for part in evidence.split(" || ")[:4])
            else:
                factor_name = topic_id
                evidence = ""
            rows.append({
                "Role": role,
                "Factor": factor_name,
                "Units": int(len(subset)),
                "Accidents": int(subset["accident_id"].nunique()),
                "S_ck": round(stability_lookup.get(topic, np.nan), 2) if topic in stability_lookup else np.nan,
                "Median membership strength": round(float(subset["membership_strength"].median()), 2),
                "Illustrative content": evidence,
            })
    return pd.DataFrame(rows)
