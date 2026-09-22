"""Deterministic Pareto screening and normalized reference-point selection.

For each role, configurations are compared on two maximized objectives:
accident-level reproducibility ``S_R`` (``stability``) and UMAP-space ``DBCV``
(``dbcv_umap``). Objectives are normalized on the role-specific Pareto set.
The selected compromise minimizes the largest normalized shortfall from the
empirical ideal point ``(1, 1)``. Exact numerical ties are resolved by the sum
of normalized shortfalls, then by the predefined configuration-grid order.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SELECTION_TOLERANCE = 1e-12
# Historical public name retained for imports outside this repository.
KNEE_TOLERANCE = SELECTION_TOLERANCE
NORM_TOLERANCE = 1e-9
STABILITY_COL = "stability"
DBCV_COL = "dbcv_umap"
SELECTED_CONFIGURATION_LEGEND = "Selected configuration"


def identify_pareto_front(
    df_role: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
) -> pd.DataFrame:
    """Mark Pareto-optimal configurations (both objectives maximized).

    Configuration ``c'`` dominates ``c`` when
    ``DBCV(c') >= DBCV(c)`` and ``S_R(c') >= S_R(c)`` with at least one strict
    inequality. The Pareto set contains all non-dominated configurations.
    """
    result = df_role.copy()
    result["is_pareto"] = False
    result["on_pareto"] = False
    result["selected"] = False
    result["is_selected_tchebycheff"] = False
    if result.empty or stability_col not in result.columns or dbcv_col not in result.columns:
        return result
    usable = result[stability_col].notna() & result[dbcv_col].notna()
    if not usable.any():
        return result
    pool = result.loc[usable]
    indices = pool.index.to_list()
    stabilities = pool[stability_col].astype(float).to_numpy()
    dbcvs = pool[dbcv_col].astype(float).to_numpy()
    on_pareto = np.ones(len(indices), dtype=bool)
    for i in range(len(indices)):
        dominated = (
            (stabilities >= stabilities[i])
            & (dbcvs >= dbcvs[i])
            & ((stabilities > stabilities[i]) | (dbcvs > dbcvs[i]))
        )
        other_dominates = np.any(dominated & (np.arange(len(indices)) != i))
        if other_dominates:
            on_pareto[i] = False
    pareto_index = np.asarray(indices)[on_pareto]
    result.loc[pareto_index, "is_pareto"] = True
    result.loc[pareto_index, "on_pareto"] = True
    return result


def mark_pareto_front(
    frame: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`identify_pareto_front`."""
    return identify_pareto_front(frame, stability_col=stability_col, dbcv_col=dbcv_col)


def normalize_pareto_objectives(
    df_pareto: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    role: str | None = None,
) -> pd.DataFrame:
    """Min–max normalize ``S_R`` and ``DBCV`` within the role Pareto set."""
    result = df_pareto.copy()
    result["stability_normalized"] = np.nan
    result["dbcv_normalized"] = np.nan
    if result.empty:
        return result
    stabilities = result[stability_col].astype(float)
    dbcvs = result[dbcv_col].astype(float)
    s_min, s_max = float(stabilities.min()), float(stabilities.max())
    d_min, d_max = float(dbcvs.min()), float(dbcvs.max())
    role_label = role or str(result.get("role", pd.Series(["?"])).iloc[0])
    if np.isclose(d_max, d_min, rtol=0.0, atol=SELECTION_TOLERANCE):
        warnings.warn(
            f"[{role_label}] DBCV is constant on the Pareto front; "
            "all Pareto configurations receive normalized DBCV = 1.",
            stacklevel=2,
        )
        result["dbcv_normalized"] = 1.0
    else:
        result["dbcv_normalized"] = (dbcvs - d_min) / (d_max - d_min)
    if np.isclose(s_max, s_min, rtol=0.0, atol=SELECTION_TOLERANCE):
        warnings.warn(
            f"[{role_label}] S_R is constant on the Pareto front; "
            "all Pareto configurations receive normalized S_R = 1.",
            stacklevel=2,
        )
        result["stability_normalized"] = 1.0
    else:
        result["stability_normalized"] = (stabilities - s_min) / (s_max - s_min)
    return result


def compute_tchebycheff_scores(
    df_pareto: pd.DataFrame,
    *,
    dbcv_norm_col: str = "dbcv_normalized",
    stability_norm_col: str = "stability_normalized",
) -> pd.DataFrame:
    """Compute primary and secondary shortfalls from the ideal point ``(1, 1)``."""
    result = df_pareto.copy()
    d_norm = result[dbcv_norm_col].astype(float)
    s_norm = result[stability_norm_col].astype(float)
    delta_d = 1.0 - d_norm
    delta_s = 1.0 - s_norm
    result["dbcv_normalized_shortfall"] = delta_d
    result["stability_normalized_shortfall"] = delta_s
    result["tchebycheff_max_shortfall"] = np.maximum(delta_d, delta_s)
    result["total_normalized_shortfall"] = delta_d + delta_s
    return result


def _configuration_grid_order(configuration_id: object) -> tuple[int, str]:
    """Return the encoded grid position, with a lexical deterministic fallback."""
    text = str(configuration_id)
    suffix = text.rsplit("_cfg_", 1)
    if len(suffix) == 2 and suffix[1].isdigit():
        return int(suffix[1]), text
    return np.iinfo(np.int64).max, text


def _select_tchebycheff_from_pareto(
    pareto: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    configuration_col: str = "configuration_id",
) -> tuple[str, pd.DataFrame, str]:
    """Select by ``min T_inf``, then ``min T_1``, then grid order."""
    normalized = normalize_pareto_objectives(
        pareto,
        stability_col=stability_col,
        dbcv_col=dbcv_col,
        role=str(pareto["role"].iloc[0]) if "role" in pareto.columns and not pareto.empty else None,
    )
    scored = compute_tchebycheff_scores(normalized)
    min_t_inf = float(scored["tchebycheff_max_shortfall"].min())
    primary = scored.loc[
        np.isclose(
            scored["tchebycheff_max_shortfall"].astype(float),
            min_t_inf,
            rtol=0.0,
            atol=SELECTION_TOLERANCE,
        )
    ]
    scored["is_tchebycheff_minimizer"] = scored.index.isin(primary.index)
    if len(primary) == 1:
        selected = primary.iloc[0]
        tie_break = "not_required"
        secondary = primary
    else:
        min_t_one = float(primary["total_normalized_shortfall"].min())
        secondary = primary.loc[
            np.isclose(
                primary["total_normalized_shortfall"].astype(float),
                min_t_one,
                rtol=0.0,
                atol=SELECTION_TOLERANCE,
            )
        ]
        if len(secondary) == 1:
            selected = secondary.iloc[0]
            tie_break = "total_normalized_shortfall"
        else:
            selected = sorted(
                (row for _, row in secondary.iterrows()),
                key=lambda row: _configuration_grid_order(row[configuration_col]),
            )[0]
            tie_break = "grid_order"
    scored["is_total_shortfall_minimizer"] = scored.index.isin(secondary.index)
    return str(selected[configuration_col]), scored, tie_break


def select_tchebycheff_configuration(
    df_role: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
) -> tuple[pd.DataFrame, str, str]:
    """Apply Pareto screening and normalized Tchebycheff reference selection."""
    marked = identify_pareto_front(df_role, stability_col=stability_col, dbcv_col=dbcv_col)
    score_columns = (
        "stability_normalized",
        "dbcv_normalized",
        "stability_normalized_shortfall",
        "dbcv_normalized_shortfall",
        "tchebycheff_max_shortfall",
        "total_normalized_shortfall",
    )
    for column in score_columns:
        marked[column] = np.nan
    marked["is_tchebycheff_minimizer"] = False
    marked["is_total_shortfall_minimizer"] = False
    marked["selection_tie_break"] = ""
    pareto = marked.loc[marked["is_pareto"]].copy()
    if pareto.empty:
        return marked, "", "none"
    if len(pareto) == 1:
        selected_id = str(pareto.iloc[0]["configuration_id"])
        selected_mask = marked["configuration_id"].astype(str).eq(selected_id)
        marked.loc[selected_mask, "is_selected_tchebycheff"] = True
        marked.loc[selected_mask, "selected"] = True
        marked.loc[selected_mask, "selection_tie_break"] = "not_required"
        return marked, selected_id, "single_pareto"

    selected_id, scored, tie_break = _select_tchebycheff_from_pareto(
        pareto,
        stability_col=stability_col,
        dbcv_col=dbcv_col,
    )
    lookup_columns = [
        *score_columns,
        "is_tchebycheff_minimizer",
        "is_total_shortfall_minimizer",
    ]
    score_lookup = scored.set_index("configuration_id")[lookup_columns].to_dict("index")
    for column in lookup_columns:
        marked[column] = marked["configuration_id"].astype(str).map(
            lambda configuration_id, col=column: score_lookup.get(configuration_id, {}).get(
                col, False if col.startswith("is_") else np.nan
            )
        )
    selected_mask = marked["configuration_id"].astype(str).eq(selected_id)
    marked.loc[selected_mask, "is_selected_tchebycheff"] = True
    marked.loc[selected_mask, "selected"] = True
    marked.loc[selected_mask, "selection_tie_break"] = tie_break
    return marked, selected_id, "normalized_tchebycheff"


def select_knee_configuration(
    df_role: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
) -> tuple[pd.DataFrame, str, str]:
    """Compatibility alias for the former public function name."""
    return select_tchebycheff_configuration(
        df_role,
        stability_col=stability_col,
        dbcv_col=dbcv_col,
    )


def select_configuration_for_role(
    merged: pd.DataFrame,
    semantic_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Select via Pareto + normalized Tchebycheff (``semantic_scores`` ignored)."""
    del semantic_scores
    return select_tchebycheff_configuration(merged)


def summarize_selected_configurations(
    selection_tables: Mapping[str, pd.DataFrame],
    *,
    parameter_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """Build one summary row per role for the selected configuration."""
    rows: list[dict] = []
    for role, table in selection_tables.items():
        if table.empty:
            continue
        selected = table.loc[table["is_selected_tchebycheff"].fillna(False).astype(bool)]
        if selected.empty:
            selected = table.loc[table["selected"].fillna(False).astype(bool)]
        if selected.empty:
            continue
        row = selected.iloc[0]
        hyperparameters = {
            key: row.get(key)
            for key in parameter_keys
            if key in row.index
        }
        rows.append({
            "role": role,
            "selected_configuration_id": str(row["configuration_id"]),
            "configuration_id": str(row["configuration_id"]),
            STABILITY_COL: row.get(STABILITY_COL),
            DBCV_COL: row.get(DBCV_COL),
            "stability_normalized": row.get("stability_normalized"),
            "dbcv_normalized": row.get("dbcv_normalized"),
            "tchebycheff_max_shortfall": row.get("tchebycheff_max_shortfall"),
            "total_normalized_shortfall": row.get("total_normalized_shortfall"),
            "selection_tie_break": row.get("selection_tie_break"),
            "selection_rule": "normalized_tchebycheff" if int(table["is_pareto"].sum()) > 1 else "single_pareto",
            "n_clusters": row.get("n_clusters"),
            "noise_fraction": row.get("noise_fraction"),
            "hyperparameters": hyperparameters,
            **hyperparameters,
        })
    return pd.DataFrame(rows)


def print_role_selection_summary(
    role: str,
    table: pd.DataFrame,
    *,
    selected_id: str,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
) -> None:
    """Print the Pareto extremes and normalized reference-point compromise."""
    n_candidates = len(table)
    pareto = table.loc[table["is_pareto"].fillna(False).astype(bool)].copy()
    n_pareto = len(pareto)
    print("-" * 50)
    print(f"Role {role}")
    print(f"{n_candidates} candidate configurations")
    print(f"{n_pareto} Pareto-optimal configurations")
    if pareto.empty:
        print("No selectable Pareto configuration.")
        print("-" * 50)
        return
    max_sr = pareto.sort_values([stability_col, dbcv_col], ascending=[False, True]).iloc[0]
    max_dbcv = pareto.sort_values([dbcv_col, stability_col], ascending=[False, True]).iloc[0]
    print("\nMaximum-S_R extreme:")
    print(f"configuration = {max_sr['configuration_id']}")
    print(f"S_R = {max_sr[stability_col]}")
    print(f"DBCV = {max_sr[dbcv_col]}")
    print("\nMaximum-DBCV extreme:")
    print(f"configuration = {max_dbcv['configuration_id']}")
    print(f"S_R = {max_dbcv[stability_col]}")
    print(f"DBCV = {max_dbcv[dbcv_col]}")
    selected = table.loc[table["configuration_id"].astype(str).eq(str(selected_id))]
    if not selected.empty:
        selected_row = selected.iloc[0]
        print("\nNormalized Tchebycheff compromise:")
        print(f"configuration = {selected_id}")
        print(f"S_R = {selected_row.get(stability_col)}")
        print(f"DBCV = {selected_row.get(dbcv_col)}")
        print(f"normalized S_R = {selected_row.get('stability_normalized')}")
        print(f"normalized DBCV = {selected_row.get('dbcv_normalized')}")
        print(f"T_inf = {selected_row.get('tchebycheff_max_shortfall')}")
        print(f"T_1 = {selected_row.get('total_normalized_shortfall')}")
        print(f"tie break = {selected_row.get('selection_tie_break')}")
    print("-" * 50)


def roles_with_multi_point_pareto_front(
    selection_tables: Mapping[str, pd.DataFrame],
    roles: Sequence[str] = ("A0", "A1", "B", "C"),
) -> tuple[str, ...]:
    """Return roles whose Pareto front contains more than one configuration."""
    eligible: list[str] = []
    for role in roles:
        frame = selection_tables.get(role, pd.DataFrame())
        if frame.empty:
            continue
        pareto_mask = frame["is_pareto"].fillna(frame.get("on_pareto", False)).astype(bool)
        if int(pareto_mask.sum()) > 1:
            eligible.append(role)
    return tuple(eligible)


def _pareto_subplot_layout(n_roles: int) -> tuple[int, int, tuple[float, float]]:
    """Return ``(n_rows, n_cols, figsize)`` for a role-wise Pareto panel grid."""
    if n_roles <= 0:
        return 1, 1, (7.0, 6.0)
    if n_roles == 1:
        return 1, 1, (7.0, 6.0)
    if n_roles == 2:
        return 1, 2, (14.0, 5.5)
    return 2, 2, (14.0, 10.0)


def _selected_mask(frame: pd.DataFrame) -> pd.Series:
    """Read the current selection flag, with support for legacy result tables."""
    if "is_selected_tchebycheff" in frame.columns:
        return frame["is_selected_tchebycheff"].fillna(False).astype(bool)
    if "selected" in frame.columns:
        return frame["selected"].fillna(False).astype(bool)
    if "is_selected_knee" in frame.columns:
        return frame["is_selected_knee"].fillna(False).astype(bool)
    return pd.Series(False, index=frame.index)


def _panel_title(
    role: str,
    index: int,
    role_labels: Mapping[str, str] | None,
) -> str:
    from manuscript_reporting import ROLE_PANEL_LETTERS, role_panel_title

    if not role_labels or role not in role_labels:
        return role_panel_title(role, index=index)
    letter = ROLE_PANEL_LETTERS[index] if index < len(ROLE_PANEL_LETTERS) else chr(ord("a") + index)
    return f"({letter}) {role_labels[role]}"


def plot_pareto_raw(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    roles: Sequence[str] = ("A0", "A1", "B", "C"),
    filename: str = "stability_landscape_all_roles.png",
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    suptitle: str | None = None,
    role_labels: Mapping[str, str] | None = None,
    axis_labels: Mapping[str, str] | None = None,
    legend_labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
) -> None:
    """Raw DBCV/S_R landscape with dominated, Pareto and selected points."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from manuscript_reporting import save_manuscript_figure

    plot_roles = tuple(roles)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axis_labels = dict(axis_labels or {})
    legend_labels = dict(legend_labels or {})
    colors = dict(colors or {})
    selected_color = colors.get("selected", "#d62728")
    pareto_color = colors.get("pareto", "#1f77b4")
    other_color = colors.get("candidates", "#757575")
    x_label = axis_labels.get("dbcv_raw", "DBCV")
    y_label = axis_labels.get("stability_raw", r"$S_R$")
    for index, (axis, role) in enumerate(zip(axes.flat, plot_roles)):
        axis.set_title(_panel_title(role, index, role_labels), fontsize=11, pad=6)
        frame = selection_tables.get(role, pd.DataFrame()).copy()
        if frame.empty:
            axis.text(0.5, 0.5, "No configuration", ha="center", va="center")
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            continue
        valid = frame[dbcv_col].notna() & frame[stability_col].notna()
        base = frame.loc[valid].copy()
        is_pareto = base["is_pareto"].fillna(base.get("on_pareto", False)).astype(bool)
        is_selected = _selected_mask(base)
        others = base.loc[~is_pareto]
        pareto_all = base.loc[is_pareto].sort_values(dbcv_col)
        selected = base.loc[is_selected]
        if not others.empty:
            axis.scatter(
                others[dbcv_col],
                others[stability_col],
                s=18,
                c=other_color,
                alpha=0.78,
                linewidths=0,
                zorder=1,
            )
        if not pareto_all.empty:
            axis.plot(
                pareto_all[dbcv_col],
                pareto_all[stability_col],
                color=pareto_color,
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )
            if not selected.empty:
                selected_index = set(selected.index)
                non_selected = pareto_all.loc[~pareto_all.index.isin(selected_index)]
            else:
                non_selected = pareto_all
            if not non_selected.empty:
                axis.scatter(
                    non_selected[dbcv_col],
                    non_selected[stability_col],
                    marker="o",
                    s=70,
                    facecolors=pareto_color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=3,
                )
        if not selected.empty:
            axis.scatter(
                selected[dbcv_col],
                selected[stability_col],
                marker="*",
                s=320,
                facecolors=selected_color,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(plot_roles):]:
        axis.remove()
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label=legend_labels.get("candidates", "Candidate configurations"), markerfacecolor=other_color, markersize=6),
        Line2D([0], [0], marker="o", linestyle="-", color=pareto_color, label=legend_labels.get("pareto", "Pareto-optimal configurations"), markerfacecolor=pareto_color, markersize=7),
        Line2D([0], [0], marker="*", linestyle="None", color="black", label=legend_labels.get("selected", SELECTED_CONFIGURATION_LEGEND), markerfacecolor=selected_color, markersize=12),
    ]
    if suptitle:
        figure.suptitle(suptitle, y=0.98, fontsize=12)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97 if suptitle else 1.0))
    save_manuscript_figure(figure, output_dir / filename)
    plt.close(figure)


def plot_pareto_normalized_tchebycheff(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    roles: Sequence[str] = ("A0", "A1", "B", "C"),
    filename: str = "pareto_normalized_tchebycheff_all_roles.png",
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    suptitle: str | None = None,
    multi_pareto_only: bool = True,
    role_labels: Mapping[str, str] | None = None,
    axis_labels: Mapping[str, str] | None = None,
    legend_labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
) -> None:
    """Normalized Pareto fronts, ideal point and Tchebycheff compromise."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    candidate_roles = tuple(roles)
    plot_roles = (
        roles_with_multi_point_pareto_front(selection_tables, candidate_roles)
        if multi_pareto_only
        else candidate_roles
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols, figsize = _pareto_subplot_layout(len(plot_roles))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axis_labels = dict(axis_labels or {})
    legend_labels = dict(legend_labels or {})
    colors = dict(colors or {})
    selected_color = colors.get("selected", "#d62728")
    pareto_color = colors.get("pareto", "#1f77b4")
    ideal_color = colors.get("ideal", "#333333")
    x_label = axis_labels.get("dbcv_normalized", r"Normalized DBCV ($\widetilde{D}$)")
    y_label = axis_labels.get("stability_normalized", r"Normalized $S_R$ ($\widetilde{S}$)")
    if not plot_roles:
        axis = axes.flat[0]
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No role with a multi-point Pareto front",
            ha="center",
            va="center",
            fontsize=11,
        )
        for extra_axis in list(axes.flat)[1:]:
            extra_axis.remove()
    for plot_index, (axis, role) in enumerate(zip(axes.flat, plot_roles)):
        axis.set_title(_panel_title(role, plot_index, role_labels), fontsize=11, pad=6)
        frame = selection_tables.get(role, pd.DataFrame()).copy()
        axis.scatter([1.0], [1.0], marker="x", s=80, color=ideal_color, linewidths=1.5, zorder=1)
        if frame.empty:
            axis.set_xlim(-0.05, 1.05)
            axis.set_ylim(-0.05, 1.05)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            continue
        pareto = frame.loc[frame["is_pareto"].fillna(frame.get("on_pareto", False)).astype(bool)].copy()
        if pareto.empty or len(pareto) <= 1:
            axis.set_xlim(-0.05, 1.05)
            axis.set_ylim(-0.05, 1.05)
            continue
        if "dbcv_normalized" not in pareto.columns or pareto["dbcv_normalized"].isna().all():
            pareto = compute_tchebycheff_scores(
                normalize_pareto_objectives(
                    pareto, stability_col=stability_col, dbcv_col=dbcv_col, role=role
                )
            )
        pareto = pareto.sort_values("dbcv_normalized")
        axis.plot(
            pareto["dbcv_normalized"],
            pareto["stability_normalized"],
            color=pareto_color,
            linewidth=1.1,
            alpha=0.8,
        )
        axis.scatter(
            pareto["dbcv_normalized"],
            pareto["stability_normalized"],
            s=55,
            facecolors=pareto_color,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )
        # Prefer recomputed Pareto coordinates: single-optimum roles may lack
        # normalized columns on the full selection table.
        selected_mask = _selected_mask(frame)
        selected_ids = set(frame.loc[selected_mask, "configuration_id"].astype(str))
        selected = pareto.loc[pareto["configuration_id"].astype(str).isin(selected_ids)]
        if selected.empty and selected_mask.any():
            selected = frame.loc[selected_mask]
        if not selected.empty:
            x_selected = selected.iloc[0].get("dbcv_normalized")
            y_selected = selected.iloc[0].get("stability_normalized")
            if pd.notna(x_selected) and pd.notna(y_selected):
                axis.scatter(
                    [float(x_selected)],
                    [float(y_selected)],
                    marker="*",
                    s=320,
                    facecolors=selected_color,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=4,
                )
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(plot_roles):]:
        axis.remove()
    handles = [
        Line2D([0], [0], marker="x", linestyle="None", color=ideal_color, markersize=8, label=legend_labels.get("ideal", "Ideal point (1, 1)")),
        Line2D([0], [0], marker="o", linestyle="-", color=pareto_color, markerfacecolor=pareto_color, markersize=7, label=legend_labels.get("pareto", "Pareto-optimal configurations")),
        Line2D([0], [0], marker="*", linestyle="None", color="black", markerfacecolor=selected_color, markersize=12, label=legend_labels.get("selected", SELECTED_CONFIGURATION_LEGEND)),
    ]
    if suptitle:
        figure.suptitle(suptitle, y=0.98, fontsize=12)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.97 if suptitle else 1.0))
    from manuscript_reporting import save_manuscript_figure

    save_manuscript_figure(figure, output_dir / filename)
    plt.close(figure)


def plot_pareto_normalized_with_knee(*args, **kwargs) -> None:
    """Compatibility alias for the former plotting function name."""
    plot_pareto_normalized_tchebycheff(*args, **kwargs)


__all__ = [
    "SELECTION_TOLERANCE",
    "KNEE_TOLERANCE",
    "identify_pareto_front",
    "mark_pareto_front",
    "normalize_pareto_objectives",
    "compute_tchebycheff_scores",
    "select_tchebycheff_configuration",
    "select_knee_configuration",
    "select_configuration_for_role",
    "summarize_selected_configurations",
    "print_role_selection_summary",
    "SELECTED_CONFIGURATION_LEGEND",
    "roles_with_multi_point_pareto_front",
    "plot_pareto_raw",
    "plot_pareto_normalized_tchebycheff",
    "plot_pareto_normalized_with_knee",
]
