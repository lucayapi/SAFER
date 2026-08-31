"""Deterministic Pareto-front screening and geometric knee-point selection.

For each role, configurations are compared on two maximized objectives:
accident-level reproducibility ``S_R`` (``stability``) and UMAP-space ``DBCV``
(``dbcv_umap``). The Pareto-optimal set is normalized per role, then the
geometric knee point is chosen as the configuration with largest perpendicular
distance to the reference line joining the two extreme solutions.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KNEE_TOLERANCE = 1e-12
NORM_TOLERANCE = 1e-9
STABILITY_COL = "stability"
DBCV_COL = "dbcv_umap"


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
    result["is_selected_knee"] = False
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
    if math.isclose(d_max, d_min, rel_tol=0.0, abs_tol=KNEE_TOLERANCE):
        warnings.warn(
            f"[{role_label}] DBCV is constant on the Pareto front; "
            "dbcv_normalized set to 0.5 for all Pareto configurations.",
            stacklevel=2,
        )
        result["dbcv_normalized"] = 0.5
    else:
        result["dbcv_normalized"] = (dbcvs - d_min) / (d_max - d_min)
    if math.isclose(s_max, s_min, rel_tol=0.0, abs_tol=KNEE_TOLERANCE):
        warnings.warn(
            f"[{role_label}] S_R is constant on the Pareto front; "
            "stability_normalized set to 0.5 for all Pareto configurations.",
            stacklevel=2,
        )
        result["stability_normalized"] = 0.5
    else:
        result["stability_normalized"] = (stabilities - s_min) / (s_max - s_min)
    return result


def compute_geometric_knee(
    df_pareto: pd.DataFrame,
    *,
    dbcv_norm_col: str = "dbcv_normalized",
    stability_norm_col: str = "stability_normalized",
) -> pd.DataFrame:
    """Compute perpendicular distance to the reference line ``D_norm + S_norm = 1``.

    ``d_knee(c) = (D_norm(c) + S_norm(c) - 1) / sqrt(2)`` measures deviation
    toward the ideal point ``I = (1, 1)``.
    """
    result = df_pareto.copy()
    d_norm = result[dbcv_norm_col].astype(float)
    s_norm = result[stability_norm_col].astype(float)
    result["knee_distance"] = (d_norm + s_norm - 1.0) / math.sqrt(2.0)
    return result


def project_knee_to_reference_line(x_k: float, y_k: float) -> tuple[float, float]:
    """Orthogonal projection of ``K = (x_k, y_k)`` onto ``x + y = 1``."""
    delta = (x_k + y_k - 1.0) / 2.0
    x_h = x_k - delta
    y_h = y_k - delta
    return float(x_h), float(y_h)


def _select_knee_from_pareto(
    pareto: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    configuration_col: str = "configuration_id",
) -> str:
    """Return the configuration_id of the geometric knee on a Pareto subset.

    Tie-break when ``knee_distance`` ties (``tol=1e-12``):
    higher raw ``S_R``, then higher raw DBCV, then lexicographic ``configuration_id``.
    """
    normalized = normalize_pareto_objectives(
        pareto,
        stability_col=stability_col,
        dbcv_col=dbcv_col,
        role=str(pareto["role"].iloc[0]) if "role" in pareto.columns and not pareto.empty else None,
    )
    with_knee = compute_geometric_knee(normalized)
    max_distance = float(with_knee["knee_distance"].max())
    tied = with_knee.loc[
        np.isclose(with_knee["knee_distance"].astype(float), max_distance, rtol=0.0, atol=KNEE_TOLERANCE)
    ]
    tied = tied.sort_values(
        [stability_col, dbcv_col, configuration_col],
        ascending=[False, False, True],
        na_position="last",
    )
    return str(tied.iloc[0][configuration_col])


def select_knee_configuration(
    df_role: pd.DataFrame,
    *,
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
) -> tuple[pd.DataFrame, str, str]:
    """Identify Pareto front, compute knee point, flag ``is_selected_knee``.

    Returns ``(annotated_table, selected_configuration_id, selection_rule)``.
    ``selection_rule`` is ``single_pareto`` or ``geometric_knee``.
    """
    marked = identify_pareto_front(df_role, stability_col=stability_col, dbcv_col=dbcv_col)
    pareto = marked.loc[marked["is_pareto"]].copy()
    if pareto.empty:
        return marked, "", "none"
    if len(pareto) == 1:
        selected_id = str(pareto.iloc[0]["configuration_id"])
        marked["is_selected_knee"] = marked["configuration_id"].astype(str).eq(selected_id)
        marked["selected"] = marked["is_selected_knee"]
        return marked, selected_id, "single_pareto"

    selected_id = _select_knee_from_pareto(pareto, stability_col=stability_col, dbcv_col=dbcv_col)
    pareto_normalized = compute_geometric_knee(
        normalize_pareto_objectives(
            pareto,
            stability_col=stability_col,
            dbcv_col=dbcv_col,
            role=str(pareto["role"].iloc[0]) if "role" in pareto.columns else None,
        )
    )
    knee_lookup = pareto_normalized.set_index("configuration_id")[
        ["stability_normalized", "dbcv_normalized", "knee_distance"]
    ].to_dict("index")
    for column in ("stability_normalized", "dbcv_normalized", "knee_distance"):
        marked[column] = marked["configuration_id"].astype(str).map(
            lambda configuration_id, col=column: knee_lookup.get(configuration_id, {}).get(col, np.nan)
        )
    marked["is_selected_knee"] = marked["configuration_id"].astype(str).eq(selected_id)
    marked["selected"] = marked["is_selected_knee"]
    return marked, selected_id, "geometric_knee"


def select_configuration_for_role(
    merged: pd.DataFrame,
    semantic_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Select configuration via Pareto + geometric knee (``semantic_scores`` ignored)."""
    del semantic_scores
    return select_knee_configuration(merged)


def summarize_selected_configurations(
    selection_tables: Mapping[str, pd.DataFrame],
    *,
    parameter_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """Build one summary row per role for the knee-selected configuration."""
    rows: list[dict] = []
    for role, table in selection_tables.items():
        if table.empty:
            continue
        selected = table.loc[table["is_selected_knee"].fillna(False).astype(bool)]
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
            "knee_distance": row.get("knee_distance"),
            "selection_rule": "geometric_knee" if int(table["is_pareto"].sum()) > 1 else "single_pareto",
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
    """Print console summary for one role (Pareto extremes and geometric knee)."""
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
    knee = table.loc[table["configuration_id"].astype(str).eq(str(selected_id))]
    if not knee.empty:
        knee_row = knee.iloc[0]
        print("\nGeometric knee:")
        print(f"configuration = {selected_id}")
        print(f"S_R = {knee_row.get(stability_col)}")
        print(f"DBCV = {knee_row.get(dbcv_col)}")
        print(f"normalized S_R = {knee_row.get('stability_normalized')}")
        print(f"normalized DBCV = {knee_row.get('dbcv_normalized')}")
        print(f"knee distance = {knee_row.get('knee_distance')}")
    print("-" * 50)


def plot_pareto_raw(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    roles: Sequence[str] = ("A0", "A1", "B", "C"),
    filename: str = "stability_landscape_all_roles.png",
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    suptitle: str | None = None,
) -> None:
    """Four-panel scatter of DBCV versus S_R with Pareto front and knee star."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_roles = tuple(roles)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    knee_color = "#d62728"
    pareto_color = "#1f77b4"
    other_color = "#bdbdbd"
    for axis, role in zip(axes.flat, plot_roles):
        frame = selection_tables.get(role, pd.DataFrame()).copy()
        if frame.empty:
            axis.text(0.5, 0.5, "No configuration", ha="center", va="center")
            axis.set_xlabel("DBCV")
            axis.set_ylabel(r"$S_R$")
            axis.set_title(role, fontsize=11, pad=6)
            continue
        valid = frame[dbcv_col].notna() & frame[stability_col].notna()
        base = frame.loc[valid].copy()
        is_pareto = base["is_pareto"].fillna(base.get("on_pareto", False)).astype(bool)
        is_knee = base["is_selected_knee"].fillna(base.get("selected", False)).astype(bool)
        others = base.loc[~is_pareto]
        pareto_all = base.loc[is_pareto].sort_values(dbcv_col)
        knee = base.loc[is_knee]
        if not others.empty:
            axis.scatter(
                others[dbcv_col],
                others[stability_col],
                s=18,
                c=other_color,
                alpha=0.55,
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
            if not knee.empty:
                knee_index = set(knee.index)
                non_knee = pareto_all.loc[~pareto_all.index.isin(knee_index)]
            else:
                non_knee = pareto_all
            if not non_knee.empty:
                axis.scatter(
                    non_knee[dbcv_col],
                    non_knee[stability_col],
                    marker="o",
                    s=70,
                    facecolors=pareto_color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=3,
                )
        if not knee.empty:
            axis.scatter(
                knee[dbcv_col],
                knee[stability_col],
                marker="*",
                s=320,
                facecolors=knee_color,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        axis.set_xlabel("DBCV")
        axis.set_ylabel(r"$S_R$")
        axis.set_title(role, fontsize=11, pad=6)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(plot_roles):]:
        axis.remove()
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Candidate configurations", markerfacecolor=other_color, markersize=6),
        Line2D([0], [0], marker="o", linestyle="-", color=pareto_color, label="Pareto-optimal configurations", markerfacecolor=pareto_color, markersize=7),
        Line2D([0], [0], marker="*", linestyle="None", color="black", label="Geometric knee", markerfacecolor=knee_color, markersize=12),
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
    figure.savefig(output_dir / filename, dpi=250, bbox_inches="tight")
    plt.close(figure)


def plot_pareto_normalized_with_knee(
    selection_tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    roles: Sequence[str] = ("A0", "A1", "B", "C"),
    filename: str = "pareto_normalized_knee_all_roles.png",
    stability_col: str = STABILITY_COL,
    dbcv_col: str = DBCV_COL,
    suptitle: str = "Normalized Pareto fronts and geometric knee points",
    show_perpendicular_deviation: bool = True,
    perpendicular_roles: Sequence[str] | None = None,
) -> None:
    """Normalized objective space with reference line, ideal point and knee projection."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_roles = tuple(roles)
    deviation_roles = set(perpendicular_roles if perpendicular_roles is not None else (plot_roles[0],))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    knee_color = "#d62728"
    pareto_color = "#1f77b4"
    ref_color = "#888888"
    ref_x = np.linspace(0.0, 1.0, 100)
    drew_deviation = False
    for axis, role in zip(axes.flat, plot_roles):
        frame = selection_tables.get(role, pd.DataFrame()).copy()
        axis.plot(ref_x, 1.0 - ref_x, color=ref_color, linestyle="--", linewidth=1.0)
        axis.scatter([1.0], [1.0], marker="x", s=80, color="#333333", linewidths=1.5, zorder=1)
        if frame.empty:
            axis.set_xlim(-0.05, 1.05)
            axis.set_ylim(-0.05, 1.05)
            axis.set_xlabel(r"Normalized DBCV ($\widetilde{D}$)")
            axis.set_ylabel(r"Normalized $S_R$ ($\widetilde{S}$)")
            axis.set_title(role, fontsize=11, pad=6)
            continue
        pareto = frame.loc[frame["is_pareto"].fillna(frame.get("on_pareto", False)).astype(bool)].copy()
        if pareto.empty:
            axis.set_xlim(-0.05, 1.05)
            axis.set_ylim(-0.05, 1.05)
            axis.set_title(role, fontsize=11, pad=6)
            continue
        if "dbcv_normalized" not in pareto.columns or pareto["dbcv_normalized"].isna().all():
            pareto = compute_geometric_knee(
                normalize_pareto_objectives(
                    pareto,
                    stability_col=stability_col,
                    dbcv_col=dbcv_col,
                    role=role,
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
        knee = frame.loc[frame["is_selected_knee"].fillna(frame.get("selected", False)).astype(bool)]
        if not knee.empty:
            x_k = float(knee.iloc[0]["dbcv_normalized"])
            y_k = float(knee.iloc[0]["stability_normalized"])
            if show_perpendicular_deviation and role in deviation_roles:
                x_h, y_h = project_knee_to_reference_line(x_k, y_k)
                axis.plot([x_h, x_k], [y_h, y_k], color=knee_color, linestyle=":", linewidth=1.2, zorder=3)
                drew_deviation = True
            axis.scatter(
                [x_k],
                [y_k],
                marker="*",
                s=320,
                facecolors=knee_color,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel(r"Normalized DBCV ($\widetilde{D}$)")
        axis.set_ylabel(r"Normalized $S_R$ ($\widetilde{S}$)")
        axis.set_title(role, fontsize=11, pad=6)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(plot_roles):]:
        axis.remove()
    handles = [
        Line2D([0], [0], color=ref_color, linestyle="--", linewidth=1.2, label="Extreme-point reference line"),
        Line2D([0], [0], marker="x", linestyle="None", color="#333333", markersize=8, label="Ideal point (1, 1)"),
        Line2D([0], [0], marker="o", linestyle="-", color=pareto_color, markerfacecolor=pareto_color, markersize=7, label="Pareto-optimal configurations"),
        Line2D([0], [0], marker="*", linestyle="None", color="black", markerfacecolor=knee_color, markersize=12, label="Geometric knee"),
    ]
    if drew_deviation:
        handles.append(
            Line2D([0], [0], color=knee_color, linestyle=":", linewidth=1.2, label="Perpendicular deviation")
        )
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
    figure.savefig(output_dir / filename, dpi=250, bbox_inches="tight")
    plt.close(figure)


__all__ = [
    "KNEE_TOLERANCE",
    "identify_pareto_front",
    "mark_pareto_front",
    "normalize_pareto_objectives",
    "compute_geometric_knee",
    "project_knee_to_reference_line",
    "select_knee_configuration",
    "select_configuration_for_role",
    "summarize_selected_configurations",
    "print_role_selection_summary",
    "plot_pareto_raw",
    "plot_pareto_normalized_with_knee",
]
