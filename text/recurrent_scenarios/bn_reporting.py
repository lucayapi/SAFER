"""Publication-ready tables, figures and diagnostics for the latent BN pipeline."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scenario_pipeline import (
    StructuralEMResult,
    ROLES,
    _bn_config,
    _bn_parent_map,
    _edge_conditional_strength,
    _latent_scope,
    _short_semantic_label,
    _theme_label_map,
    fit_latent_bn_analysis,
    finalize_latent_bn,
    is_latent_conditioned,
)


def _bic_k1(selection: pd.DataFrame) -> float:
    k1 = selection[(selection["K"] == 1) & selection["selected_for_K"]]
    if not k1.empty:
        return float(k1["bic"].iloc[0])
    k1_any = selection[selection["K"] == 1]
    if k1_any.empty:
        return float("nan")
    return float(k1_any.loc[k1_any["admissible"], "bic"].min()) if k1_any["admissible"].any() else float("nan")


def write_k_selection_summary(
    selection: pd.DataFrame,
    selected: StructuralEMResult,
    output_dir: Path,
    selection_warnings: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Write enriched K-selection diagnostics (BIC, entropy, ICL)."""

    best = selection[selection["selected_for_K"]].copy()
    if best.empty:
        best = selection.groupby("K", as_index=False).first()
    summary_rows = []
    all_best_bic = float(best["bic"].min()) if not best.empty else float("nan")
    bic_k1 = _bic_k1(selection)
    for _, row in best.sort_values("K").iterrows():
        k_value = int(row["K"])
        bic = float(row["bic"])
        entropy = 0.0 if k_value == 1 else float(row.get("entropy", np.nan))
        normalized_entropy = 0.0 if k_value == 1 else float(row.get("normalized_entropy", np.nan))
        icl = bic + 2.0 * entropy if np.isfinite(entropy) else np.nan
        n_adm = int(selection.loc[(selection["K"] == k_value) & selection["admissible"]].shape[0])
        summary_rows.append({
            "K": k_value,
            "best_BIC": bic,
            "best_bic": bic,
            "best_ICL": icl,
            "delta_BIC": bic - all_best_bic,
            "delta_bic": bic - all_best_bic,
            "best_log_likelihood": float(row["log_likelihood"]),
            "number_parameters": float(row.get("number_parameters", np.nan)),
            "n_admissible_runs": n_adm,
            "n_admissible": n_adm,
            "convergence_rate": float(selection.loc[selection["K"] == k_value, "converged"].mean()),
            "min_N_eff": float(row.get("min_effective_n", row.get("min_N_eff", np.nan))),
            "min_effective_n": float(row.get("min_effective_n", np.nan)),
            "min_family_weight": float(row.get("min_weight", row.get("min_family_weight", np.nan))),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "ICL_diagnostic": icl,
            "mean_max_posterior": float(row.get("mean_max_posterior", np.nan)),
            "latent_heterogeneity_supported": k_value > 1,
            "selected_for_K": bool(row.get("selected_for_K", False)),
            "selected_final": bool(row.get("selected_final", False)),
            "BIC_K1": bic_k1,
            "delta_BIC_vs_K1": bic_k1 - bic if np.isfinite(bic_k1) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        all_best_icl = float(summary["best_ICL"].min())
        summary["delta_ICL"] = summary["best_ICL"] - all_best_icl
    if selected.n_states > 1 and np.isfinite(bic_k1):
        sel_bic = float(summary.loc[summary["selected_final"], "best_BIC"].iloc[0]) if summary["selected_final"].any() else float(selected.bic)
        summary["delta_BIC_vs_K1"] = bic_k1 - sel_bic
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "K_selection_summary.csv", index=False)
    summary.to_csv(output_dir / "K_selection.csv", index=False)
    if selection_warnings:
        pd.DataFrame({"warning": list(selection_warnings)}).to_csv(output_dir / "K_selection_warnings.csv", index=False)
    return summary


def write_selected_model_convergence(selection: pd.DataFrame, selected_k: int, output_dir: Path) -> pd.DataFrame:
    subset = selection[(selection["K"] == selected_k) & selection["selected_final"]]
    if subset.empty:
        return pd.DataFrame()
    row = subset.iloc[0]
    try:
        history = json.loads(str(row.get("full_iteration_history", row.get("last_10_log_likelihoods", "[]"))))
    except (json.JSONDecodeError, TypeError):
        history = []
    conv = pd.DataFrame([{
        "initialization": row.get("initialization"),
        "converged": bool(row.get("converged")),
        "n_iterations": int(row.get("n_iter", 0)),
        "final_log_likelihood": float(row.get("log_likelihood", np.nan)),
        "absolute_delta": float(row.get("last_loglik_delta", np.nan)),
        "relative_delta": float(row.get("relative_loglik_delta", np.nan)),
        "stopped_by_tolerance": bool(history[-1].get("stopped_by_tolerance", False)) if history else False,
        "stopped_by_max_iterations": bool(history[-1].get("stopped_by_max_iterations", False)) if history else False,
        "graph_stable": bool(row.get("same_graph", False)),
        "full_iteration_history": json.dumps(history),
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    conv.to_csv(output_dir / "selected_model_convergence.csv", index=False)
    return conv


def _prepare_k_selection_plot_data(selection: pd.DataFrame) -> tuple[pd.DataFrame, list[int], dict[int, int]]:
    plot_data = selection.copy()
    plot_data["init_rank"] = plot_data.groupby("K").cumcount()
    plot_data["n_initializations_K"] = plot_data.groupby("K")["K"].transform("size")
    candidate_k = sorted(int(value) for value in plot_data["K"].unique())
    k_positions = {value: index for index, value in enumerate(candidate_k)}
    plot_data["x_position"] = plot_data["K"].map(k_positions)
    plot_data["x_plot"] = plot_data["x_position"] + (
        plot_data["init_rank"] - (plot_data["n_initializations_K"] - 1) / 2
    ) * 0.035
    return plot_data, candidate_k, k_positions


def _style_k_selection_axis(
    axis,
    candidate_k: Sequence[int],
    k_positions: Mapping[int, int],
    selected_k: int,
) -> None:
    selected_position = k_positions.get(selected_k, 0)
    axis.axvline(selected_position, color="#D62728", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.set_xticks(range(len(candidate_k)))
    axis.set_xticklabels([str(value) for value in candidate_k])
    axis.set_xlabel("Latent cardinality K")
    axis.grid(alpha=0.25, axis="y")


def _plot_k_selection_all_inits_axis(axis, plot_data: pd.DataFrame, best_per_k: pd.DataFrame) -> None:
    from manuscript_reporting import K_SELECTION_ADMISSIBLE_COLOR, K_SELECTION_BEST_LINE_COLOR

    admissible = plot_data[plot_data["admissible"]]
    inadmissible = plot_data[~plot_data["admissible"]]
    axis.scatter(inadmissible["x_plot"], inadmissible["bic"], s=18, color="#D9D9D9", alpha=0.35, label="Non-admissible")
    axis.scatter(admissible["x_plot"], admissible["bic"], s=26, color=K_SELECTION_ADMISSIBLE_COLOR, alpha=0.55, label="Admissible")
    axis.plot(
        best_per_k["x_position"],
        best_per_k["bic"],
        color=K_SELECTION_BEST_LINE_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="Best admissible BIC",
    )
    axis.set_ylabel("Observed-data BIC")
    axis.legend(loc="best", frameon=False, fontsize=8)


def _plot_k_selection_best_axis(
    axis,
    best_per_k: pd.DataFrame,
    selected_k: int,
    k_positions: Mapping[int, int],
    selected_rows: pd.DataFrame,
) -> None:
    from manuscript_reporting import K_SELECTION_BEST_LINE_COLOR, K_SELECTION_SELECTED_COLOR

    axis.plot(
        best_per_k["x_position"],
        best_per_k["bic"],
        color=K_SELECTION_BEST_LINE_COLOR,
        linewidth=3,
        marker="o",
        markersize=9,
    )
    axis.set_ylabel("Observed-data BIC")
    if selected_rows.empty:
        return
    selected_position = k_positions.get(selected_k, 0)
    axis.scatter(
        selected_rows["x_position"],
        selected_rows["bic"],
        s=180,
        marker="*",
        color=K_SELECTION_SELECTED_COLOR,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    axis.annotate(
        f"K*={selected_k}",
        (selected_position, float(selected_rows["bic"].iloc[0])),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#B22222",
    )


def render_latent_k_selection_figure(
    selection: pd.DataFrame,
    selected_k: int,
    output_dir: Path,
    *,
    stem: str = "latent_K_selection",
) -> tuple[Path, Path]:
    from manuscript_reporting import save_manuscript_figure

    plot_data, candidate_k, k_positions = _prepare_k_selection_plot_data(selection)
    best_per_k = plot_data[plot_data["selected_for_K"]].sort_values("K")
    selected_rows = plot_data[plot_data["selected_final"]]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure, (all_axis, best_axis) = plt.subplots(
        1, 2, figsize=(15, 6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]},
    )
    _plot_k_selection_all_inits_axis(all_axis, plot_data, best_per_k)
    _plot_k_selection_best_axis(best_axis, best_per_k, selected_k, k_positions, selected_rows)
    for current_axis in (all_axis, best_axis):
        _style_k_selection_axis(current_axis, candidate_k, k_positions, selected_k)
    figure.tight_layout()
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    save_manuscript_figure(figure, png_path, dpi=220)
    save_manuscript_figure(figure, pdf_path, dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5))
    _plot_k_selection_best_axis(axis, best_per_k, selected_k, k_positions, selected_rows)
    _style_k_selection_axis(axis, candidate_k, k_positions, selected_k)
    figure.tight_layout()
    save_manuscript_figure(figure, figures_dir / "latent_K_selection_best.png", dpi=220)
    save_manuscript_figure(figure, figures_dir / "latent_K_selection_best.pdf", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.5, 5))
    _plot_k_selection_all_inits_axis(axis, plot_data, best_per_k)
    _style_k_selection_axis(axis, candidate_k, k_positions, selected_k)
    figure.tight_layout()
    save_manuscript_figure(figure, figures_dir / "latent_K_selection_all_inits.png", dpi=220)
    save_manuscript_figure(figure, figures_dir / "latent_K_selection_all_inits.pdf", dpi=220)
    plt.close(figure)

    return png_path, pdf_path


def optimal_k_by_criterion(k_summary: pd.DataFrame, column: str) -> int | None:
    """Return K minimizing one criterion column among admissible best-per-K rows."""

    if k_summary.empty or column not in k_summary.columns:
        return None
    valid = k_summary[k_summary[column].notna()]
    if valid.empty:
        return None
    return int(valid.loc[valid[column].idxmin(), "K"])


def _mark_criterion_optimum(
    axis,
    frame: pd.DataFrame,
    x_positions: np.ndarray,
    column: str,
    k_star: int | None,
    *,
    color: str,
) -> None:
    if k_star is None:
        return
    match = frame[frame["K"].astype(int) == int(k_star)]
    if match.empty:
        return
    x_index = int(np.where(frame["K"].astype(int).to_numpy() == int(k_star))[0][0])
    value = float(match[column].iloc[0])
    axis.scatter(
        x_positions[x_index],
        value,
        s=180,
        marker="*",
        color=color,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    axis.annotate(
        f"K*={k_star}",
        (x_positions[x_index], value),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#B22222",
    )


def render_latent_k_criteria_figure(k_summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Two-panel figure: best admissible BIC and ICL vs latent cardinality K."""

    from manuscript_reporting import K_SELECTION_BEST_LINE_COLOR, K_SELECTION_SELECTED_COLOR, save_manuscript_figure

    frame = k_summary.sort_values("K").copy()
    if frame.empty:
        raise ValueError("Cannot render K criteria figure from an empty summary.")
    icl_column = "best_ICL" if "best_ICL" in frame.columns else "ICL_diagnostic"
    k_values = frame["K"].astype(int).tolist()
    x_positions = np.arange(len(k_values))
    k_star_bic = optimal_k_by_criterion(frame, "best_BIC")
    k_star_icl = optimal_k_by_criterion(frame, icl_column)

    figure, (bic_axis, icl_axis) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    bic_axis.plot(
        x_positions,
        frame["best_BIC"],
        color=K_SELECTION_BEST_LINE_COLOR,
        linewidth=3,
        marker="o",
        markersize=9,
    )
    icl_axis.plot(
        x_positions,
        frame[icl_column],
        color=K_SELECTION_BEST_LINE_COLOR,
        linewidth=3,
        marker="o",
        markersize=9,
    )
    _mark_criterion_optimum(
        bic_axis, frame, x_positions, "best_BIC", k_star_bic, color=K_SELECTION_SELECTED_COLOR,
    )
    _mark_criterion_optimum(
        icl_axis, frame, x_positions, icl_column, k_star_icl, color=K_SELECTION_SELECTED_COLOR,
    )
    for axis, ylabel in ((bic_axis, "Observed-data BIC"), (icl_axis, "ICL")):
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(value) for value in k_values])
        axis.set_xlabel("Latent cardinality K")
        axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "latent_K_selection_bic_icl.png"
    pdf_path = figures_dir / "latent_K_selection_bic_icl.pdf"
    save_manuscript_figure(figure, png_path, dpi=220)
    save_manuscript_figure(figure, pdf_path, dpi=220)
    plt.close(figure)
    return png_path, pdf_path


def render_structural_em_convergence_figure(
    selection: pd.DataFrame,
    selected_k: int,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    from manuscript_reporting import save_manuscript_figure

    subset = selection[(selection["K"] == selected_k) & selection["selected_final"]]
    if subset.empty:
        return None, None
    row = subset.iloc[0]
    try:
        history = json.loads(str(row.get("full_iteration_history", "[]")))
    except (json.JSONDecodeError, TypeError):
        history = []
    if not history:
        try:
            lls = json.loads(str(row["last_10_log_likelihoods"]))
            history = [{"iteration": index + 1, "log_likelihood": value} for index, value in enumerate(lls)]
        except (json.JSONDecodeError, TypeError, KeyError):
            return None, None
    iterations = [item["iteration"] for item in history]
    lls = [item["log_likelihood"] for item in history]
    converged = bool(row.get("converged"))
    rel_delta = float(row.get("relative_loglik_delta", np.nan))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(iterations, lls, marker="o", color="#4C78A8")
    axis.set_xlabel("EM iteration")
    axis.set_ylabel("Observed log-likelihood")
    axis.grid(alpha=0.25)
    axis.text(0.02, 0.98, f"Converged: {'Yes' if converged else 'No'}", transform=axis.transAxes, va="top")
    axis.text(0.02, 0.90, f"Final relative delta = {rel_delta:.2e}", transform=axis.transAxes, va="top")
    figure.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "structural_em_convergence_selected_K.png"
    pdf_path = figures_dir / "structural_em_convergence_selected_K.pdf"
    save_manuscript_figure(figure, png_path, dpi=220)
    save_manuscript_figure(figure, pdf_path, dpi=220)
    plt.close(figure)
    if not converged:
        warnings.warn("Le modèle sélectionné n'a pas convergé selon les critères EM.", RuntimeWarning)
    return png_path, pdf_path


def write_latent_family_summary(
    result: StructuralEMResult,
    responsibilities: pd.DataFrame | np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    selection_warnings: list[str] | None = None,
) -> pd.DataFrame:
    if isinstance(responsibilities, pd.DataFrame):
        tau = responsibilities.filter(like="family_").to_numpy(dtype=float)
    else:
        tau = np.asarray(responsibilities, dtype=float)
    if result.n_states == 1:
        entropy = 0.0
        normalized_entropy = 0.0
    else:
        entropy = float(-np.sum(tau * np.log(np.clip(tau, 1e-300, 1.0))))
        normalized_entropy = entropy / max(len(tau) * math.log(max(result.n_states, 2)), 1e-12)
    hard = np.argmax(tau, axis=1)
    max_posterior = tau.max(axis=1)
    threshold = float(_bn_config(config).get("median_max_posterior_warning", 0.70))
    warnings_out = list(selection_warnings or [])
    rows = []
    for state in range(result.n_states):
        mask = hard == state
        certainties = max_posterior[mask] if mask.any() else max_posterior
        median_cert = float(np.median(certainties))
        if result.n_states > 1 and median_cert < threshold:
            warnings_out.append(f"Family {state + 1}: median_max_posterior={median_cert:.3f} < {threshold}.")
        rows.append({
            "family_id": state + 1,
            "omega": float(result.weights[state]),
            "N_eff": float(tau[:, state].sum()),
            "hard_assignment_count": int(mask.sum()),
            "mean_max_posterior": float(certainties.mean()),
            "median_max_posterior": median_cert,
            "q25_max_posterior": float(np.quantile(certainties, 0.25)),
            "q75_max_posterior": float(np.quantile(certainties, 0.75)),
            "fraction_below_0.50": float((certainties < 0.50).mean()),
            "fraction_below_0.60": float((certainties < 0.60).mean()),
            "fraction_below_0.70": float((certainties < 0.70).mean()),
            "normalized_entropy": normalized_entropy if state == 0 else np.nan,
        })
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "latent_family_summary.csv", index=False)
    return summary


def render_latent_family_weights_figure(
    family_summary: pd.DataFrame,
    responsibilities: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    from manuscript_reporting import ROLE_COLORS, role_boxplot_kwargs, save_manuscript_figure

    tau = responsibilities.filter(like="family_").to_numpy(dtype=float)
    hard = np.argmax(tau, axis=1)
    figure, (weight_axis, certainty_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    family_labels = [f"Family {int(family_id) + 1}" for family_id in family_summary["family_id"].astype(int)]
    x_positions = np.arange(len(family_labels))
    bar_colors = [ROLE_COLORS["A0"], ROLE_COLORS["A1"], ROLE_COLORS["B"], ROLE_COLORS["C"]]
    bars = weight_axis.bar(
        x_positions,
        family_summary["omega"],
        color=[bar_colors[index % len(bar_colors)] for index in range(len(family_labels))],
    )
    weight_axis.set_xticks(x_positions)
    weight_axis.set_xticklabels(family_labels)
    weight_axis.set_xlabel("Latent family")
    weight_axis.set_ylabel("omega")
    for bar, (_, row) in zip(bars, family_summary.iterrows()):
        weight_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['omega']:.1%} — N_eff = {row['N_eff']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    cert_by_family = [tau[hard == state, state] for state in range(tau.shape[1])]
    boxplot_roles = ["A0", "A1", "B", "C"]
    for family_index, values in enumerate(cert_by_family):
        role = boxplot_roles[family_index % len(boxplot_roles)]
        certainty_axis.boxplot(
            [values],
            positions=[family_index + 1],
            widths=0.55,
            showfliers=True,
            patch_artist=True,
            **role_boxplot_kwargs(role),
        )
    certainty_axis.set_xticks(np.arange(1, len(family_labels) + 1))
    certainty_axis.set_xticklabels(family_labels)
    certainty_axis.set_ylabel("Max posterior membership")
    figure.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "latent_family_weights_and_certainty.png"
    pdf_path = figures_dir / "latent_family_weights_and_certainty.pdf"
    save_manuscript_figure(figure, png_path, dpi=220)
    save_manuscript_figure(figure, pdf_path, dpi=220)
    plt.close(figure)
    return png_path, pdf_path


def enrich_family_factor_profiles(
    profiles: pd.DataFrame,
    factor_prevalence: pd.DataFrame | None,
    result: StructuralEMResult,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    enriched = profiles.copy()
    if factor_prevalence is not None and not factor_prevalence.empty:
        lookup = factor_prevalence.set_index("variable_name")["observation_prevalence"].to_dict()
        enriched["global_observation_prevalence"] = enriched["variable_name"].map(lookup)
    else:
        enriched["global_observation_prevalence"] = np.nan
    enriched["delta_probability"] = enriched["probability"] - enriched["global_observation_prevalence"]
    if result.n_states == 2:
        pivot = enriched.pivot(index="variable_name", columns="family_id", values="probability")
        if 1 in pivot.columns and 2 in pivot.columns:
            contrast = pivot[2] - pivot[1]
            enriched["family_contrast"] = enriched.apply(
                lambda row: float(contrast.get(row["variable_name"], np.nan)), axis=1,
            )
        else:
            enriched["family_contrast"] = np.nan
    else:
        enriched["family_contrast"] = np.nan
        for family_id in enriched["family_id"].unique():
            mask = enriched["family_id"] == family_id
            other = enriched.loc[~mask].groupby("variable_name")["probability"].mean()
            enriched.loc[mask, "contrast_vs_rest"] = enriched.loc[mask].apply(
                lambda row: float(row["probability"] - other.get(row["variable_name"], np.nan)), axis=1,
            )
    enriched.to_csv(output_dir / "family_factor_profiles.csv", index=False)
    return enriched


def render_family_profile_figures(
    profiles: pd.DataFrame,
    roles: Mapping[str, str],
    label_map: Mapping[str, str],
    result: StructuralEMResult,
    output_dir: Path,
) -> None:
    from manuscript_reporting import ROLE_COLORS, save_manuscript_figure

    pivot = profiles.pivot(index="variable_name", columns="family_id", values="probability").fillna(0)
    ordered = sorted(pivot.index, key=lambda node: (roles.get(node, ""), node))
    pivot = pivot.loc[ordered]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, max(5, len(pivot) * 0.22)))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set(yticks=np.arange(len(pivot)), yticklabels=pivot.index, xlabel="Latent family")
    figure.colorbar(image, ax=axis, label="P(X=1|Z)")
    figure.tight_layout()
    save_manuscript_figure(figure, figures_dir / "family_factor_profiles_full.png", dpi=220)
    save_manuscript_figure(figure, figures_dir / "family_factor_profiles_full.pdf", dpi=220)
    plt.close(figure)

    if "delta_probability" in profiles.columns:
        delta = profiles.pivot(index="variable_name", columns="family_id", values="delta_probability").fillna(0).loc[ordered]
        top_variables = delta.abs().max(axis=1).sort_values(ascending=False).head(min(20, len(delta))).index
        delta_top = delta.loc[top_variables]
        figure, axis = plt.subplots(figsize=(10, max(4, len(delta_top) * 0.35)))
        image = axis.imshow(delta_top.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axis.set(yticks=np.arange(len(delta_top)), yticklabels=delta_top.index, xlabel="Latent family")
        figure.colorbar(image, ax=axis, label="Delta probability")
        figure.tight_layout()
        save_manuscript_figure(figure, figures_dir / "family_factor_profiles_article.png", dpi=220)
        save_manuscript_figure(figure, figures_dir / "family_factor_profiles_article.pdf", dpi=220)
        plt.close(figure)

    if result.n_states == 2 and "family_contrast" in profiles.columns:
        contrast = profiles.drop_duplicates("variable_name").set_index("variable_name")["family_contrast"].loc[ordered]
        top = contrast.reindex(contrast.abs().sort_values(ascending=False).head(20).index)
        labels = [f"[{roles.get(v, '?')}] {label_map.get(v, _short_semantic_label(v))}" for v in top.index]
        figure, axis = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
        axis.barh(range(len(top)), top.values, color=[ROLE_COLORS["A0"] if value >= 0 else ROLE_COLORS["C"] for value in top.values])
        axis.set_yticks(range(len(top)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_xlabel("P(X=1|Family 2) - P(X=1|Family 1)")
        figure.tight_layout()
        save_manuscript_figure(figure, figures_dir / "family_factor_contrast_K2.png", dpi=220)
        save_manuscript_figure(figure, figures_dir / "family_factor_contrast_K2.pdf", dpi=220)
        plt.close(figure)


def write_parent_configuration_support(
    result: StructuralEMResult,
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = result.nodes
    data = matrix[nodes].to_numpy(dtype=np.int8)
    tau = result.responsibilities
    parents = _bn_parent_map(nodes, result.edges)
    index = {node: position for position, node in enumerate(nodes)}
    rows = []
    for child in nodes:
        parent_nodes = parents[child]
        if not parent_nodes:
            continue
        parent_indices = [index[parent] for parent in parent_nodes]
        combinations = list(__import__("itertools").product((0, 1), repeat=len(parent_indices)))
        if parent_indices:
            codes = sum(
                data[:, parent_index].astype(int) * (2 ** (len(parent_indices) - offset - 1))
                for offset, parent_index in enumerate(parent_indices)
            )
        else:
            codes = np.zeros(len(data), dtype=int)
        for combination_index, parent_values in enumerate(combinations):
            mask = codes == combination_index
            if is_latent_conditioned(roles[child], _latent_scope(config)):
                for state in range(result.n_states):
                    mass = tau[:, state] * mask
                    effective = float(mass.sum())
                    raw = int(mask.sum())
                    rows.append({
                        "child": child,
                        "child_role": roles[child],
                        "parents": "|".join(parent_nodes),
                        "n_parents": len(parent_nodes),
                        "parent_configuration": "|".join(map(str, parent_values)),
                        "family_id": state + 1,
                        "effective_count": effective,
                        "raw_count": raw,
                    })
            else:
                rows.append({
                    "child": child,
                    "child_role": roles[child],
                    "parents": "|".join(parent_nodes),
                    "n_parents": len(parent_nodes),
                    "parent_configuration": "|".join(map(str, parent_values)),
                    "family_id": None,
                    "effective_count": float(mask.sum()),
                    "raw_count": int(mask.sum()),
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = detail.groupby("child", as_index=False).agg(
            min_parent_configuration_count=("effective_count", "min"),
            median_parent_configuration_count=("effective_count", "median"),
            max_parent_configuration_count=("effective_count", "max"),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "parent_configuration_support.csv", index=False)
    summary.to_csv(output_dir / "parent_support_summary.csv", index=False)
    threshold = float(_bn_config(config).get("min_parent_configuration_support_warning", 5))
    if not summary.empty and (summary["min_parent_configuration_count"] < threshold).any():
        warnings.warn(
            f"CPT avec configurations parentales effective_count < {threshold}.",
            RuntimeWarning,
        )
    return detail, summary


def write_k_bic_article_summary(k_summary: pd.DataFrame, selected_k: int, output_dir: Path) -> pd.DataFrame:
    """Compact BIC table for the manuscript (K=1, K*, ΔBIC)."""

    rows = []
    for k_value in (1, selected_k):
        subset = k_summary[k_summary["K"] == k_value]
        if subset.empty:
            continue
        rows.append({"quantity": f"BIC(K={k_value})", "value": float(subset["best_BIC"].iloc[0])})
    if len(rows) == 2:
        delta = rows[0]["value"] - rows[1]["value"]
        rows.append({"quantity": f"ΔBIC_{{1→{selected_k}}}", "value": delta})
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "K_selection_bic_article.csv", index=False)
    return frame


def write_k_criteria_article_summary(k_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compact BIC/ICL comparison for the manuscript and concordance diagnostic."""

    icl_column = "best_ICL" if "best_ICL" in k_summary.columns else "ICL_diagnostic"
    k_star_bic = optimal_k_by_criterion(k_summary, "best_BIC")
    k_star_icl = optimal_k_by_criterion(k_summary, icl_column)
    rows: list[dict[str, Any]] = []
    for criterion, k_star, column in (
        ("BIC", k_star_bic, "best_BIC"),
        ("ICL", k_star_icl, icl_column),
    ):
        if k_star is None:
            continue
        value = float(k_summary.loc[k_summary["K"].astype(int) == int(k_star), column].iloc[0])
        rows.append({"criterion": criterion, "K_star": int(k_star), "value_at_K_star": value})
        k1_subset = k_summary[k_summary["K"].astype(int) == 1]
        if not k1_subset.empty:
            k1_value = float(k1_subset[column].iloc[0])
            rows[-1]["value_at_K1"] = k1_value
            rows[-1][f"delta_{{1→{k_star}}}"] = k1_value - value
    concordant = k_star_bic is not None and k_star_bic == k_star_icl
    rows.append({
        "criterion": "concordant",
        "K_star": concordant,
        "value_at_K_star": np.nan,
    })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "K_selection_criteria_article.csv", index=False)
    return frame


def run_structure_bootstrap(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    selected_k: int,
    result: StructuralEMResult,
    theme_dictionary: pd.DataFrame | None,
    output_dir: Path,
) -> pd.DataFrame | None:
    cfg = _bn_config(config)
    bootstrap_cfg = cfg.get("bn_structure_bootstrap", {})
    if not bool(bootstrap_cfg.get("enabled", False)):
        return None
    from scenario_pipeline import _fit_structural_em_initialization

    label_map = _theme_label_map(theme_dictionary)
    n_resamples = int(bootstrap_cfg.get("n_resamples", 30))
    fraction = float(bootstrap_cfg.get("sample_fraction", bootstrap_cfg.get("fraction", 0.8)))
    n_inits = int(bootstrap_cfg.get("n_initializations_per_resample", 3))
    seed = int(bootstrap_cfg.get("random_state", config.get("random_state", 42)))
    nodes = list(roles)
    data = matrix[nodes].to_numpy(dtype=np.int8)
    rng = np.random.default_rng(seed)
    edge_counts: dict[tuple[str, str], int] = {}
    edge_strengths: dict[tuple[str, str], list[float]] = {}
    n_samples = max(int(len(data) * fraction), 1)
    for _ in range(n_resamples):
        indices = rng.choice(len(data), size=n_samples, replace=True)
        sample = data[indices]
        best_bic = math.inf
        best_result = None
        for init_index in range(n_inits):
            candidate = _fit_structural_em_initialization(
                sample, nodes, dict(roles), selected_k,
                int(rng.integers(0, 1_000_000)), "random" if init_index else "empty", config,
            )
            if candidate.bic < best_bic:
                best_bic = candidate.bic
                best_result = candidate
        if best_result is None:
            continue
        for edge in best_result.edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            strength = _edge_conditional_strength(best_result, edge[0], edge[1])
            edge_strengths.setdefault(edge, []).append(strength)
    rows = []
    for (parent, child), count in sorted(edge_counts.items()):
        strengths = edge_strengths.get((parent, child), [0.0])
        rows.append({
            "parent": parent,
            "child": child,
            "parent_role": roles[parent],
            "child_role": roles[child],
            "parent_label": label_map.get(parent, parent),
            "child_label": label_map.get(child, child),
            "selection_count": count,
            "selection_frequency": count / n_resamples,
            "mean_conditional_strength": float(np.mean(strengths)),
            "median_conditional_strength": float(np.median(strengths)),
        })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "edge_bootstrap_stability.csv", index=False)
    if not frame.empty:
        render_edge_bootstrap_figure(frame, output_dir)
        write_learned_bn_stable_edges_table(result, frame, label_map, roles, config, output_dir)
    render_learned_bn_stable_dependencies(result, frame if not frame.empty else None, label_map, roles, config, output_dir)
    if not frame.empty:
        _write_diagnostic_bn_network_figures(result, frame, label_map, roles, config, output_dir)
    return frame


STABLE_BN_FIGURE_TITLE = "Stable dependencies in the learned Bayesian network"
STABLE_BN_FIGURE_LEGEND = (
    "Observed-factor dependencies retained in the selected Bayesian network and most "
    "consistently recovered under accident-level bootstrap resampling. Only edges exceeding "
    "the bootstrap-frequency threshold are displayed for readability. Factors are arranged "
    "according to their accident-process roles. Edge stability reflects resampling "
    "reproducibility and should not be interpreted as causal confidence."
)


def _stable_edges_for_display(
    result: StructuralEMResult,
    stability: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> list[tuple[tuple[str, str], float]]:
    threshold = float(_bn_config(config).get("article_edge_stability_threshold", 0.60))
    max_edges = int(_bn_config(config).get("article_bn_max_edges", 15))
    freq_lookup: dict[tuple[str, str], float] = {}
    if stability is not None and not stability.empty:
        freq_lookup = {
            (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
            for _, row in stability.iterrows()
        }
    candidates: list[tuple[tuple[str, str], float]] = []
    for parent, child in result.edges:
        edge = (parent, child)
        frequency = freq_lookup.get(edge)
        if frequency is None:
            continue
        if frequency >= threshold:
            candidates.append((edge, frequency))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:max_edges]


def write_learned_bn_stable_edges_table(
    result: StructuralEMResult,
    stability: pd.DataFrame | None,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    edges = _stable_edges_for_display(result, stability, config)
    rows = []
    for (parent, child), frequency in edges:
        rows.append({
            "Parent factor": label_map.get(parent, parent),
            "Child factor": label_map.get(child, child),
            "Direction": f"{roles[parent]} → {roles[child]}",
            "Bootstrap frequency": frequency,
        })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "learned_bn_stable_edges_table.csv", index=False)
    return frame


def render_learned_bn_stable_dependencies(
    result: StructuralEMResult,
    stability: pd.DataFrame | None,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Single Results BN figure: stable observed-factor dependencies only (no Z)."""

    import textwrap

    try:
        from matplotlib.patches import FancyBboxPatch
        from manuscript_reporting import ROLE_COLORS, ROLE_NODE_FILL, format_bootstrap_frequency, save_manuscript_figure
    except ImportError:
        return

    threshold = float(_bn_config(config).get("article_edge_stability_threshold", 0.60))
    edges = _stable_edges_for_display(result, stability, config)
    active_nodes: set[str] = set()
    for (parent, child), _ in edges:
        active_nodes.add(parent)
        active_nodes.add(child)

    nodes_by_role = {
        role: sorted(node for node in active_nodes if roles.get(node) == role)
        for role in ROLES
    }
    max_nodes_in_column = max((len(nodes_by_role[role]) for role in ROLES), default=0)
    if max_nodes_in_column == 0:
        max_nodes_in_column = 1

    vertical_spacing = 0.95
    positions: dict[str, tuple[float, float]] = {}
    for role_index, role in enumerate(ROLES):
        role_nodes = nodes_by_role[role]
        if not role_nodes:
            continue
        start_y = (len(role_nodes) - 1) / 2 * vertical_spacing
        for node_index, node in enumerate(role_nodes):
            positions[node] = (float(role_index), start_y - node_index * vertical_spacing)

    figure_height = max(3.2, max_nodes_in_column * 0.82 + 1.0)
    figure, axis = plt.subplots(figsize=(max(6.5, 1.7 * len(ROLES)), figure_height))
    y_span = max(max_nodes_in_column - 0.5, 0.5) * vertical_spacing
    header_y = y_span + 0.45
    axis.set_xlim(-0.55, len(ROLES) - 0.45)
    axis.set_ylim(-y_span - 0.35, header_y + 0.25)
    axis.axis("off")

    for role_index, role in enumerate(ROLES):
        axis.text(
            role_index,
            header_y,
            role,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=ROLE_COLORS[role],
        )
        if not nodes_by_role[role]:
            axis.text(
                role_index,
                header_y - 0.28,
                f"No edge ≥ {threshold:.2f}",
                ha="center",
                va="top",
                fontsize=7,
                color="#888888",
                style="italic",
            )

    for node, (x_pos, y_pos) in positions.items():
        role = roles[node]
        label = str(label_map.get(node, node))
        wrapped = "\n".join(textwrap.wrap(label, width=22)[:3])
        line_count = wrapped.count("\n") + 1
        box_height = 0.28 + 0.11 * line_count
        box = FancyBboxPatch(
            (x_pos - 0.38, y_pos - box_height / 2),
            0.76,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=ROLE_NODE_FILL[role],
            edgecolor=ROLE_COLORS[role],
            linewidth=0.9,
            alpha=0.98,
            transform=axis.transData,
        )
        axis.add_patch(box)
        axis.text(x_pos, y_pos, wrapped, ha="center", va="center", fontsize=6.4, color="#222222")

    for edge_index, ((parent, child), frequency) in enumerate(edges):
        start_x, start_y = positions[parent]
        end_x, end_y = positions[child]
        rad = 0.12 if edge_index % 2 == 0 else -0.12
        axis.annotate(
            "",
            xy=(end_x - 0.38, end_y),
            xytext=(start_x + 0.38, start_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#333333",
                linewidth=0.8 + 3.2 * frequency,
                alpha=0.55 + 0.45 * frequency,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )
        mid_x = (start_x + end_x) / 2
        mid_y = max(start_y, end_y) + 0.18 + 0.04 * (edge_index % 3)
        axis.text(
            mid_x,
            mid_y,
            format_bootstrap_frequency(frequency),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    if not edges:
        axis.text(
            0.5,
            0.5,
            f"No stable edge ≥ {threshold:.2f}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )

    figure.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, figures_dir / "learned_bn_stable_dependencies.png", dpi=220)
    save_manuscript_figure(figure, figures_dir / "learned_bn_stable_dependencies.pdf", dpi=220)
    plt.close(figure)


def render_edge_bootstrap_figure(stability: pd.DataFrame, output_dir: Path) -> None:
    from manuscript_reporting import save_manuscript_figure

    frame = stability.sort_values("selection_frequency", ascending=True)
    labels = frame.apply(
        lambda row: f"{row.get('parent_label', row['parent'])} → {row.get('child_label', row['child'])}", axis=1,
    )
    figure, axis = plt.subplots(figsize=(10, max(4, len(frame) * 0.28)))
    axis.barh(labels, frame["selection_frequency"], color="#4C78A8")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Bootstrap selection frequency")
    figure.tight_layout()
    figures_dir = output_dir / "figures" / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(figure, figures_dir / "edge_bootstrap_stability.png", dpi=220)
    save_manuscript_figure(figure, figures_dir / "edge_bootstrap_stability.pdf", dpi=220)
    plt.close(figure)


def _write_diagnostic_bn_network_figures(
    result: StructuralEMResult,
    stability: pd.DataFrame,
    label_map: Mapping[str, str],
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    if not bool(_bn_config(config).get("write_diagnostic_bn_figures", False)):
        return
    try:
        import networkx as nx
        from manuscript_reporting import save_manuscript_figure
    except ImportError:
        return
    diagnostic_dir = output_dir / "figures" / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    freq_lookup = {(r["parent"], r["child"]): float(r["selection_frequency"]) for _, r in stability.iterrows()}
    graph = nx.DiGraph()
    graph.add_nodes_from(result.nodes)
    graph.add_edges_from(result.edges)
    positions = {}
    for role_index, role in enumerate(ROLES):
        role_nodes = [node for node in result.nodes if roles[node] == role]
        center = (len(role_nodes) - 1) / 2
        for node_index, node in enumerate(sorted(role_nodes)):
            positions[node] = (role_index, center - node_index)
    labels = {node: _short_semantic_label(label_map.get(node, node)) for node in result.nodes}
    figure, axis = plt.subplots(figsize=(14, max(6, len(result.nodes) * 0.2)))
    nx.draw_networkx_nodes(graph, positions, ax=axis, node_size=1200,
                           node_color=[plt.cm.tab10(i % 10) for i, _ in enumerate(graph.nodes)])
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=7)
    for edge in graph.edges:
        freq = freq_lookup.get(edge, 0.0)
        nx.draw_networkx_edges(graph, positions, edgelist=[edge], ax=axis, arrows=True,
                               width=0.5 + 3.5 * freq, alpha=0.3 + 0.7 * freq)
    axis.axis("off")
    figure.tight_layout()
    save_manuscript_figure(figure, diagnostic_dir / "learned_bn_stable_edges.png", dpi=220)
    save_manuscript_figure(figure, diagnostic_dir / "learned_bn_stable_edges.pdf", dpi=220)
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, np.generic)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_bn_diagnostic_summary(
    result: StructuralEMResult,
    selection: pd.DataFrame,
    k_summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    prototypes: pd.DataFrame,
    matrix: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
    warnings_list: Sequence[str] | None = None,
    parent_summary: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
) -> dict[str, Any]:
    cfg = _bn_config(config)
    selected_row = selection[selection["selected_final"]]
    best_bic = float(selected_row["bic"].iloc[0]) if not selected_row.empty else float("nan")
    bic_k1 = _bic_k1(selection)
    delta_vs_k1 = bic_k1 - best_bic if np.isfinite(bic_k1) else float("nan")
    icl_column = "best_ICL" if "best_ICL" in k_summary.columns else "ICL_diagnostic"
    k_star_bic = optimal_k_by_criterion(k_summary, "best_BIC")
    k_star_icl = optimal_k_by_criterion(k_summary, icl_column)
    icl_k1 = float("nan")
    if not k_summary.empty and icl_column in k_summary.columns:
        k1_rows = k_summary[k_summary["K"].astype(int) == 1]
        if not k1_rows.empty:
            icl_k1 = float(k1_rows[icl_column].iloc[0])
    selected_icl = float("nan")
    if k_star_bic is not None and icl_column in k_summary.columns:
        selected_rows = k_summary[k_summary["K"].astype(int) == int(k_star_bic)]
        if not selected_rows.empty:
            selected_icl = float(selected_rows[icl_column].iloc[0])
    delta_icl_vs_k1 = icl_k1 - selected_icl if np.isfinite(icl_k1) and np.isfinite(selected_icl) else float("nan")
    tau = result.responsibilities
    if result.n_states == 1:
        entropy = normalized_entropy = 0.0
    else:
        entropy = float(-np.sum(tau * np.log(np.clip(tau, 1e-300, 1.0))))
        normalized_entropy = entropy / max(len(tau) * math.log(max(result.n_states, 2)), 1e-12)
    max_posterior = tau.max(axis=1)
    grid = sorted(selection[selection["selected_for_K"]]["K"].unique()) if "selected_for_K" in selection.columns else [result.n_states]
    mpe_gap_threshold = float(cfg.get("near_equivalent_mpe_log_gap", 0.10))
    n_near_mpe = 0
    if "mpe_log_gap_1_2" in scenarios.columns:
        n_near_mpe = int((scenarios["mpe_log_gap_1_2"].astype(float) < mpe_gap_threshold).sum())
    n_exact = int(prototypes.get("prototype_exact_mpe_match", pd.Series(dtype=bool)).sum()) if "prototype_exact_mpe_match" in prototypes.columns else 0
    n_non_exact = int((~prototypes.get("prototype_exact_mpe_match", pd.Series(dtype=bool))).sum()) if "prototype_exact_mpe_match" in prototypes.columns else 0
    payload = {
        "selected_K": int(result.n_states),
        "selected_BIC": best_bic,
        "BIC_K1": bic_k1,
        "delta_BIC_vs_K1": delta_vs_k1,
        "K_star_BIC": k_star_bic,
        "K_star_ICL": k_star_icl,
        "criteria_concordant": bool(k_star_bic is not None and k_star_bic == k_star_icl),
        "selected_ICL": selected_icl,
        "ICL_K1": icl_k1,
        "delta_ICL_vs_K1": delta_icl_vs_k1,
        "latent_heterogeneity_supported": result.n_states > 1,
        "K_on_lower_boundary": result.n_states == min(grid) if grid else False,
        "K_on_upper_boundary": result.n_states == max(grid) if grid else False,
        "convergence_rate_selected_K": float(selection.loc[selection["K"] == result.n_states, "converged"].mean()) if len(selection) else np.nan,
        "selected_model_converged": bool(selected_row["converged"].iloc[0]) if not selected_row.empty else False,
        "selected_model_final_relative_delta": float(selected_row["relative_loglik_delta"].iloc[0]) if not selected_row.empty else np.nan,
        "n_accidents": int(len(matrix)),
        "n_factors": len(result.nodes),
        "n_edges": len(result.edges),
        "min_family_weight": float(result.weights.min()),
        "min_family_N_eff": float(tau.sum(axis=0).min()),
        "mean_max_posterior": float(max_posterior.mean()),
        "median_max_posterior": float(np.median(max_posterior)),
        "normalized_entropy": normalized_entropy,
        "min_parent_configuration_count": float(parent_summary["min_parent_configuration_count"].min()) if parent_summary is not None and not parent_summary.empty else np.nan,
        "n_bootstrap_edges": int(len(bootstrap)) if bootstrap is not None else 0,
        "median_edge_selection_frequency": float(bootstrap["selection_frequency"].median()) if bootstrap is not None and not bootstrap.empty else np.nan,
        "fraction_edges_frequency_ge_050": float((bootstrap["selection_frequency"] >= 0.5).mean()) if bootstrap is not None and not bootstrap.empty else np.nan,
        "fraction_edges_frequency_ge_075": float((bootstrap["selection_frequency"] >= 0.75).mean()) if bootstrap is not None and not bootstrap.empty else np.nan,
        "min_family_scenario_support": float(scenarios["family_positive_support"].min()) if not scenarios.empty and "family_positive_support" in scenarios.columns else np.nan,
        "min_global_scenario_support": float(scenarios["global_positive_support"].min()) if not scenarios.empty and "global_positive_support" in scenarios.columns else np.nan,
        "min_support_enrichment": float(scenarios["support_enrichment_ratio"].min()) if not scenarios.empty and "support_enrichment_ratio" in scenarios.columns else np.nan,
        "max_support_enrichment": float(scenarios["support_enrichment_ratio"].max()) if not scenarios.empty and "support_enrichment_ratio" in scenarios.columns else np.nan,
        "n_exact_scenario_prototypes": n_exact,
        "n_non_exact_scenario_prototypes": n_non_exact,
        "n_near_equivalent_MPE_families": n_near_mpe,
        "latent_scope": _latent_scope(config),
        "sum_omega": float(result.weights.sum()),
        "sum_N_eff": float(tau.sum()),
        "warnings": list(warnings_list or []),
    }
    payload = _json_safe(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(output_dir / "bn_diagnostic_summary.csv", index=False)
    (output_dir / "bn_diagnostic_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_latent_scope_sensitivity(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame | None:
    cfg = _bn_config(config)
    sensitivity = cfg.get("latent_scope_sensitivity", {})
    if not bool(sensitivity.get("enabled", False)):
        return None
    alternatives = [str(value) for value in sensitivity.get("alternatives", ["upstream_only", "all_roles"])]
    rows = []
    for scope in alternatives:
        scoped_config = dict(config)
        scoped_config["bayesian_networks"] = {**cfg, "latent_scope": scope, "latent_states": [2], "n_initializations": 1, "show_progress": False}
        try:
            selected, _, _ = fit_latent_bn_analysis(matrix, roles, scoped_config, output_dir / f"_scope_{scope}")
            final = finalize_latent_bn(selected, matrix, roles, scoped_config)
            rows.append({"latent_scope": scope, "K": final.n_states, "bic": float(final.bic), "pgmpy_valid": bool(final.model.check_model())})
        except Exception as error:
            rows.append({"latent_scope": scope, "error": str(error)})
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "latent_scope_sensitivity.csv", index=False)
    return frame


def write_bn_reporting(
    result: StructuralEMResult,
    selection: pd.DataFrame,
    scenarios: pd.DataFrame,
    prototypes: pd.DataFrame,
    profiles: pd.DataFrame,
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    config: Mapping[str, Any],
    output_dir: Path,
    selection_warnings: Sequence[str] | None = None,
    factor_prevalence: pd.DataFrame | None = None,
    responsibilities: pd.DataFrame | None = None,
    theme_dictionary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if responsibilities is None:
        responsibilities = pd.read_parquet(output_dir / "posterior_responsibilities.parquet")
    label_map = _theme_label_map(theme_dictionary)
    all_warnings = list(selection_warnings or [])
    k_summary = write_k_selection_summary(selection, result, output_dir, selection_warnings)
    write_k_bic_article_summary(k_summary, result.n_states, output_dir)
    write_k_criteria_article_summary(k_summary, output_dir)
    write_selected_model_convergence(selection, result.n_states, output_dir)
    render_latent_k_selection_figure(selection, result.n_states, output_dir)
    render_latent_k_criteria_figure(k_summary, output_dir)
    render_structural_em_convergence_figure(selection, result.n_states, output_dir)
    family_summary = write_latent_family_summary(result, responsibilities, config, output_dir, all_warnings)
    if result.n_states > 1:
        render_latent_family_weights_figure(family_summary, responsibilities, output_dir)
    enriched = enrich_family_factor_profiles(profiles, factor_prevalence, result, label_map, roles, output_dir)
    render_family_profile_figures(enriched, roles, label_map, result, output_dir)
    _, parent_summary = write_parent_configuration_support(result, matrix, roles, config, output_dir)
    bootstrap = run_structure_bootstrap(matrix, roles, config, result.n_states, result, theme_dictionary, output_dir)

    from scenario_latex_export import write_recurrent_scenarios_article

    write_recurrent_scenarios_article(scenarios, output_dir)
    diagnostic = write_bn_diagnostic_summary(
        result, selection, k_summary, scenarios, prototypes, matrix, config, output_dir,
        all_warnings, parent_summary, bootstrap,
    )
    return diagnostic
