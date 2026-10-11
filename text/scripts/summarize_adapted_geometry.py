"""Summarize matched refits and fit accident-clustered error associations.

The analysis is descriptive. A role margin uses the target's true label and is
therefore a diagnostic variable, not a deployable confidence score or cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replication.runner import load_replication_config, output_root, task_matrix, training_seeds

METRICS = ("source_distance_k10", "source_label_entropy_k10", "role_margin")
COMMON_TARGETS = ("caou", "nicollin")


def load_complete(config: dict) -> tuple[pd.DataFrame, dict[tuple[str, str, str], list[Path]]]:
    root = output_root(config).parent / "features"
    comparisons = []
    groups: dict[tuple[str, str, str], list[Path]] = {}
    tasks = task_matrix(config)
    expected_tasks = len(config["models"]) * len(training_seeds(config))
    if len(tasks) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} refits, found {len(tasks)}")
    for task in tasks:
        model_id, seed = task["model_id"], int(task["training_seed"])
        directory = root / model_id / f"seed_{seed}"
        complete = directory / "complete.json"
        if not complete.is_file():
            raise FileNotFoundError(f"Missing completed geometry task: {complete}")
        info = json.loads(complete.read_text(encoding="utf-8"))
        if info.get("model_id") != model_id or info.get("seed") != seed or len(info.get("targets", [])) != 3:
            raise ValueError(f"Invalid geometry manifest: {complete}")
        comparison = directory / "run_comparison.csv"
        if not comparison.is_file():
            raise FileNotFoundError(comparison)
        comp = pd.read_csv(comparison)
        if len(comp) != 3 or set(comp.target) != set(info["targets"]):
            raise ValueError(f"Incomplete BA comparison: {comparison}")
        comparisons.append(comp)
        key = (str(comp.backbone.iloc[0]), str(comp.source.iloc[0]), str(comp.method.iloc[0]))
        for target in info["targets"]:
            path = directory / f"features_{target}.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            if target in COMMON_TARGETS:
                groups.setdefault(key, []).append(path)
    expected_paths_per_model = len(COMMON_TARGETS) * len(training_seeds(config))
    if len(groups) != len(config["models"]) or any(len(paths) != expected_paths_per_model for paths in groups.values()):
        raise ValueError("The balanced two-target, five-seed design is incomplete")
    return pd.concat(comparisons, ignore_index=True), groups


def fit_association(frame: pd.DataFrame, *, include_margin: bool) -> list[dict]:
    data = frame.copy()
    predictors = list(METRICS if include_margin else METRICS[:2])
    for name in predictors:
        mean, sd = data[name].mean(), data[name].std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Constant/nonfinite geometry indicator: {name}")
        data[name + "_z"] = (data[name] - mean) / sd
    # The same accident appears in five seeds. A corpus-prefixed cluster ID
    # keeps its repeated units and seeds together.
    data["cluster"] = data.target.astype(str) + ":" + data.accident_id.astype(str)
    terms = " + ".join(name + "_z" for name in predictors)
    formula = "error ~ " + terms + " + C(true_macro) * C(target)"
    model = smf.gee(formula, groups="cluster", data=data,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
    fit = model.fit(maxiter=100)
    if not fit.converged or not np.isfinite(fit.params.to_numpy()).all():
        raise RuntimeError("Accident-clustered GEE failed to converge")
    ci = fit.conf_int()
    rows = []
    for term in predictors:
        name = term + "_z"
        rows.append({"model": "with_margin" if include_margin else "distance_entropy_only",
                     "predictor": term, "log_odds_per_sd": float(fit.params[name]),
                     "robust_se": float(fit.bse[name]), "odds_ratio_per_sd": float(np.exp(fit.params[name])),
                     "ci_low": float(np.exp(ci.loc[name, 0])),
                     "ci_high": float(np.exp(ci.loc[name, 1])),
                     "n_unit_seed_rows": len(data), "n_accident_clusters": data.cluster.nunique()})
    return rows


def text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.08, .93, title, fontsize=17, weight="bold", va="top")
    y = .87
    from textwrap import fill
    for paragraph in paragraphs:
        wrapped = fill(paragraph, width=90)
        fig.text(.08, y, wrapped, fontsize=11, va="top", linespacing=1.5)
        y -= .035 * (wrapped.count("\n") + 1) + .035
    pdf.savefig(fig)
    plt.close(fig)


def make_pdf(out: Path, comparison: pd.DataFrame, summaries: pd.DataFrame,
             coefficients: pd.DataFrame) -> None:
    n_refits = comparison[["backbone", "source", "method", "seed"]].drop_duplicates().shape[0]
    with PdfPages(out / "adapted_geometry_diagnostics.pdf") as pdf:
        text_page(pdf, "Adapted-space transfer-error diagnostics", [
            f"This campaign refits {n_refits} adapted conditions because their original checkpoints and embeddings were deleted. Each exported embedding is paired with predictions from its own refit. Original-run BA is displayed only as a reproducibility comparison.",
            "Source distance is mean cosine distance to the ten nearest source units. Label entropy is Shannon entropy (natural log) of those neighbours' roles. The role margin is distance to the nearest wrong-role source centroid minus distance to the true-role source centroid. Positive margin favours the true role.",
            "The margin uses the target true label, so it is a diagnostic quantity and cannot be computed for an unlabelled deployment item. All three indicators come from the final adapted representation for the same run. They do not identify a causal mechanism.",
        ])
        drift = comparison.groupby(["backbone", "source", "method"], as_index=False).agg(
            mean_abs_ba_drift=("delta_ba", lambda x: float(np.abs(x).mean())),
            max_abs_ba_drift=("delta_ba", lambda x: float(np.abs(x).max())),
            n_over_1pp=("abs_delta_over_1pp", "sum"))
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(drift.backbone + " / " + drift.source + " / " + drift.method, drift.mean_abs_ba_drift * 100)
        ax.axvline(1, linestyle="--", color="darkred", label="1 BA point")
        ax.set_xlabel("Mean absolute BA difference from historical run (points)")
        ax.legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=False)
        for ax, metric in zip(axes, METRICS):
            pivot = summaries.pivot_table(index="backbone", columns="error", values=metric, aggfunc="mean")
            pivot.plot.bar(ax=ax, title=metric.replace("_", " "), legend=False)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
        axes[-1].legend(["Correct", "Error"], loc="best")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        for ax, model in zip(axes, ("distance_entropy_only", "with_margin")):
            subset = coefficients[coefficients.model == model]
            for method_index, (method, color) in enumerate((
                    ("cross_entropy", "#1f77b4"), ("supcon", "#ff7f0e"), ("softtriple", "#2ca02c"))):
                for i, predictor in enumerate(METRICS):
                    part = subset[(subset.predictor == predictor) & (subset.method == method)]
                    if len(part):
                        ax.scatter([i + (method_index - 1) * .15] * len(part), part.odds_ratio_per_sd,
                                   alpha=.7, color=color, label=method if i == 0 and ax is axes[0] else None)
            ax.axhline(1, color="black", linewidth=.8)
            ax.set_xticks(range(3), ["Distance", "Entropy", "Margin"])
            ax.set_title(model.replace("_", " "))
            ax.set_ylabel("Odds ratio per 1 SD (each pipeline)")
        axes[0].legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        text_page(pdf, "Interpretation and limits", [
            "Separate accident-clustered binomial GEE models are fitted for each backbone, source and objective on the two common targets. They adjust for true role, target and their interaction. Standard errors cluster all units and seeds of the same accident. This is a marginal association model, not a mixed-effects causal model.",
            "Distance, entropy and margin are standardized within each fitted pipeline before coefficients are compared. A change in coefficients after including margin may reflect correlation between geometric indicators. Multiple exploratory comparisons are reported without a confirmatory family-wise claim.",
            "A similar coefficient across objectives would support a recurring association in these representations, not establish that source coverage determines transfer. Role prevalence, narrative structure, segmentation and reporting context remain potential explanations for corpus differences.",
            "The paired historical BA comparison is essential: these are refits using original recipes and seeds, but the deleted checkpoints cannot be reconstructed bit-for-bit. Large BA drift should limit comparisons with the earlier report.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="output/replication_recipes/adapted_geometry.yaml")
    args = parser.parse_args()
    config = load_replication_config(args.config)
    comparison, groups = load_complete(config)
    out = output_root(config).parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "historical_ba_comparison.csv", index=False)
    summaries, coefficients, seed_summaries = [], [], []
    for (backbone, source, method), paths in sorted(groups.items()):
        frame = pd.concat((pd.read_csv(path, dtype={"accident_id": str, "fact_id": str}) for path in paths), ignore_index=True)
        if frame.duplicated(["seed", "target", "accident_id", "fact_id"]).any():
            raise ValueError(f"Duplicate target unit for {backbone}/{source}/{method}")
        if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite geometry indicator")
        group = frame.groupby(["target", "true_macro", "error"], as_index=False).agg(
            n_unit_seed_rows=("error", "size"), **{name: (name, "mean") for name in METRICS})
        group.insert(0, "method", method); group.insert(0, "source", source); group.insert(0, "backbone", backbone)
        summaries.append(group)
        by_seed = frame.groupby(["target", "seed"], as_index=False).agg(
            n_units=("error", "size"), error_rate=("error", "mean"),
            **{name: (name, "mean") for name in METRICS})
        by_seed.insert(0, "method", method); by_seed.insert(0, "source", source); by_seed.insert(0, "backbone", backbone)
        seed_summaries.append(by_seed)
        for include_margin in (False, True):
            for row in fit_association(frame, include_margin=include_margin):
                coefficients.append({"backbone": backbone, "source": source, "method": method, **row})
        print(f"[association] {backbone}/{source}/{method}: {len(frame)} unit-seed rows", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    coef = pd.DataFrame(coefficients)
    summary.to_csv(out / "role_target_error_profiles.csv", index=False)
    coef.to_csv(out / "gee_coefficients.csv", index=False)
    spread = coef.groupby(["backbone", "source", "model", "predictor"], as_index=False).agg(
        min_log_odds=("log_odds_per_sd", "min"), max_log_odds=("log_odds_per_sd", "max"),
        sd_across_methods=("log_odds_per_sd", "std"), n_methods=("method", "nunique"))
    if not (spread.n_methods == 3).all():
        raise ValueError("Cannot compare objectives: an adapted method is missing")
    spread["range_across_methods"] = spread.max_log_odds - spread.min_log_odds
    spread.to_csv(out / "coefficient_spread_across_methods.csv", index=False)
    pd.concat(seed_summaries, ignore_index=True).to_csv(out / "seed_target_summary.csv", index=False)
    make_pdf(out, comparison, summary, coef)
    (out / "parameters.json").write_text(json.dumps({"n_refits": len(task_matrix(config)), "k": 10,
        "distance": "mean exact cosine distance to 10 nearest source units",
        "entropy": "Shannon entropy in nats among 10 nearest source labels",
        "margin": "nearest wrong-role centroid cosine distance minus true-role centroid cosine distance",
        "association": "binomial GEE, independence working covariance, accident-cluster robust SE",
        "targets": list(COMMON_TARGETS)}, indent=2), encoding="utf-8")
    print(f"[complete] {out / 'adapted_geometry_diagnostics.pdf'}")


if __name__ == "__main__":
    main()
