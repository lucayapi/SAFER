"""Equal-size source-mixture learning curves on the held-out Company corpus.

Sampling and training use EPICEA only. Company labels are read to check the
fixed target and calculate post-fit metrics; they never select configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import multisource_dg as ms


def quotas(budget: int, domains: tuple[str, ...]) -> dict[str, int]:
    """Allocate a unit budget evenly, with at most one unit of difference."""
    base, remainder = divmod(budget, len(domains))
    return {domain: base + (i < remainder) for i, domain in enumerate(domains)}


def feasible_conditions(budget: int, available: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        condition: q
        for condition, domains in ms.CONDITIONS.items()
        for q in [quotas(budget, domains)]
        if all(q[d] <= available[d] for d in domains)
    }


def _target_reference(cfg: dict) -> pd.DataFrame:
    registry = ms.resolve_test_corpus(
        "nicollin", anchor=ms.ROOT, require_files=True, require_emb_csv=False
    ).data_csv
    if ms._absolute(cfg["target_file"]).resolve() != registry.resolve():
        raise ValueError("Configured target differs from the nicollin test registry")
    target = ms.load_filtered_metadata(str(ms._absolute(cfg["target_file"])), text_col="sentence")
    return ms._key_table(target)


def prepare(cfg: dict, run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    budgets = sorted({int(n) for n in cfg["budgets"]})
    if len(budgets) != len(cfg["budgets"]) or budgets[0] <= 0:
        raise ValueError("Budgets must be distinct positive integers in ascending order")
    if budgets != [int(n) for n in cfg["budgets"]]:
        raise ValueError("Budgets must be in ascending order")
    draws = int(cfg["draws"])
    seeds = [int(seed) for seed in cfg["training_seeds"]]
    if draws <= 0 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Invalid draws or training seeds")
    reference = _target_reference(cfg)
    sources = {d: ms._read_source(ms._absolute(cfg["source_files"][d]), d) for d in ms.SOURCES}
    available = {d: len(frame) for d, frame in sources.items()}
    conditions_by_budget = {n: feasible_conditions(n, available) for n in budgets}
    if any("btp" not in conditions for conditions in conditions_by_budget.values()):
        raise ValueError("BTP must be available as a comparator at every budget")
    if any(not any(len(ms.CONDITIONS[c]) > 1 for c in conditions) for conditions in conditions_by_budget.values()):
        raise ValueError("At least one multi-source condition is required at every budget")

    counts = sum((sources[d]["pred_label"].value_counts().reindex(ms.ROLES, fill_value=0)
                  for d in ms.SOURCES))
    reference_mix = counts.to_numpy(dtype=float) / float(counts.sum())
    source_quotas: dict[str, tuple[int, ...]] = {
        d: tuple(sorted({q[d] for conditions in conditions_by_budget.values()
                         for q in conditions.values() if d in q}))
        for d in ms.SOURCES
    }

    # All checks and source-only sampling precede any output write.
    sampled: dict[int, dict[str, dict[int, pd.DataFrame]]] = {}
    for draw in range(draws):
        sampled[draw] = {
            d: ms._sample_nested(
                sources[d], source_quotas[d], reference_mix,
                seed=20261015 + 1009 * draw + 37 * i,
                candidates=int(cfg["sample_candidates"]),
            )
            for i, d in enumerate(ms.SOURCES)
        }

    run_dir.mkdir(parents=True)
    (run_dir / "inputs").mkdir()
    reference.to_csv(run_dir / "target_keys.csv", index=False)
    sample_rows: list[dict] = []
    tasks: list[dict] = []
    for budget in budgets:
        for draw in range(draws):
            for condition, q in conditions_by_budget[budget].items():
                pieces = []
                for domain, requested in q.items():
                    piece = sampled[draw][domain][requested]
                    role_counts = piece["pred_label"].value_counts().reindex(ms.ROLES, fill_value=0)
                    deviation = 100.0 * float(np.max(np.abs(
                        role_counts.to_numpy(dtype=float) / len(piece) - reference_mix
                    )))
                    if deviation > float(cfg["max_role_share_deviation_pp"]):
                        raise ValueError(
                            f"Source role mix exceeds limit: N={budget}, draw={draw}, "
                            f"condition={condition}, domain={domain}, {deviation:.2f} pp"
                        )
                    sample_rows.append({
                        "budget": budget, "draw": draw, "condition": condition,
                        "domain": domain, "requested_units": requested,
                        "actual_units": len(piece), "accidents": piece["accident_id"].nunique(),
                        "max_role_share_deviation_pp": deviation,
                        **{f"n_{role}": int(role_counts[role]) for role in ms.ROLES},
                    })
                    pieces.append(piece)
                combined = pd.concat(pieces, ignore_index=True)
                if combined[["accident_id", "fact_id"]].astype(str).duplicated().any():
                    raise ValueError(f"Duplicate source keys: N={budget}, {condition}, draw={draw}")
                if abs(len(combined) - budget) > int(cfg["max_budget_deviation_units"]):
                    raise ValueError(
                        f"Unit count exceeds tolerance: N={budget}, {condition}, "
                        f"draw={draw}, actual={len(combined)}"
                    )
                source_csv = run_dir / "inputs" / f"n{budget}_{condition}_draw{draw}.csv"
                combined.to_csv(source_csv, index=False)
                for seed in seeds:
                    tasks.append({
                        "task_id": len(tasks), "arm": "learning_curve", "budget": budget,
                        "condition": condition, "draw": draw, "seed": seed,
                        "source_csv": str(source_csv),
                        "run_dir": str(run_dir / "runs" / f"n{budget}" / condition / f"draw_{draw}_seed_{seed}"),
                    })
    expected = sum(len(conditions) for conditions in conditions_by_budget.values()) * draws * len(seeds)
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} tasks, got {len(tasks)}")
    pd.DataFrame(sample_rows).to_csv(run_dir / "source_samples.csv", index=False)
    pd.DataFrame(tasks).to_csv(run_dir / "tasks.csv", index=False)
    (run_dir / "campaign_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "source_reference_mix.json").write_text(
        json.dumps(dict(zip(ms.ROLES, reference_mix.tolist())), indent=2), encoding="utf-8"
    )
    (run_dir / "design.json").write_text(json.dumps({
        "budgets": budgets, "available_source_units": available,
        "conditions_by_budget": {str(n): list(conditions) for n, conditions in conditions_by_budget.items()},
        "source_quotas": {d: list(q) for d, q in source_quotas.items()},
        "expected_fits": expected,
    }, indent=2), encoding="utf-8")
    print(f"Prepared {expected} fits; Company n={len(reference)}; output={run_dir}")


def _draw_plot(summary: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "btp": "BTP", "metallurgie": "Metallurgy", "caou": "Chemistry",
        "btp_metallurgie": "BTP + Metallurgy", "btp_caou": "BTP + Chemistry",
        "metallurgie_caou": "Metallurgy + Chemistry",
        "btp_metallurgie_caou": "All three",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition, part in summary.groupby("condition", sort=False):
        part = part.sort_values("budget")
        ax.errorbar(part["budget"], 100 * part["balanced_accuracy_mean"],
                    yerr=100 * part["balanced_accuracy_sd_draw_means"],
                    marker="o", capsize=3, label=labels[condition])
    ax.set(xlabel="Total source units", ylabel="Company balanced accuracy (%)")
    ax.set_xticks(sorted(summary["budget"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    btp = contrasts[contrasts["comparator"] == "btp"]
    for condition, part in btp.groupby("condition", sort=False):
        part = part.sort_values("budget")
        ax.plot(part["budget"], 100 * part["delta_ba"], marker="o", label=labels[condition])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Total source units", ylabel="BA difference versus BTP (points)")
    ax.set_xticks(sorted(summary["budget"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "contrasts_vs_btp.pdf")
    plt.close(fig)


def analyze(cfg: dict, run_dir: Path) -> None:
    tasks = ms._read_tasks(run_dir)
    reference = pd.read_csv(run_dir / "target_keys.csv", dtype={"accident_id": str, "fact_id": str})
    groups = np.sort(reference["accident_id"].unique())
    group_idx = pd.Categorical(reference["accident_id"], categories=groups).codes
    true_counts = np.zeros((len(groups), len(ms.ROLES)), dtype=np.float64)
    for j, role in enumerate(ms.ROLES):
        np.add.at(true_counts[:, j], group_idx,
                  (reference["true_macro"].to_numpy() == role).astype(float))
    if (true_counts.sum(axis=0) == 0).any():
        raise ValueError("A target role has zero support")
    per_run: list[dict] = []
    tp_sums: dict[tuple[int, str], np.ndarray] = {}
    tp_counts: dict[tuple[int, str], int] = {}
    for task in tasks.to_dict("records"):
        out = Path(task["run_dir"])
        if not (out / "done.json").is_file():
            raise FileNotFoundError(f"Task incomplete: {task['task_id']}")
        frame = ms._aligned_predictions(out / "predictions" / "predictions_nicollin.csv", reference)
        measured = ms._metrics(frame)
        saved = json.loads((out / "target_metrics.json").read_text(encoding="utf-8"))
        if abs(measured["balanced_accuracy"] - float(saved["balanced_accuracy"])) > 1e-10:
            raise ValueError(f"Metric mismatch: task {task['task_id']}")
        per_run.append({**{k: task[k] for k in ("task_id", "budget", "condition", "draw", "seed")},
                        **measured})
        key = (int(task["budget"]), task["condition"])
        tp = ms._run_tp_by_accident(frame, groups, len(groups))
        tp_sums[key] = tp_sums.get(key, np.zeros_like(tp)) + tp
        tp_counts[key] = tp_counts.get(key, 0) + 1

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    run_df = pd.DataFrame(per_run)
    run_df.to_csv(analysis_dir / "per_run.csv", index=False)
    summaries: list[dict] = []
    metric_names = ("balanced_accuracy", "accuracy", "macro_f1") + tuple(
        f"{role.lower()}_{name}" for role in ms.ROLES for name in ("precision", "recall")
    )
    for (budget, condition), part in run_df.groupby(["budget", "condition"], sort=True):
        row = {"budget": budget, "condition": condition,
               "n_runs": len(part), "n_draws": part["draw"].nunique()}
        for metric in metric_names:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_sd_runs"] = part[metric].std(ddof=1)
        draw_means = part.groupby("draw")["balanced_accuracy"].mean()
        row["balanced_accuracy_sd_draw_means"] = draw_means.std(ddof=1)
        seed_means = part.groupby("seed")["balanced_accuracy"].mean()
        row["balanced_accuracy_sd_seed_means"] = seed_means.std(ddof=1)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(analysis_dir / "summary.csv", index=False)

    paired_rows: list[dict] = []
    for (budget, draw, seed), part in run_df.groupby(["budget", "draw", "seed"]):
        values = part.set_index("condition")["balanced_accuracy"]
        expected = {task["condition"] for task in tasks.to_dict("records")
                    if task["budget"] == budget and task["draw"] == draw and task["seed"] == seed}
        if set(values.index) != expected:
            raise ValueError(f"Incomplete comparisons: N={budget}, draw={draw}, seed={seed}")
        for condition in values.index:
            if condition == "btp":
                continue
            domains = ms.CONDITIONS[condition]
            paired_rows.append({
                "budget": budget, "draw": draw, "seed": seed, "condition": condition,
                "delta_vs_btp": float(values[condition] - values["btp"]),
                "delta_vs_mean_constituent_singles": (
                    float(values[condition] - values[list(domains)].mean())
                    if len(domains) > 1 and set(domains).issubset(values.index) else np.nan
                ),
            })
    pd.DataFrame(paired_rows).to_csv(analysis_dir / "paired_run_contrasts.csv", index=False)

    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    n_resamples = int(cfg["bootstrap_resamples"])
    weights = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)),
                              size=n_resamples).astype(np.float32)
    denominators = weights @ true_counts.astype(np.float32)
    if (denominators == 0).any():
        raise ValueError("A bootstrap sample omitted a target role")
    boot: dict[tuple[int, str], np.ndarray] = {}
    point: dict[tuple[int, str], float] = {}
    for key, tp_sum in tp_sums.items():
        tp_mean = (tp_sum / tp_counts[key]).astype(np.float32)
        boot[key] = ((weights @ tp_mean) / denominators).mean(axis=1)
        point[key] = float((tp_mean.sum(axis=0) / true_counts.sum(axis=0)).mean())
    contrast_rows: list[dict] = []
    for budget in sorted(run_df["budget"].unique()):
        for condition, domains in ms.CONDITIONS.items():
            key = (int(budget), condition)
            if key not in boot or condition == "btp":
                continue
            comparators = [("btp", boot[(budget, "btp")], point[(budget, "btp")])]
            if len(domains) > 1 and all((budget, d) in boot for d in domains):
                comparators.append((
                    "mean_constituent_singles",
                    np.mean([boot[(budget, d)] for d in domains], axis=0),
                    float(np.mean([point[(budget, d)] for d in domains])),
                ))
            for name, other_boot, other_point in comparators:
                difference = boot[key] - other_boot
                contrast_rows.append({
                    "budget": budget, "condition": condition, "comparator": name,
                    "delta_ba": point[key] - other_point,
                    "ci_low": float(np.quantile(difference, 0.025)),
                    "ci_high": float(np.quantile(difference, 0.975)),
                    "bootstrap_resamples": n_resamples,
                })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(analysis_dir / "paired_accident_bootstrap.csv", index=False)
    samples = pd.read_csv(run_dir / "source_samples.csv")
    samples.groupby(["budget", "condition", "draw"], as_index=False).agg(
        actual_units=("actual_units", "sum"), accidents=("accidents", "sum")
    ).to_csv(analysis_dir / "sample_sizes.csv", index=False)
    _draw_plot(summary, contrasts, analysis_dir)
    (analysis_dir / "README.md").write_text(
        "# Equal-size multi-source learning curves\n\n"
        f"{len(tasks)} Qwen3 CE fits. Company: {len(reference)} fixed target units; "
        f"A1: {int((reference['true_macro'] == 'A1').sum())}.\n"
        "All model settings are fixed; Company is evaluation-only. Whole source "
        "accidents are sampled with nested source prefixes across budgets. "
        "Chemistry alone is unavailable above its corpus size.\n"
        "Means and SDs across fits, draw means, and seed means are in summary.csv. "
        "Paired run differences are in paired_run_contrasts.csv. "
        "Bootstrap CIs resample complete Company accidents conditional on the "
        "fitted models; they do not incorporate training variation. "
        "The mean-of-singletons contrast is omitted wherever a constituent "
        "single-source condition is infeasible.\n"
        "The Company target was already inspected in the earlier campaign, "
        "so these learning curves are exploratory rather than an untouched "
        "confirmatory test. Source identity, role mix, and semantic coverage "
        "remain intertwined; no causal mechanism is established.\n",
        encoding="utf-8",
    )
    print(f"Analysis complete: {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "count", "fit", "analyze"))
    parser.add_argument("--config", default="configs/multisource_learning_curves.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = ms._config(args.config) if args.phase == "prepare" else json.loads(
        (run_dir / "campaign_config.json").read_text(encoding="utf-8")
    )
    if args.phase == "prepare":
        prepare(cfg, run_dir)
    elif args.phase == "count":
        print(len(ms._read_tasks(run_dir)))
    elif args.phase == "fit":
        if args.task_index is None:
            parser.error("fit requires --task-index")
        ms.fit(cfg, run_dir, args.task_index)
    else:
        analyze(cfg, run_dir)


if __name__ == "__main__":
    main()
