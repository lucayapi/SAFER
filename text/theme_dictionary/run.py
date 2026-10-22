"""CLI for the descriptive TDM-CC-BU-K analysis.

Run from text/: python -m theme_dictionary.run --config theme_dictionary/config.yaml
Use --k 3 or --k 1 to fit the two models in separate output directories.
No clustering, embedding or LLM calls occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model import (
    BasketEvaluator, ComplexityLimitError, Theme, compile_observations, deletion_scores, em_fit,
    evaluate_basket, initial_theta, mine_dictionary, overlap_index,
    unique_baskets,
)


TEXT_ROOT = Path(__file__).resolve().parents[1]
ROLES = {"A0", "A1", "B", "C"}


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (TEXT_ROOT / path).resolve()


def load_inputs(config: dict):
    source = resolve(config["data"]["source_run"])
    manifest_path = source / "manifest.json"
    matrix_path = source / "accident_factor_matrix.csv"
    dictionary_path = source / "factor_dictionary.csv"
    editorial_path = source / "factor_dictionary_editorial_review.csv"
    for path in (manifest_path, matrix_path, dictionary_path):
        if not path.exists():
            raise FileNotFoundError(f"Required completed factor-analysis output missing: {path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("Source factor-analysis manifest is not complete")
    matrix = pd.read_csv(matrix_path, dtype={"accident_id": str})
    dictionary = pd.read_csv(dictionary_path, dtype=str).fillna("")
    if matrix.columns[0] != "accident_id" or matrix["accident_id"].isna().any():
        raise ValueError("First matrix column must be nonmissing accident_id")
    if matrix["accident_id"].duplicated().any():
        raise ValueError("Duplicate accident_id in matrix")
    factors = list(matrix.columns[1:])
    if len(factors) != len(set(factors)):
        raise ValueError("Duplicate factor columns")
    if dictionary["factor_id"].duplicated().any() or set(factors) != set(dictionary["factor_id"]):
        raise ValueError("Factor dictionary and matrix columns differ or have duplicate IDs")
    if not set(dictionary["role"]) <= ROLES:
        raise ValueError("Invalid factor role")
    values = matrix[factors].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Accident-factor matrix must contain only 0 and 1")
    dictionary = dictionary.set_index("factor_id").loc[factors].reset_index()
    dictionary["display_label"] = dictionary["llm_label"]
    if editorial_path.exists():
        editorial = pd.read_csv(editorial_path, dtype=str).fillna("")
        if (editorial["factor_id"].duplicated().any() or
                set(editorial["factor_id"]) != set(factors)):
            raise ValueError("Editorial display labels do not align with factor dictionary")
        editorial = editorial.set_index("factor_id").loc[factors]
        dictionary["display_label"] = [
            displayed or original for displayed, original in zip(
                editorial["display_label"], dictionary["llm_label"]
            )
        ]
    roles = dictionary["role"].tolist()
    baskets = [tuple(int(item) for item in np.flatnonzero(row)) for row in values]
    audit = {
        "accidents": len(matrix), "factors": len(factors),
        "factor_roles": dict(Counter(roles)),
        "empty_baskets": sum(not basket for basket in baskets),
        "max_basket_size": max(map(len, baskets), default=0),
        "unique_baskets": len(set(baskets)),
        "source_run": str(source),
        "source_manifest_sha256": sha256(manifest_path),
        "matrix_sha256": sha256(matrix_path),
        "dictionary_sha256": sha256(dictionary_path),
        "editorial_labels_sha256": sha256(editorial_path) if editorial_path.exists() else None,
    }
    return matrix, dictionary, baskets, audit


def _initial_for_start(themes: list[Theme], n: int, state: dict, scales: list[float]) -> np.ndarray:
    if state.get("theta") is not None:
        return np.asarray(state["theta"], dtype=float)
    if state["round"] > 0 and state["start"] == 0 and state.get("warm_theta") is not None:
        return np.asarray(state["warm_theta"], dtype=float)
    return initial_theta(themes, n, scales[state["start"]])


def fit_one(
    k: int, candidates: list[Theme], observations, config: dict,
    output: Path, input_hash: str,
) -> dict:
    model_cfg = config["model"]
    n = sum(freq for _, freq in observations)
    scales = [float(value) for value in model_cfg["initial_scales"]]
    threshold = 0.5 * np.log(n)
    checkpoint = output / f"checkpoint_k{k}.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["input_hash"] != input_hash or state["k"] != k:
            raise ValueError(f"Incompatible checkpoint: {checkpoint}")
        if state.get("phase") == "complete":
            log(f"K={k}: using completed checkpoint")
            return state["result"]
    else:
        state = {
            "input_hash": input_hash, "k": k, "phase": "fitting",
            "active": list(range(len(candidates))), "round": 0, "start": 0,
            "iteration": 0, "theta": None, "last_ll": None,
            "stable": 0, "best_theta": None, "best_ll": None,
            "warm_theta": None, "history": [], "em_runs": [],
            "elapsed_seconds": 0.0,
        }
        atomic_json(checkpoint, state)
    begin = time.monotonic()
    while True:
        active = state["active"]
        themes = [candidates[idx] for idx in active]
        log(f"K={k}: compiling {len(observations)} distinct baskets with {len(themes)} themes")
        compiled = compile_observations(
            observations, themes, k,
            int(model_cfg["max_groups_per_component"]),
            int(model_cfg["max_total_groups"]),
            progress=lambda done, total, groups: log(
                f"K={k}: compiled {done}/{total} baskets, {groups:,} cover groups"
            ),
        )
        with BasketEvaluator(
            compiled, themes, k, int(model_cfg["max_groups_per_component"]),
            int(config["parallel"]["workers"]),
        ) as evaluator:
            while state["start"] < len(scales):
                start = state["start"]
                theta0 = _initial_for_start(themes, n, state, scales)
                log(f"K={k}, pruning round {state['round']}, EM start {start + 1}/{len(scales)}")

                def save_iteration(iteration, theta, ll, stable):
                    state.update(iteration=iteration, theta=theta.tolist(),
                                 last_ll=float(ll), stable=stable)
                    atomic_json(checkpoint, state)
                    if iteration == 1 or iteration % 10 == 0:
                        log(f"K={k}, round {state['round']}, start {start + 1}, iteration {iteration}, score={ll:.6f}")

                theta, ll, _, iterations = em_fit(
                    compiled, themes, theta0, k,
                    int(model_cfg["max_groups_per_component"]),
                    int(config["parallel"]["workers"]),
                    float(model_cfg["em_tolerance"]), int(model_cfg["em_max_iterations"]),
                    iteration_callback=save_iteration,
                    start_iteration=int(state["iteration"]),
                    previous_ll=float(state["last_ll"]) if state["last_ll"] is not None else -np.inf,
                    stable_iterations=int(state["stable"]), evaluator=evaluator.evaluate,
                )
                if state["best_ll"] is None or ll > state["best_ll"]:
                    state["best_ll"] = float(ll)
                    state["best_theta"] = theta.tolist()
                state.setdefault("em_runs", []).append({
                    "round": state["round"], "start": start + 1,
                    "iterations": iterations, "score": float(ll),
                    "converged": True,
                })
                state.update(start=start + 1, iteration=0, theta=None,
                             last_ll=None, stable=0)
                atomic_json(checkpoint, state)
                log(f"K={k}, start {start + 1}: converged after {iterations} iterations; score={ll:.6f}")
            best = np.asarray(state["best_theta"], dtype=float)
            ll, expected, log_no = evaluator.evaluate(best)
        psi = deletion_scores(best, log_no, n)
        remove = [position for position, value in enumerate(psi) if value < threshold]
        state["history"].append({
            "round": state["round"], "themes": len(active),
            "score": ll, "bic_threshold": float(threshold),
            "removed": len(remove),
            "removed_candidate_ids": [active[pos] for pos in remove],
        })
        log(f"K={k}, round {state['round']}: removing {len(remove)} of {len(active)} themes")
        if not remove:
            result = {
                "k": k, "active": active, "theta": best.tolist(),
                "score": ll, "expected_usage": expected.tolist(),
                "deletion_scores": [float(value) if np.isfinite(value) else None for value in psi],
                "bic_threshold": float(threshold),
                "elapsed_seconds": state["elapsed_seconds"] + time.monotonic() - begin,
                "history": state["history"], "em_runs": state.get("em_runs", []),
            }
            state.update(phase="complete", result=result, elapsed_seconds=result["elapsed_seconds"])
            atomic_json(checkpoint, state)
            return result
        keep = [pos for pos in range(len(active)) if pos not in set(remove)]
        if not keep:
            raise RuntimeError("BIC pruning removed every theme; no exact model remains")
        state.update(
            active=[active[pos] for pos in keep], round=state["round"] + 1,
            start=0, iteration=0, theta=None, last_ll=None, stable=0,
            warm_theta=best[keep].tolist(), best_theta=None, best_ll=None,
            elapsed_seconds=state["elapsed_seconds"] + time.monotonic() - begin,
        )
        atomic_json(checkpoint, state)
        begin = time.monotonic()


def save_model_outputs(
    result: dict, candidates: list[Theme], matrix: pd.DataFrame,
    dictionary: pd.DataFrame, baskets: list[tuple[int, ...]],
    observations, config: dict, output: Path,
) -> dict:
    k = result["k"]
    active = result["active"]
    themes = [candidates[idx] for idx in active]
    theta = np.asarray(result["theta"], dtype=float)
    labels = dictionary["display_label"].tolist()
    original_labels = dictionary["llm_label"].tolist()
    factors = dictionary["factor_id"].tolist()
    roles = dictionary["role"].tolist()
    n = len(matrix)
    catalog = []
    for local, (candidate_id, theme) in enumerate(zip(active, themes)):
        catalog.append({
            "theme_id": f"T{candidate_id:05d}",
            "size": len(theme.items),
            "roles": ",".join(sorted({roles[i] for i in theme.items})),
            "factor_ids": json.dumps([factors[i] for i in theme.items]),
            "factor_labels": json.dumps([labels[i] for i in theme.items], ensure_ascii=False),
            "original_llm_labels": json.dumps(
                [original_labels[i] for i in theme.items], ensure_ascii=False
            ),
            "observed_count": theme.support,
            "observed_frequency": theme.support / n,
            "theta": theta[local],
            "posterior_expected_accidents": result["expected_usage"][local],
            "deletion_score": result["deletion_scores"][local],
        })
    frame = pd.DataFrame(catalog).sort_values(
        ["size", "theta", "observed_count"], ascending=[True, False, False]
    )
    frame.to_csv(output / f"theme_catalog_k{k}.csv", index=False)
    frame[frame["size"] > 1].groupby("size", sort=True).head(5).to_csv(
        output / f"top_themes_by_size_k{k}.csv", index=False
    )
    frame[frame["roles"].str.split(",").apply(lambda r: "B" in r)].to_csv(
        output / f"themes_with_B_k{k}.csv", index=False
    )
    candidate_groups = Counter(
        (len(theme.items), ",".join(sorted({roles[i] for i in theme.items})))
        for theme in candidates
    )
    selected_groups = Counter(
        (len(theme.items), ",".join(sorted({roles[i] for i in theme.items})))
        for theme in themes
    )
    pd.DataFrame([
        {"size": size, "roles": role_group,
         "candidate_themes": candidate_groups[(size, role_group)],
         "retained_themes": selected_groups[(size, role_group)]}
        for size, role_group in sorted(candidate_groups)
    ]).to_csv(output / f"theme_counts_by_size_roles_k{k}.csv", index=False)
    unique_map = {}
    basket_fit_rows = []
    for basket, _ in observations:
        value, _, partition = evaluate_basket(
            basket, themes, theta, k, int(config["model"]["max_groups_per_component"]),
            map_partition=True,
        )
        unique_map[basket] = (value, partition)
    for basket, frequency in observations:
        value = unique_map[basket][0]
        model_expected = float(n * np.exp(value))
        basket_fit_rows.append({
            "basket_size": len(basket), "factor_ids": json.dumps([factors[i] for i in basket]),
            "observed_accidents": frequency, "empirical_frequency": frequency / n,
            "model_probability": float(np.exp(value)),
            "model_expected_accidents": model_expected,
            "observed_to_expected": frequency / model_expected if model_expected > 0 else float("inf"),
            "log_basket_score": value,
        })
    pd.DataFrame(basket_fit_rows).sort_values(
        ["observed_to_expected", "observed_accidents"], ascending=[False, False]
    ).to_csv(output / f"basket_fit_k{k}.csv", index=False)
    partitions = [unique_map[basket][1] for basket in baskets]
    partition_rows = []
    examples = []
    for accident_id, basket, partition in zip(matrix["accident_id"], baskets, partitions):
        theme_ids = [f"T{active[local]:05d}" for local in partition]
        partition_rows.append({
            "accident_id": accident_id,
            "basket_size": len(basket),
            "factor_ids": json.dumps([factors[i] for i in basket]),
            "map_theme_ids": json.dumps(theme_ids),
            "map_theme_count": len(partition),
            "log_basket_score": unique_map[basket][0],
        })
        masks = [themes[local].mask for local in partition]
        if 3 <= len(basket) <= 8 and any(a & b for a, b in combinations(masks, 2)):
            examples.append({
                "accident_id": accident_id,
                "factor_labels": [labels[i] for i in basket],
                "map_themes": [[labels[i] for i in themes[local].items] for local in partition],
            })
    pd.DataFrame(partition_rows).to_csv(output / f"accident_partitions_k{k}.csv", index=False)
    overlap_count, total, rho = overlap_index(partitions, themes)
    summary = {
        "method": f"TDM-CC-BU-{k}", "accidents": total,
        "candidate_themes": len(candidates), "selected_themes": len(themes),
        "selected_nontrivial": sum(len(theme.items) > 1 for theme in themes),
        "selected_by_size": dict(Counter(str(len(theme.items)) for theme in themes)),
        "overlapping_map_partitions": overlap_count,
        "overlap_index": rho,
        "score": result["score"],
        "score_per_accident": result["score"] / total,
        "runtime_seconds": result["elapsed_seconds"],
        "pruning_rounds": len(result["history"]),
        "examples": examples[:10],
    }
    atomic_json(output / f"summary_k{k}.json", summary)
    atomic_json(output / f"diagnostics_k{k}.json", {
        "accidents_represented": n,
        "unique_baskets_represented": len(observations),
        "empty_baskets": sum(not basket for basket in baskets),
        "all_em_starts_converged": all(run["converged"] for run in result.get("em_runs", [])),
        "em_runs": result.get("em_runs", []),
        "pruning_history": result["history"],
        "total_themes_eliminated": sum(row["removed"] for row in result["history"]),
        "observed_baskets_with_expected_count_below_one": sum(
            row["model_expected_accidents"] < 1 for row in basket_fit_rows
        ),
        "observed_baskets_with_at_least_three_accidents_and_expected_count_below_one": sum(
            row["observed_accidents"] >= 3 and row["model_expected_accidents"] < 1
            for row in basket_fit_rows
        ),
    })
    return summary


def write_results_report(output: Path, audit: dict, summaries: list[dict], comparison: pd.DataFrame) -> None:
    """Generate an article-style readout without inventing an interpretation."""
    lines = [
        "# Theme dictionary results: Building construction and masonry",
        "",
        f"The analysis covers **{audit['accidents']:,} accidents** and "
        f"**{audit['factors']} factors**. {audit['empty_baskets']} accidents have "
        "no discovered factor; they remain in the likelihood.",
        "",
        "## Methods compared",
        "",
        "| Method | Initial themes | Retained themes | Retained multi-factor themes | Runtime (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict("records"):
        lines.append(
            f"| {row['method']} | {row['candidate_themes']} | "
            f"{row['selected_themes']} | {row['selected_nontrivial']} | "
            f"{row['runtime_seconds']:.1f} |"
        )
    lines.extend([
        "", "The candidate-itemset row is a support screen, not a fitted generative model.",
        "When both K values are fitted, they share the same candidates and",
        "differ only in their permitted cover-partition overlap.", "",
    ])
    for summary in summaries:
        k = summary["method"].rsplit("-", 1)[-1]
        catalog = pd.read_csv(output / f"theme_catalog_k{k}.csv")
        counts = pd.read_csv(output / f"theme_counts_by_size_roles_k{k}.csv")
        diagnostics = json.loads((output / f"diagnostics_k{k}.json").read_text(encoding="utf-8"))
        lines.extend([
            f"## TDM-CC-BU-{k}", "",
            f"Selected **{summary['selected_nontrivial']} multi-factor themes**. "
            f"The MAP partition uses overlapping themes in "
            f"**{summary['overlapping_map_partitions']}/{summary['accidents']} accidents "
            f"({summary['overlap_index']:.1%})**. The observed-data log score per accident "
            f"is {summary['score_per_accident']:.3f}.", "",
            f"{sum(run['converged'] for run in diagnostics['em_runs'])} of "
            f"{len(diagnostics['em_runs'])} EM fits converged across "
            f"{summary['pruning_rounds']} pruning rounds; "
            f"{diagnostics['total_themes_eliminated']} candidate themes were eliminated. "
            f"Of {diagnostics['unique_baskets_represented']} distinct observed baskets, "
            f"{diagnostics['observed_baskets_with_at_least_three_accidents_and_expected_count_below_one']} "
            "occurred at least three times but had a fitted expected count below one. "
            "The full in-sample basket fit is tabulated separately.", "",
            "| Theme size | Roles | Candidates | Retained |",
            "|---:|---|---:|---:|",
        ])
        for row in counts.to_dict("records"):
            lines.append(
                f"| {row['size']} | {row['roles']} | "
                f"{row['candidate_themes']} | {row['retained_themes']} |"
            )
        lines.extend([
            "",
            "| Size | Theme | Factor labels | Observed accidents | Observed frequency | Estimated selection probability |",
            "|---:|---|---|---:|---:|---:|",
        ])
        top = catalog[catalog["size"] > 1].groupby("size", sort=True).head(5)
        for row in top.to_dict("records"):
            labels = "; ".join(json.loads(row["factor_labels"]))
            labels = labels.replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {row['size']} | {row['theme_id']} | {labels} | "
                f"{row['observed_count']} | {row['observed_frequency']:.3%} | "
                f"{row['theta']:.4f} |"
            )
        if top.empty:
            lines.append("| — | — | No multi-factor theme retained | — | — | — |")
        lines.extend(["", "### Example most-likely latent partitions", ""])
        if not summary["examples"]:
            lines.append("No overlapping partition with 3–8 observed factors was found.")
        for example in summary["examples"][:3]:
            lines.append(f"- Accident `{example['accident_id']}`: observed factors: " +
                         "; ".join(example["factor_labels"]) + ".")
            lines.append("  MAP themes: " + " + ".join(
                "{" + "; ".join(theme) + "}" for theme in example["map_themes"]
            ) + ".")
        lines.append("")
    lines.extend([
        "## Interpretation", "",
        "Observed frequency is the proportion of accidents containing every factor",
        "of a theme. Estimated selection probability is a latent model parameter;",
        "the two quantities are not interchangeable. Theme overlap is calculated",
        "from the most likely latent partition of each observed basket. All findings",
        "are descriptive associations conditional on the existing factor definitions",
        "and accident-to-factor assignments. They are not causal effects.", "",
    ])
    (output / "results_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="theme_dictionary/config.yaml")
    parser.add_argument("--source-run", default=None,
                        help="Completed accident-component run to analyse (relative to text/)")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--k", type=int, choices=(1, 3), default=None,
                        help="Fit only this overlap constraint in an independent run")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.source_run is not None:
        config["data"]["source_run"] = str(resolve(args.source_run))
    configured_k = [int(value) for value in config["model"]["overlap_values"]]
    if configured_k != [3, 1]:
        raise ValueError("The configured comparison must contain K=3 then K=1")
    overlap_values = [args.k] if args.k is not None else configured_k
    matrix, dictionary, baskets, audit = load_inputs(config)
    input_hash = hashlib.sha256(
        (yaml.safe_dump(config, sort_keys=True) + audit["matrix_sha256"] +
         audit["dictionary_sha256"] + (audit["editorial_labels_sha256"] or "") +
         sha256(Path(__file__)) + sha256(Path(__file__).with_name("model.py"))).encode()
    ).hexdigest()
    # A code/config/input revision starts a separate run, leaving earlier audits intact.
    output = resolve(args.run_dir) if args.run_dir else resolve(config["output_dir"]) / input_hash[:12]
    if args.k is not None:
        output = output / f"k{args.k}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("analysis_hash") != input_hash:
            raise ValueError("Run directory contains outputs for different input/config/code hashes")
    audit["analysis_hash"] = input_hash
    atomic_json(output / "input_audit.json", audit)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"status": "running", "analysis_hash": input_hash,
                "overlap_values": overlap_values, "source": audit,
                "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(manifest_path, manifest)
    log(f"Results directory: {output}")
    try:
        model = config["model"]
        mining_started = time.monotonic()
        candidates = mine_dictionary(
            baskets, dictionary["role"].tolist(),
            int(model["max_theme_length"]), int(model["min_observed_count"]),
        )
        mining_seconds = time.monotonic() - mining_started
        log(f"Input: {len(baskets)} accidents, {len(dictionary)} factors, {len(candidates)} initial themes")
        pd.DataFrame([
            {"theme_id": f"T{idx:05d}", "factor_ids": json.dumps([
                dictionary.iloc[item]["factor_id"] for item in theme.items
            ]), "size": len(theme.items), "observed_count": theme.support}
            for idx, theme in enumerate(candidates)
        ]).to_csv(output / "frequent_candidate_themes.csv", index=False)
        observations = unique_baskets(baskets)
        if args.preflight:
            for k in overlap_values:
                log(f"Preflight K={k}: exact cover enumeration")
                compiled = compile_observations(
                    observations, candidates, k,
                    int(model["max_groups_per_component"]),
                    int(model["max_total_groups"]),
                    progress=lambda done, total, groups: log(
                        f"K={k}: {done}/{total} baskets, {groups:,} groups"
                    ),
                )
                pd.DataFrame([
                    {"basket_size": len(obs.basket), "frequency": obs.frequency,
                     "cover_groups": sum(len(component.groups) for component in obs.components),
                     "max_component_groups": max((len(component.groups) for component in obs.components), default=0),
                     "factor_ids": json.dumps([dictionary.iloc[item]["factor_id"] for item in obs.basket])}
                    for obs in compiled
                ]).to_csv(output / f"preflight_complexity_k{k}.csv", index=False)
            manifest["status"] = "preflight_complete"
            atomic_json(manifest_path, manifest)
            return
        summaries = []
        for k in overlap_values:
            result = fit_one(k, candidates, observations, config, output, input_hash)
            summaries.append(save_model_outputs(
                result, candidates, matrix, dictionary, baskets, observations, config, output
            ))
        comparison = [
            {"method": "Candidate itemsets (all singletons; cross-role support screen)",
             "candidate_themes": len(candidates),
             "selected_themes": len(candidates), "selected_nontrivial": len(candidates) - len(dictionary),
             "runtime_seconds": mining_seconds}
        ] + [
            {key: summary.get(key) for key in (
                "method", "candidate_themes", "selected_themes", "selected_nontrivial", "runtime_seconds"
            )} for summary in summaries
        ]
        comparison_frame = pd.DataFrame(comparison)
        comparison_frame.to_csv(output / "method_comparison.csv", index=False)
        write_results_report(output, audit, summaries, comparison_frame)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        models=summaries)
        atomic_json(manifest_path, manifest)
        log(f"Complete. Results: {output}")
    except Exception as exc:
        if isinstance(exc, ComplexityLimitError):
            factor_ids = dictionary["factor_id"].tolist()
            problematic = exc.basket
            atomic_json(output / "complexity_failure.json", {
                "error": str(exc),
                "basket_factor_ids": [factor_ids[item] for item in problematic] if problematic else None,
                "accident_ids": matrix.loc[
                    [basket == problematic for basket in baskets], "accident_id"
                ].tolist() if problematic is not None else [],
                "cover_groups_lower_bound": exc.lower_bound_groups,
                "initial_candidate_count": len(candidates),
                "max_groups_per_component": int(config["model"]["max_groups_per_component"]),
            })
        manifest.update(status="failed", failure_type=type(exc).__name__, failure_message=str(exc),
                        failed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
