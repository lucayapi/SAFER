"""Primary downstream analysis: global BN + empirical scenario mining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from global_bn import (
    assert_global_bn_outputs,
    fit_global_bn,
    run_global_bn_bootstrap,
    write_global_bn_edges,
    write_global_bn_summary,
)
from scenario_mining import (
    assert_scenario_mining_outputs,
    build_scenario_article_table,
    mine_recurrent_scenarios,
    write_scenarios_latex_csv,
)
from scenario_pipeline import (
    ROLES,
    build_frozen_bn_inputs,
    load_units,
    write_bn_accident_inclusion_audit,
)
from scenario_reporting import render_scenario_reporting


def _output_paths(run_dir: Path) -> dict[str, Path]:
    """All BN/scenario outputs under a single ``bn_results/`` tree."""
    root = run_dir / "bn_results"
    return {
        "root": root,
        "matrix": root / "matrix",
        "network": root / "network",
        "scenarios": root / "scenarios",
        "figures": root / "figures",
        "legacy": run_dir / "legacy_latent_family",
    }


def print_primary_summary(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    bn_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    edges: pd.DataFrame,
    mining: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    config: Mapping[str, Any],
) -> None:
    n_accidents = len(matrix)
    n_factors = len([column for column in matrix.columns if column != "accident_id"])
    role_counts = {role: sum(1 for node in roles if roles[node] == role) for role in ROLES}
    stable_threshold = float(
        config.get("scenario_mining", {}).get(
            "bn_display_bootstrap_threshold",
            config.get("bayesian_networks", {}).get("bn_display_bootstrap_threshold", 0.60),
        )
    )
    min_count = int(config.get("scenario_mining", {}).get("scenario_min_accident_count", 5))
    candidates = mining["candidates_all"]
    recurrent_all = mining.get("recurrent_all", pd.DataFrame())

    n_stable = 0
    n_positive_stable = 0
    n_negative_stable = 0
    if not bootstrap.empty and not edges.empty:
        stable_mask = bootstrap["selection_frequency"].ge(stable_threshold)
        stable_edges = bootstrap.loc[stable_mask, ["parent", "child"]]
        n_stable = len(stable_edges)
        edge_lookup = edges.set_index(["parent_factor", "child_factor"])
        for _, row in stable_edges.iterrows():
            key = (str(row["parent"]), str(row["child"]))
            if key not in edge_lookup.index:
                continue
            signed = float(edge_lookup.loc[key, "conditional_contrast_signed"])
            if signed > 0:
                n_positive_stable += 1
            elif signed < 0:
                n_negative_stable += 1

    n_admissible = int(mining.get("n_admissible", len(candidates)))
    n_observed = int(mining.get("n_observed", (candidates["scenario_accident_count"] > 0).sum() if not candidates.empty else 0))

    print("\nGLOBAL BN")
    print("---------")
    print(f"N accidents: {n_accidents}")
    print(f"N factors: {n_factors} ({role_counts})")
    if not bn_summary.empty:
        print(f"Learned BN edges: {int(bn_summary['n_edges'].iloc[0])}")
    print(f"Stable edges >= {stable_threshold}: {n_stable}")
    print(f"Positive stable edges: {n_positive_stable}")
    print(f"Negative stable edges: {n_negative_stable}")

    print("\nSCENARIO MINING")
    print("---------------")
    print(f"Admissible combinations: {n_admissible}")
    print(f"Observed at least once: {n_observed}")
    for threshold in config.get("scenario_mining", {}).get("threshold_sensitivity_counts", [3, 5, 8, 10]):
        count = int((candidates["scenario_accident_count"] >= threshold).sum()) if not candidates.empty else 0
        print(f"n >= {threshold}: {count}")
    print(f"Closed recurrent patterns at n>={min_count}: {len(recurrent_all)}")
    print(f"Main-text scenarios displayed: {len(mining['article'])}")

    print("\nFILES")
    print("-----")
    root = paths.get("root")
    if root is not None and root.is_dir():
        print(f"Root: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                print(path)

    article = mining.get("article")
    if article is not None and not article.empty:
        print("\nscenarios_article.csv")
        print(build_scenario_article_table(article).to_string(index=False))


def run_global_bn_scenario_mining(
    config: Mapping[str, Any],
    run_dir: Path,
    partition_selections: Mapping[str, str],
    *,
    units: pd.DataFrame | None = None,
    matrix: pd.DataFrame | None = None,
    theme_dictionary: pd.DataFrame | None = None,
    roles: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    paths = _output_paths(run_dir)
    root = paths["root"]
    matrix_dir = paths["matrix"]
    network_dir = paths["network"]
    scenario_dir = paths["scenarios"]
    figures_dir = paths["figures"]
    for directory in (root, matrix_dir, network_dir, scenario_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    excluded_themes = pd.DataFrame()
    if matrix is None or roles is None or theme_dictionary is None:
        if units is None:
            units, _ = load_units(config)
        matrix, theme_dictionary, excluded_themes, roles = build_frozen_bn_inputs(
            units, run_dir, partition_selections, config, matrix_dir,
        )
        write_bn_accident_inclusion_audit(units, matrix, roles, matrix_dir)
    else:
        excluded_path = matrix_dir / "excluded_themes.csv"
        if excluded_path.is_file():
            excluded_themes = pd.read_csv(excluded_path)

    stable_threshold = float(
        config.get("scenario_mining", {}).get(
            "bn_display_bootstrap_threshold",
            config.get("bayesian_networks", {}).get("bn_display_bootstrap_threshold", 0.60),
        )
    )

    result = fit_global_bn(matrix, roles, config)
    bn_summary = write_global_bn_summary(result, matrix, network_dir)
    bootstrap = run_global_bn_bootstrap(matrix, roles, config, result, network_dir)
    edges = write_global_bn_edges(
        result, matrix, _theme_label_map(theme_dictionary), roles, bootstrap, network_dir,
        stable_threshold=stable_threshold,
    )
    assert_global_bn_outputs(result, matrix, roles, bootstrap, edges, config)

    mining = mine_recurrent_scenarios(
        matrix, roles, result, bootstrap, theme_dictionary, config, scenario_dir,
    )
    assert_scenario_mining_outputs(
        matrix, roles, result, mining["candidates_all"], config,
        recurrent_all=mining["recurrent_all"], article=mining["article"],
    )
    write_scenarios_latex_csv(
        mining["article_full"], result, bootstrap, scenario_dir, stable_threshold=stable_threshold,
    )
    build_scenario_article_table(mining["article"]).to_csv(scenario_dir / "scenarios_article_table.csv", index=False)
    render_scenario_reporting(
        result, bootstrap, mining["candidates_all"], mining["article"],
        theme_dictionary, config, figures_dir, network_dir,
        article_full=mining["article_full"],
        recurrent_all=mining["recurrent_all"],
        n_admissible=mining["n_admissible"],
        n_observed=mining["n_observed"],
    )

    payload = {
        "n_accidents": len(matrix),
        "n_factors": len(result.nodes),
        "role_counts": {role: sum(1 for node in roles if roles[node] == role) for role in ROLES},
        "n_bn_edges": len(result.edges),
        "n_stable_edges": int((bootstrap["selection_frequency"] >= stable_threshold).sum()) if not bootstrap.empty else 0,
        "n_admissible": int(mining["n_admissible"]),
        "n_observed": int(mining["n_observed"]),
        "n_candidates": len(mining["candidates_all"]),
        "n_closed_recurrent": len(mining["recurrent_all"]),
        "n_article_scenarios": len(mining["article"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    }
    (root / "primary_analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_primary_summary(matrix, roles, bn_summary, bootstrap, edges, mining, paths, config)
    return {
        "matrix": matrix,
        "roles": roles,
        "theme_dictionary": theme_dictionary,
        "excluded_themes": excluded_themes,
        "result": result,
        "bootstrap": bootstrap,
        "edges": edges,
        "mining": mining,
        "paths": paths,
        "summary": payload,
    }


def _theme_label_map(theme_dictionary: pd.DataFrame | None) -> dict[str, str]:
    from scenario_pipeline import _theme_label_map as _map

    return _map(theme_dictionary)
