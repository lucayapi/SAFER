"""CLI for the versioned exact global-BN and scenario analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from scenario_analysis import _output_paths, run_global_bn_scenario_mining
from scenario_pipeline import (
    build_frozen_bn_inputs,
    load_bn_analysis_config,
    load_selected_configurations,
    load_units,
    write_bn_accident_inclusion_audit,
)


def _fingerprint(matrix: pd.DataFrame, config: Mapping[str, Any], selections: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(matrix.columns).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(matrix, index=False).values.tobytes())
    relevant = {
        "scenario_mining": config.get("scenario_mining", {}),
        "bayesian_networks": {
            key: value for key, value in config.get("bayesian_networks", {}).items()
            if key in {
                "d_max", "alpha", "bn_display_bootstrap_threshold", "bn_structure_bootstrap",
                "cpt_regularization_min_observed_count",
            }
        },
        "selections": dict(sorted(selections.items())),
    }
    digest.update(json.dumps(relevant, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("recurrent_scenarios/config.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--reestimate", action="store_true")
    args = parser.parse_args()

    scenario_dir = Path(__file__).resolve().parent
    run_dir = args.run_dir or scenario_dir / "runs" / "theme_discovery_audit" / args.dataset
    config = load_bn_analysis_config(args.config, args.dataset, run_dir)
    selections = load_selected_configurations(run_dir)
    units, _ = load_units(config)
    paths = _output_paths(run_dir)
    matrix, dictionary, _, roles = build_frozen_bn_inputs(units, run_dir, selections, config, paths["matrix"])
    write_bn_accident_inclusion_audit(units, matrix, roles, paths["matrix"])
    fingerprint = _fingerprint(matrix, config, selections)
    manifest_path = paths["root"] / "exact_bn_manifest.json"
    required = [
        paths["root"] / "primary_analysis_summary.json",
        paths["network"] / "bn_edges_full.csv",
        paths["network"] / "bn_bootstrap_edges.csv",
        paths["network"] / "bn_cpt_support_diagnostics.csv",
        paths["network"] / "bn_final_cpts.csv",
        paths["scenarios"] / "scenario_bn_discrepancy.csv",
        paths["scenarios"] / "scenario_bn_cpt_sensitivity.csv",
    ]
    if not args.reestimate and manifest_path.is_file() and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            print(f"[exact-bn] cache valid: {paths['root']}")
            return

    print(f"[exact-bn] START dataset={args.dataset} output={paths['root']}")
    analysis = run_global_bn_scenario_mining(
        config, run_dir, selections, units=units, matrix=matrix,
        theme_dictionary=dictionary, roles=roles,
    )
    manifest_path.write_text(json.dumps({
        "algorithm": "exact_role_constrained_global_bn_v1",
        "fingerprint": fingerprint,
        "dataset": args.dataset,
        "n_accidents": len(matrix),
        "n_factors": len(roles),
        "partition_selections": selections,
        "summary": analysis["summary"],
    }, indent=2, default=str), encoding="utf-8")
    print(f"[exact-bn] COMPLETE output={paths['root']}")


if __name__ == "__main__":
    main()
