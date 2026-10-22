"""Select the smallest declared theme-support threshold allowing exact TDM fits.

This runs only the exact cover-group preflight, with no EM estimation. The
selected threshold and every failed attempt are recorded before model fitting.
Run on the mesocentre from text/, not on a workstation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import model as model_module
from .model import ComplexityLimitError, compile_observations, mine_dictionary, unique_baskets
from . import run as run_module
from .run import load_inputs, log, resolve, sha256


def choose_threshold(config: dict, baskets: list[tuple[int, ...]], roles: list[str],
                     thresholds: list[int]) -> tuple[int | None, list[dict]]:
    model = config["model"]
    observations = unique_baskets(baskets)
    attempts: list[dict] = []
    for threshold in thresholds:
        candidates = mine_dictionary(
            baskets, roles, int(model["max_theme_length"]), threshold
        )
        attempt: dict = {"min_observed_count": threshold,
                         "candidate_themes": len(candidates),
                         "unique_baskets": len(observations), "overlap_checks": []}
        log(f"Theme support {threshold}: {len(candidates)} candidates; exact preflight")
        feasible = True
        for k in (int(value) for value in model["overlap_values"]):
            try:
                compiled = compile_observations(
                    observations, candidates, k,
                    int(model["max_groups_per_component"]),
                    int(model["max_total_groups"]),
                    progress=lambda done, total, groups: log(
                        f"support={threshold}, K={k}: {done}/{total} baskets, "
                        f"{groups:,} exact groups"
                    ),
                )
                groups = sum(len(component.groups) for obs in compiled
                             for component in obs.components)
                attempt["overlap_checks"].append({"k": k, "status": "feasible",
                                                  "exact_groups": groups})
                del compiled
                gc.collect()
            except ComplexityLimitError as error:
                attempt["overlap_checks"].append({
                    "k": k, "status": "resource_limit", "message": str(error),
                    "basket": list(error.basket) if error.basket is not None else None,
                    "lower_bound_groups": error.lower_bound_groups,
                })
                feasible = False
                gc.collect()
                break
        attempts.append(attempt)
        if feasible:
            log(f"Selected theme support {threshold}; exact preflight passed")
            return threshold, attempts
    return None, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--thresholds", default="12,16,20,25,30,40,50,60,80")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    thresholds = [int(item) for item in args.thresholds.split(",")]
    if not thresholds or thresholds != sorted(set(thresholds)) or min(thresholds) < 1:
        raise ValueError("Thresholds must be positive, unique and increasing")
    config_path = resolve(args.config)
    source = resolve(args.source_run)
    output = resolve(args.output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["source_run"] = str(source)
    if [int(value) for value in config["model"]["overlap_values"]] != [3, 1]:
        raise ValueError("The exact preflight expects K=3 followed by K=1")
    matrix, dictionary, baskets, audit = load_inputs(config)
    if not np.isin(matrix.iloc[:, 1:].to_numpy(), [0, 1]).all():
        raise ValueError("Input matrix must be binary")
    signature = hashlib.sha256(json.dumps({
        "thresholds": thresholds,
        "config_sha256": sha256(config_path),
        "matrix_sha256": audit["matrix_sha256"],
        "dictionary_sha256": audit["dictionary_sha256"],
        "selector_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(Path(model_module.__file__)),
        "runner_sha256": sha256(Path(run_module.__file__)),
    }, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "feasibility_scan.json"
    selected_path = output / "config_selected.yaml"
    if report_path.is_file() and selected_path.is_file():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("selection_hash") == signature and prior.get("status") == "feasible":
            selected_config = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
            if selected_config["model"]["min_observed_count"] == prior["selected_min_observed_count"]:
                log(f"Reusing exact preflight: support={prior['selected_min_observed_count']}")
                return
    selected, attempts = choose_threshold(
        config, baskets, dictionary["role"].tolist(), thresholds
    )
    report = {
        "status": "feasible" if selected is not None else "no_feasible_threshold",
        "selection_hash": signature,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "smallest declared support with exact K=3 and K=1 preflights below configured group limits",
        "thresholds": thresholds, "selected_min_observed_count": selected,
        "config_sha256": sha256(config_path), "source": audit,
        "max_groups_per_component": int(config["model"]["max_groups_per_component"]),
        "max_total_groups": int(config["model"]["max_total_groups"]),
        "attempts": attempts,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if selected is None:
        raise RuntimeError(
            f"No declared theme-support threshold passed exact preflight; see {report_path}"
        )
    config["model"]["min_observed_count"] = selected
    selected_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log(f"Resolved TDM config: {selected_path}")


if __name__ == "__main__":
    main()
