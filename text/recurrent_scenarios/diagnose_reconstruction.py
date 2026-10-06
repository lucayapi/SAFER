"""Audit whether archived and newly fitted UMAP partitions agree.

Run from text/. This reads the reference artifacts and three existing
replicate-0 archives; it writes only to a separate diagnostic directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from paired_recovery import _selected_parameters, match_factors
from scenario_pipeline import (
    _fit_cluster_with_embedding,
    load_bn_analysis_config,
    load_embeddings,
    load_units,
    sha256_file,
)


_FIT_CONTEXT: dict | None = None


def _fit_with_context(target: str, context: dict) -> tuple[np.ndarray, np.ndarray]:
    item = context[target]
    return _fit_cluster_with_embedding(
        (), item["embeddings"], context["parameters"],
        context["umap_seed"], context["config"],
    )


def _fit(target: str) -> tuple[np.ndarray, np.ndarray]:
    if _FIT_CONTEXT is None:
        raise RuntimeError("Fit context was not initialized")
    return _fit_with_context(target, _FIT_CONTEXT)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _versions() -> dict[str, str | None]:
    names = ("numpy", "scipy", "scikit-learn", "umap-learn", "hdbscan",
             "numba", "llvmlite", "pynndescent", "pandas")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _archive_match(run_dir: Path, directory: str, role: str) -> dict[str, float]:
    path = run_dir / directory / "replicates" / "replicate_0000.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    return {name: float(value["jaccard"]) for name, value in record["matches"].items()
            if name.startswith(role + "__")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("recurrent_scenarios/config.yaml"))
    parser.add_argument("--dataset", default="btp_carpentry_and_joinery")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="C", choices=("A0", "A1", "B", "C"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_bn_analysis_config(args.config.resolve(), args.dataset, run_dir)
    units, _ = load_units(config)
    embeddings = load_embeddings(config, units, output / "embedding_cache")
    parameters = _selected_parameters(run_dir)[args.role]
    role_mask = units["_role"].eq(args.role).to_numpy()
    full_units = units.loc[role_mask].reset_index(drop=True)
    full_embeddings = embeddings[role_mask]
    selected_dir = run_dir / "discovery" / args.role / "selected"
    assignments = pd.read_csv(selected_dir / "topic_assignments.csv", dtype={"fact_id": str})
    if not full_units["_fact_id"].reset_index(drop=True).eq(assignments["fact_id"]).all():
        raise ValueError("Current factual-unit order differs from the selected archive")
    old_labels = np.load(selected_dir / "labels.npy")
    old_coordinates = np.load(selected_dir / "umap.npy")
    if len(old_labels) != len(full_units) or len(old_coordinates) != len(full_units):
        raise ValueError("Selected partition and current corpus have different lengths")

    primary = json.loads((run_dir / "paired_recovery/replicates/replicate_0000.json").read_text())
    sampled = set(primary["accident_ids"])
    newer = json.loads((run_dir / "paired_parameter_sensitivity/fixed/replicates/replicate_0000.json").read_text())
    if primary["accident_ids"] != newer["accident_ids"]:
        raise ValueError("Archived replicate-0 accident samples differ")
    subset_mask = units["_accident_id"].isin(sampled).to_numpy() & role_mask
    subset_units = units.loc[subset_mask].reset_index(drop=True)
    subset_embeddings = embeddings[subset_mask]

    global _FIT_CONTEXT
    _FIT_CONTEXT = {
        "full": {"embeddings": full_embeddings},
        "subset": {"embeddings": subset_embeddings},
        "parameters": parameters,
        "umap_seed": int(config["validation"]["random_state"]),
        "config": config,
    }
    fits = {}
    # The paired-recovery job forks before its first UMAP fit. Keep that order:
    # forking after a serial fit can inherit initialized Numba state.
    if "fork" in multiprocessing.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            fits["full_fork"], fits["subset_fork_1"], fits["subset_fork_2"] = list(
                pool.map(_fit, ("full", "subset", "subset")))
    fits["full_serial"] = _fit("full")
    fits["subset_serial_1"] = _fit("subset")
    fits["subset_serial_2"] = _fit("subset")
    from joblib import Parallel, delayed

    fits["full_loky"], fits["subset_loky"] = Parallel(
        n_jobs=2, backend="loky", batch_size=1)(
            delayed(_fit_with_context)(target, _FIT_CONTEXT) for target in ("full", "subset"))

    # Refit HDBSCAN on saved UMAP coordinates to isolate the UMAP stage.
    import hdbscan

    hcfg = config["screening"]["hdbscan"]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(parameters["hdbscan_min_cluster_size"]),
        min_samples=int(parameters["hdbscan_min_samples"]),
        cluster_selection_method=str(parameters["hdbscan_cluster_selection_method"]),
        metric=str(hcfg.get("metric", "euclidean")),
        prediction_data=bool(hcfg.get("prediction_data", True)),
    )
    labels_on_saved_umap = clusterer.fit_predict(old_coordinates)

    role_map = json.loads((run_dir / "bn_results_exact/matrix/variable_roles.json").read_text())
    reference_names = [name for name, role in role_map.items() if role == args.role]
    reference = assignments.rename(columns={"fact_id": "_fact_id"})[["_fact_id", "topic_id"]].copy()
    reference["reference_factor"] = reference["topic_id"].map(
        lambda value: f"{args.role}__T{int(value.rsplit('_', 1)[1]) + 1:02d}"
        if isinstance(value, str) and value else "")
    reference = reference.set_index("_fact_id").loc[subset_units["_fact_id"]].reset_index()
    archives = {
        "smoke_serial": _archive_match(run_dir, "paired_recovery_single_job_smoke", args.role),
        "primary_parallel": _archive_match(run_dir, "paired_recovery", args.role),
        "new_parallel": _archive_match(run_dir, "paired_parameter_sensitivity/fixed", args.role),
    }
    fit_results = {}
    for name, (labels, coordinates) in fits.items():
        np.save(output / f"{name}_labels.npy", labels)
        np.save(output / f"{name}_umap.npy", coordinates)
        item = {
            "n_clusters": int(np.unique(labels[labels >= 0]).size),
            "labels_sha256": _digest(labels), "umap_sha256": _digest(coordinates),
        }
        if name.startswith("full_"):
            item.update({
                "ari_to_archived_labels": float(adjusted_rand_score(old_labels, labels)),
                "mean_absolute_umap_difference": float(np.mean(np.abs(coordinates - old_coordinates))),
            })
        else:
            matches = match_factors(reference, labels, args.role, reference_names)
            jaccard = {factor: float(value["jaccard"]) for factor, value in matches.items()}
            item["factor_jaccard"] = jaccard
            item["max_jaccard_difference_from_archive"] = {
                key: max(abs(jaccard[factor] - values[factor]) for factor in jaccard)
                for key, values in archives.items() if values
            }
        fit_results[name] = item

    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    audit = {
        "role": args.role, "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "git_revision": git_revision, "versions": _versions(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SLURM_CPUS_PER_TASK")},
        "source_sha256": {
            "units": sha256_file(Path(config["data"]["units_path"])),
            "embeddings": sha256_file(Path(config["data"]["embeddings_path"])),
            "selected_labels": sha256_file(selected_dir / "labels.npy"),
            "selected_umap": sha256_file(selected_dir / "umap.npy"),
        },
        "parameters": parameters, "n_full_units": len(full_units),
        "n_subset_units": len(subset_units),
        "archived_clusters": int(np.unique(old_labels[old_labels >= 0]).size),
        "hdbscan_on_saved_umap": {
            "n_clusters": int(np.unique(labels_on_saved_umap[labels_on_saved_umap >= 0]).size),
            "ari_to_archived_labels": float(adjusted_rand_score(old_labels, labels_on_saved_umap)),
            "labels_equal": bool(np.array_equal(old_labels, labels_on_saved_umap)),
        },
        "fits": fit_results,
    }
    path = output / "diagnosis.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Diagnostic written to {path}", flush=True)
    for name, item in fit_results.items():
        print(name, item["n_clusters"], item.get("max_jaccard_difference_from_archive", {}), flush=True)


if __name__ == "__main__":
    main()
