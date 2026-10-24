#!/usr/bin/env python3
"""LODO M3 + contextual E2-B with disjoint source training narratives.

This campaign never uses target-domain narratives for fitting or selection.
For each target and seed it compares a full-source M3, a data-matched M3,
and that identical data-matched M3 followed by E2-B.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
from contextlib import closing
from pathlib import Path

import numpy as np

import rq1_e2_crossfit as x
import rq1_refiner as refiner
import sat_lodo_gold_v3 as sat

VERSION = "rq1_e2_disjoint_v1"
SEEDS = (13, 23, 37)
E2_FRACTION = 0.20
VARIANT = "E2-B"


def payload(bundle: Path) -> dict:
    return {"version": VERSION, "gold_manifest_sha256": x.digest(bundle / "manifest.json"),
            "code_sha256": x.digest(Path(__file__)),
            "crossfit_code_sha256": x.digest(Path(x.__file__)),
            "refiner_code_sha256": x.digest(Path(refiner.__file__)),
            "seeds": list(SEEDS), "e2_fraction": E2_FRACTION,
            "targets": list(sat.CORPORA), "variant": VARIANT}


def check_binding(bundle: Path, root: Path) -> None:
    path = root / "binding.json"
    if not path.is_file() or x.json_read(path) != payload(bundle):
        raise ValueError(f"Campaign provenance differs or is missing: {path}")


def partition(rows: list[dict], target: str, seed: int) -> tuple[list[dict], list[dict]]:
    assigned = x.hash_group_split(rows, 2, 20261024 + seed + sat.CORPORA.index(target),
                                  fraction=E2_FRACTION)
    m3 = [r for r in rows if assigned[x.key(r)] == 0]
    e2 = [r for r in rows if assigned[x.key(r)] == 1]
    if not m3 or not e2 or {x.key(r) for r in m3} & {x.key(r) for r in e2}:
        raise ValueError("Invalid disjoint source partition")
    if ({x.normalized_text_hash(r) for r in m3} &
            {x.normalized_text_hash(r) for r in e2}):
        raise ValueError("Duplicate narrative crosses the M3/E2 partition")
    if set(map(x.key, m3 + e2)) != set(map(x.key, rows)):
        raise ValueError("Missing source narrative in partition")
    for corpus in {r["corpus"] for r in rows}:
        if not any(r["corpus"] == corpus for r in m3) or not any(r["corpus"] == corpus for r in e2):
            raise ValueError(f"Empty source-domain partition: {corpus}")
    return m3, e2


def preflight(args: argparse.Namespace) -> None:
    bundle, root = args.bundle.resolve(), args.out.resolve()
    x.verified_bundle(bundle)
    desired = payload(bundle)
    root.mkdir(parents=True, exist_ok=True)
    binding = root / "binding.json"
    if binding.is_file() and x.json_read(binding) != desired:
        previous = x.json_read(binding)
        changed = {name for name in set(previous) | set(desired)
                   if previous.get(name) != desired.get(name)}
        # The initial launcher wrote JSON arrays but compared them to Python
        # tuples. Permit exactly that preflight-only campaign to resume after
        # the serialization fix; never rebind a smoke or trained model.
        if (changed != {"code_sha256"} or (root / "smoke_ok.json").exists()
                or (root / "smoke_check").exists()
                or any(root.glob("target_*"))):
            raise ValueError(f"Existing run has different data or code: {root}; changed={sorted(changed)}")
        print("Updating preflight-only binding after JSON list/tuple fix", flush=True)
    min_free = args.min_free_gib
    free = shutil.disk_usage(root).free / 2**30
    if free < min_free:
        raise OSError(f"Only {free:.1f} GiB free; {min_free:.1f} GiB required for two GPU tasks")
    manifest = {"version": VERSION, "bundle_manifest_sha256": desired["gold_manifest_sha256"],
                "seeds": {}, "test_ids_sha256": {}, "n_test": {}}
    for target in sat.CORPORA:
        groups = x.bundle_groups(bundle, target)
        manifest["test_ids_sha256"][target] = x.ids_hash(groups["test"])
        manifest["n_test"][target] = len(groups["test"])
        manifest["seeds"][target] = {}
        for seed in SEEDS:
            m3, e2 = partition(groups["train"], target, seed)
            manifest["seeds"][target][str(seed)] = {
                "m3_ids_sha256": x.ids_hash(m3), "e2_ids_sha256": x.ids_hash(e2),
                "n_m3": len(m3), "n_e2": len(e2),
                "by_corpus_m3": dict(collections.Counter(r["corpus"] for r in m3)),
                "by_corpus_e2": dict(collections.Counter(r["corpus"] for r in e2))}
    split_path = root / "split_manifest.json"
    if split_path.is_file() and x.json_read(split_path) != manifest:
        raise ValueError("Stored partition changed")
    x.json_write(split_path, manifest)
    x.json_write(binding, desired)
    print(json.dumps({"preflight_ok": True, "free_gib": round(free, 1),
                      "bundle": str(bundle), "out": str(root),
                      "test_total": sum(manifest["n_test"].values())}, indent=2), flush=True)


def smoke(args: argparse.Namespace) -> None:
    bundle, root = args.bundle.resolve(), args.out.resolve()
    check_binding(bundle, root)
    # Reuse the verified official one-step LoRA and token/character alignment smoke.
    smoke_root = root / "smoke_check"
    smoke_root.mkdir(parents=True, exist_ok=True)
    x.json_write(smoke_root / "binding.json", x.binding(bundle))
    x.smoke(argparse.Namespace(bundle=bundle, out=smoke_root,
                               wtpsplit_repo=args.wtpsplit_repo))
    if not (smoke_root / "crossfit_smoke_ok.json").is_file():
        raise RuntimeError("GPU smoke marker missing")
    x.json_write(root / "smoke_ok.json", {"bundle_manifest_sha256": x.digest(bundle / "manifest.json"),
                                          "smoke_sha256": x.digest(smoke_root / "crossfit_smoke_ok.json")})


def valid_finished(folder: Path, bundle: Path, expected: dict) -> bool:
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    saved = x.json_read(marker)
    if saved.get("expected") != expected or saved.get("bundle_manifest_sha256") != x.digest(bundle / "manifest.json"):
        raise ValueError(f"Completed rotation has changed provenance: {folder}")
    for relative, sha in saved["files"].items():
        if x.digest(folder / relative) != sha:
            raise ValueError(f"Completed result changed: {folder / relative}")
    for tag in ("m3_shared", "m3_full"):
        selection = x.json_read(folder / tag / "m3_selection.json")
        if (selection["allowed_ids_sha256"] !=
                (expected["m3_ids_sha256"] if tag == "m3_shared" else saved["full_ids_sha256"])):
            raise ValueError(f"Training IDs changed for {folder / tag}")
        adapter = folder / tag / "selected_adapter"
        if (not x.checkpoint.complete_adapter(adapter) or
                x.final_lodo.adapter_hashes(adapter) != selection["refit_adapter_sha256"]):
            raise ValueError(f"Selected adapter changed: {adapter}")
    return True


def run(args: argparse.Namespace) -> None:
    import torch
    bundle, root = args.bundle.resolve(), args.out.resolve()
    check_binding(bundle, root)
    if not (root / "smoke_ok.json").is_file():
        raise RuntimeError("GPU smoke must pass first")
    expected_smoke = x.json_read(root / "smoke_ok.json")
    if (expected_smoke["bundle_manifest_sha256"] != x.digest(bundle / "manifest.json") or
            expected_smoke["smoke_sha256"] != x.digest(root / "smoke_check" / "crossfit_smoke_ok.json")):
        raise ValueError("Smoke provenance changed")
    target = sat.CORPORA[args.task_index // len(SEEDS)]
    seed = SEEDS[args.task_index % len(SEEDS)]
    groups = x.bundle_groups(bundle, target)
    m3_rows, e2_rows = partition(groups["train"], target, seed)
    split = x.json_read(root / "split_manifest.json")
    expected = split["seeds"][target][str(seed)]
    if (expected["m3_ids_sha256"] != x.ids_hash(m3_rows) or
            expected["e2_ids_sha256"] != x.ids_hash(e2_rows) or
            split["test_ids_sha256"][target] != x.ids_hash(groups["test"])):
        raise ValueError("Frozen split or target test changed")
    folder = root / f"target_{target}" / f"seed_{seed}"
    if valid_finished(folder, bundle, expected):
        print(f"Already complete: {folder}", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    smoke_adapter = x.adapter_path(root / "smoke_check" / "smoke" / "trainer")
    if not x.checkpoint.complete_adapter(smoke_adapter):
        raise RuntimeError("Smoke reference adapter missing")
    # Same seed/configuration and checkpoint-selection rule; only the permitted
    # amount of source training data differs between the full and paired M3.
    shared = x.train_m3_select_and_refit(bundle, args.wtpsplit_repo.resolve(), folder,
                                         m3_rows, target, "m3_shared", seed, smoke_adapter)
    full = x.train_m3_select_and_refit(bundle, args.wtpsplit_repo.resolve(), folder,
                                       groups["train"], target, "m3_full", seed, smoke_adapter)
    shared_adapter = folder / "m3_shared" / "selected_adapter"
    full_adapter = folder / "m3_full" / "selected_adapter"
    shared_db_path, full_db_path = folder / "shared_features.sqlite3", folder / "full_features.sqlite3"
    shared_rows = e2_rows + groups["val"] + groups["test"]
    full_rows = groups["val"] + groups["test"]
    x.add_features(bundle, shared_adapter, shared_rows, shared_db_path,
                   f"Shared M3 features {target}/{seed}")
    x.add_features(bundle, full_adapter, full_rows, full_db_path,
                   f"Full M3 features {target}/{seed}")
    outputs = folder / "predictions"
    outputs.mkdir(exist_ok=True)
    metrics = {}
    with closing(x.feature_db(shared_db_path)) as shared_db, closing(x.feature_db(full_db_path)) as full_db:
        for label, db in (("M3_shared", shared_db), ("M3_full", full_db)):
            choice, curve = x.baseline_threshold_from_features(db, groups["val"])
            x.json_write(folder / f"{label}_validation.json", {"choice": choice, "curve": curve})
            preds = [sat.legacy.spans_from_probabilities(row["raw_text"],
                       x.cached_probabilities(db, row), choice["threshold"]) for row in groups["test"]]
            x.save_crossfit_prediction(outputs / f"{label}.jsonl", groups["test"], preds,
                                       label, choice["threshold"], seed)
            metrics[label] = refiner.test_metrics(groups["test"], preds, label, target)
        train = [x.variant_features(x.boundary_example(row, x.feature_tuple(shared_db, row), .02), VARIANT)
                 for row in e2_rows]
        validation = {t: [x.variant_features(x.boundary_example(row, x.feature_tuple(shared_db, row), t), VARIANT)
                          for row in groups["val"]] for t in x.E2_THRESHOLDS}
        e2_dir = folder / "e2_b"
        if all((e2_dir / name).is_file() for name in ("best.pt", "selection.json", "history.json")):
            choice = x.json_read(e2_dir / "selection.json")
        else:
            choice = x.fit_e2_variant(VARIANT, seed, train, validation, e2_dir, "cuda")
        state = torch.load(e2_dir / "best.pt", map_location="cpu", weights_only=False)
        model = x.boundary_network(VARIANT).to("cuda")
        model.load_state_dict(state["state_dict"])
        test_examples = [x.variant_features(x.boundary_example(row, x.feature_tuple(shared_db, row),
                          choice["threshold"]), VARIANT) for row in groups["test"]]
        predictions = [x.prediction_spans(model, example, "cuda", choice["penalty"])
                       for example in test_examples]
        x.save_crossfit_prediction(outputs / "M3_plus_E2_B.jsonl", groups["test"], predictions,
                                   "M3_plus_E2_B", choice["threshold"], seed)
        metrics["M3_plus_E2_B"] = refiner.test_metrics(groups["test"], predictions,
                                                       "M3_plus_E2_B", target)
    x.json_write(folder / "metrics.json", {"target": target, "seed": seed, "methods": metrics,
                 "reference": sat.REFERENCE, "n_test": len(groups["test"]),
                 "source_validation_used_for_selection": True,
                 "target_test_used_for_selection": False,
                 "m3_shared_adapter_sha256": shared["refit_adapter_sha256"],
                 "m3_full_adapter_sha256": full["refit_adapter_sha256"]})
    keep = [folder / "metrics.json", folder / "M3_shared_validation.json",
            folder / "M3_full_validation.json", e2_dir / "selection.json",
            outputs / "M3_shared.jsonl", outputs / "M3_full.jsonl",
            outputs / "M3_plus_E2_B.jsonl"]
    x.json_write(folder / "complete.json", {"expected": expected,
        "full_ids_sha256": x.ids_hash(groups["train"]),
        "bundle_manifest_sha256": x.digest(bundle / "manifest.json"),
        "files": {str(path.relative_to(folder)).replace("\\", "/"): x.digest(path) for path in keep}})
    print(json.dumps({"target": target, "seed": seed,
        "exact_f1": {name: item["exact_f1"] for name, item in metrics.items()}}, indent=2), flush=True)


def summary(args: argparse.Namespace) -> None:
    bundle, root = args.bundle.resolve(), args.out.resolve()
    check_binding(bundle, root)
    split = x.json_read(root / "split_manifest.json")
    records = []
    bootstraps = []
    macro_draws = collections.defaultdict(list)
    for target in sat.CORPORA:
        test_rows = x.bundle_groups(bundle, target)["test"]
        for seed in SEEDS:
            folder = root / f"target_{target}" / f"seed_{seed}"
            expected = split["seeds"][target][str(seed)]
            if not valid_finished(folder, bundle, expected):
                raise RuntimeError(f"Missing target/seed result: {target}/{seed}")
            result = x.json_read(folder / "metrics.json")
            if result["n_test"] != split["n_test"][target]:
                raise ValueError("Test set size differs from frozen manifest")
            per_document = {}
            for method in ("M3_shared", "M3_full", "M3_plus_E2_B"):
                predictions = x.report.prediction_file(
                    folder / "predictions" / f"{method}.jsonl", test_rows)
                per_document[method] = x.report.metric_rows(
                    test_rows, predictions, VERSION, target, method)
            rng = np.random.default_rng(20261024 + seed + sat.CORPORA.index(target) * 100)
            sample = rng.integers(0, len(test_rows), size=(2000, len(test_rows)))
            sampled = {}
            observed = {}
            for method, rows in per_document.items():
                if [r["document_id"] for r in rows] != [r["document_id"] for r in test_rows]:
                    raise ValueError("Paired bootstrap document order differs")
                counts = np.asarray([[r["span_tp"], r["span_pred"], r["span_ref"]]
                                     for r in rows], dtype=np.int64)
                summed = counts[sample].sum(axis=1)
                sampled[method] = np.divide(2 * summed[:, 0], summed[:, 1] + summed[:, 2],
                    out=np.zeros(2000), where=(summed[:, 1] + summed[:, 2]) != 0)
                totals = counts.sum(axis=0)
                observed[method] = 2 * totals[0] / (totals[1] + totals[2])
                if abs(observed[method] - result["methods"][method]["exact_f1"]) > 1e-9:
                    raise ValueError(f"Stored metric differs from prediction: {target}/{seed}/{method}")
            for baseline in ("M3_shared", "M3_full"):
                delta = sampled["M3_plus_E2_B"] - sampled[baseline]
                macro_draws[seed, baseline].append(delta)
                bootstraps.append({"target": target, "seed": seed,
                    "comparison": f"M3_plus_E2_B_minus_{baseline}",
                    "delta_exact_f1": observed["M3_plus_E2_B"] - observed[baseline],
                    "ci95_low": float(np.quantile(delta, .025)),
                    "ci95_high": float(np.quantile(delta, .975)), "reps": 2000})
            for method, item in result["methods"].items():
                records.append({"target": target, "seed": seed, "method": method,
                                "n_test": result["n_test"],
                                "exact_f1": item["exact_f1"],
                                "boundary_f1": item["boundary_f1"],
                                "soft_f1": item["soft_f1"],
                                "soft75_f1": item["soft75_f1"]})
    if len(records) != len(sat.CORPORA) * len(SEEDS) * 3:
        raise RuntimeError("Incomplete result matrix")
    for (seed, baseline), samples in macro_draws.items():
        if len(samples) != len(sat.CORPORA):
            raise ValueError("Incomplete macro bootstrap")
        draw = np.mean(np.stack(samples), axis=0)
        observed = np.mean([r["delta_exact_f1"] for r in bootstraps
                            if r["seed"] == seed and r["comparison"] ==
                            f"M3_plus_E2_B_minus_{baseline}"])
        bootstraps.append({"target": "macro_domains", "seed": seed,
            "comparison": f"M3_plus_E2_B_minus_{baseline}",
            "delta_exact_f1": float(observed),
            "ci95_low": float(np.quantile(draw, .025)),
            "ci95_high": float(np.quantile(draw, .975)), "reps": 2000})
    result_dir = root / "summary"
    result_dir.mkdir(exist_ok=True)
    path = result_dir / "per_target_seed.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    aggregates = []
    for method in ("M3_shared", "M3_full", "M3_plus_E2_B"):
        for target in (*sat.CORPORA, "macro_domains"):
            values = ([r for r in records if r["method"] == method and r["target"] == target]
                      if target != "macro_domains" else
                      [r for r in records if r["method"] == method])
            aggregates.append({"method": method, "target": target, "n": len(values),
                **{field: float(np.mean([r[field] for r in values]))
                   for field in ("exact_f1", "boundary_f1", "soft_f1", "soft75_f1")}})
    path = result_dir / "mean_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader(); writer.writerows(aggregates)
    gains = []
    lookup = {(r["target"], r["seed"], r["method"]): r for r in records}
    for target in sat.CORPORA:
        for seed in SEEDS:
            refined = lookup[(target, seed, "M3_plus_E2_B")]
            matched = lookup[(target, seed, "M3_shared")]
            full = lookup[(target, seed, "M3_full")]
            gains.append({"target": target, "seed": seed,
                          "exact_gain_vs_matched_m3": refined["exact_f1"] - matched["exact_f1"],
                          "exact_gain_vs_full_m3": refined["exact_f1"] - full["exact_f1"]})
    path = result_dir / "paired_gains.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gains[0]))
        writer.writeheader(); writer.writerows(gains)
    path = result_dir / "paired_bootstrap.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bootstraps[0]))
        writer.writeheader(); writer.writerows(bootstraps)
    x.json_write(result_dir / "manifest.json", {"version": VERSION,
        "bundle_manifest_sha256": x.digest(bundle / "manifest.json"),
        "test_ids_sha256": split["test_ids_sha256"],
        "per_target_seed_sha256": x.digest(result_dir / "per_target_seed.csv"),
        "mean_scores_sha256": x.digest(result_dir / "mean_scores.csv"),
        "paired_gains_sha256": x.digest(result_dir / "paired_gains.csv"),
        "paired_bootstrap_sha256": x.digest(result_dir / "paired_bootstrap.csv")})
    print(json.dumps({"summary": str(result_dir), "rows": len(records)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, action in (("preflight", preflight), ("smoke", smoke),
                         ("run", run), ("summary", summary)):
        p = commands.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "preflight":
            p.add_argument("--min-free-gib", type=float, default=40.0)
        if name in ("smoke", "run"):
            p.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "run":
            p.add_argument("--task-index", type=int, required=True,
                           choices=range(len(sat.CORPORA) * len(SEEDS)))
        p.set_defaults(func=action)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
