#!/usr/bin/env python3
"""One-seed exclusion-aware refinement with one disjoint source holdout.

Per target, an auxiliary SaT sees 80% of source training narratives and
predicts the remaining 20% for E2 training. A final SaT fitted on all source
training narratives supplies source-validation and held-out-target features.
The target never contributes to training or selection.
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
import rq1_e2_disjoint as disjoint
import rq1_e2_excluded as e2
import rq1_refiner as refiner
import sat_lodo_gold_v3 as sat

VERSION = "rq1_exclusion_holdout_v1"
SEED = e2.SEED


def split(bundle: Path, target: str):
    groups = x.bundle_groups(bundle, target)
    sat_rows, e2_rows = disjoint.partition(groups["train"], target, SEED)
    for row in e2_rows:
        e2.check_annotations(row)
    return groups, sat_rows, e2_rows


def split_record(bundle: Path, target: str) -> dict:
    groups, sat_rows, e2_rows = split(bundle, target)
    return {"sat_train_sha256": x.ids_hash(sat_rows),
            "refiner_train_sha256": x.ids_hash(e2_rows),
            "full_source_sha256": x.ids_hash(groups["train"]),
            "source_validation_sha256": x.ids_hash(groups["val"]),
            "target_test_sha256": x.ids_hash(groups["test"]),
            "n_sat_train": len(sat_rows), "n_refiner_train": len(e2_rows),
            "refiner_excluded_by_source": {
                corpus: sum(len(row.get("excluded_spans", [])) for row in e2_rows
                            if row["corpus"] == corpus)
                for corpus in sat.CORPORA if corpus != target}}


def binding(bundle: Path) -> dict:
    return {"version": VERSION, "seed": SEED, "refiner_fraction": disjoint.E2_FRACTION,
            "bundle_manifest_sha256": x.digest(bundle / "manifest.json"),
            "script_sha256": x.digest(Path(__file__)),
            "crossfit_code_sha256": x.digest(Path(x.__file__)),
            "disjoint_code_sha256": x.digest(Path(disjoint.__file__)),
            "excluded_code_sha256": x.digest(Path(e2.__file__)),
            "refiner_code_sha256": x.digest(Path(refiner.__file__)),
            "sat_code_sha256": x.digest(Path(sat.__file__))}


def verify(bundle: Path, root: Path) -> dict:
    x.verified_bundle(bundle)
    if not (root / "binding.json").is_file() or x.json_read(root / "binding.json") != binding(bundle):
        raise ValueError(f"Different or missing holdout provenance: {root}")
    manifest = x.json_read(root / "split_manifest.json")
    if manifest != {target: split_record(bundle, target) for target in sat.CORPORA}:
        raise ValueError("Source holdout or target tests have changed")
    return manifest


def prepare(args) -> None:
    bundle, root = args.bundle.resolve(), args.out.resolve()
    x.verified_bundle(bundle)
    wanted = binding(bundle)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "binding.json"
    if marker.is_file():
        if x.json_read(marker) != wanted:
            raise ValueError(f"Existing holdout run has different code or data: {root}")
    elif any(root.iterdir()):
        raise ValueError(f"Unbound output directory is not empty: {root}")
    else:
        x.json_write(marker, wanted)
    manifest = {target: split_record(bundle, target) for target in sat.CORPORA}
    path = root / "split_manifest.json"
    if path.is_file() and x.json_read(path) != manifest:
        raise ValueError("Stored source holdout differs")
    x.json_write(path, manifest)
    print(json.dumps({"ready": True, "targets": manifest,
                      "free_gib": round(shutil.disk_usage(root).free / 2**30, 1)},
                     indent=2), flush=True)


def smoke(args) -> None:
    import torch

    bundle, root = args.bundle.resolve(), args.out.resolve()
    verify(bundle, root)
    smoke_root = root / "smoke_source"
    smoke_root.mkdir(exist_ok=True)
    marker = smoke_root / "binding.json"
    if marker.is_file() and x.json_read(marker) != x.binding(bundle):
        raise ValueError("Existing smoke has different upstream provenance")
    x.json_write(marker, x.binding(bundle))
    x.smoke(argparse.Namespace(bundle=bundle, out=smoke_root,
                               wtpsplit_repo=args.wtpsplit_repo))
    if not x.checkpoint.complete_adapter(x.adapter_path(smoke_root / "smoke" / "trainer")):
        raise RuntimeError("SaT smoke did not yield a reloadable adapter")
    target = sat.CORPORA[0]
    groups = x.bundle_groups(bundle, target)
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    model = sat.load_pinned_sat(model_dir, tokenizer_dir,
                                x.adapter_path(smoke_root / "smoke" / "trainer"))
    for row in groups["train"]:
        if row["units"] and row.get("excluded_spans"):
            feature = refiner.extract_document(model, row["raw_text"])
            example = e2.boundary_example(row, feature, .02)
            if np.any(example.labels == 1):
                break
    else:
        raise ValueError("No candidate boundary could be supervised on an excluded-span source example")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Source-only SaT smoke passed", flush=True)


def complete(folder: Path, expected: dict) -> bool:
    path = folder / "complete.json"
    if not path.is_file():
        return False
    saved = x.json_read(path)
    if saved.get("expected") != expected:
        raise ValueError(f"Completed target has different provenance: {folder}")
    for relative, sha in saved["files"].items():
        if x.digest(folder / relative) != sha:
            raise ValueError(f"Completed result changed: {folder / relative}")
    return True


def run(args) -> None:
    import torch

    bundle, root = args.bundle.resolve(), args.out.resolve()
    manifest = verify(bundle, root)
    target = sat.CORPORA[args.target_index]
    groups, sat_rows, e2_rows = split(bundle, target)
    expected = {"target": target, **manifest[target],
                "binding_sha256": x.digest(root / "binding.json")}
    folder = root / f"target_{target}"
    if complete(folder, expected):
        print(f"Already complete: {folder}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("SaT and E2 training require a CUDA GPU")
    smoke_adapter = x.adapter_path(root / "smoke_source" / "smoke" / "trainer")
    if not x.checkpoint.complete_adapter(smoke_adapter):
        raise RuntimeError("SaT GPU smoke must pass before training")
    folder.mkdir(parents=True, exist_ok=True)
    auxiliary = x.train_m3_select_and_refit(bundle, args.wtpsplit_repo.resolve(), folder,
        sat_rows, target, "sat_auxiliary", 100 + args.target_index, smoke_adapter)
    auxiliary_db = folder / "auxiliary_features.sqlite3"
    x.add_features(bundle, Path(auxiliary["adapter_path"]), e2_rows, auxiliary_db,
                   f"Out-of-sample E2 training features {target}")
    final = x.train_m3_select_and_refit(bundle, args.wtpsplit_repo.resolve(), folder,
        groups["train"], target, "sat_final", 500 + args.target_index, smoke_adapter)
    final_db = folder / "final_features.sqlite3"
    x.add_features(bundle, Path(final["adapter_path"]),
                   groups["val"] + groups["test"], final_db,
                   f"Final SaT validation and test features {target}")
    with closing(x.feature_db(auxiliary_db)) as db:
        train = [e2.boundary_example(row, x.feature_tuple(db, row), .02)
                 for row in e2_rows]
    with closing(x.feature_db(final_db)) as db:
        baseline = [sat.legacy.spans_from_probabilities(
            row["raw_text"], x.cached_probabilities(db, row), final["selected_threshold"])
            for row in groups["test"]]
        validation = {threshold: [e2.inference_example(row, x.feature_tuple(db, row), threshold)
                                  for row in groups["val"]]
                      for threshold in x.E2_THRESHOLDS}
        choice = e2.fit_refiner(train, validation, folder / "refiner", "cuda",
                                "factual_and_excluded_boundaries")
        state = torch.load(folder / "refiner" / "best.pt", map_location="cpu",
                           weights_only=False)
        model = x.boundary_network(e2.VARIANT).to("cuda")
        model.load_state_dict(state["state_dict"])
        test_examples = [e2.inference_example(row, x.feature_tuple(db, row),
                                              choice["threshold"])
                         for row in groups["test"]]
        predictions = [x.prediction_spans(model, example, "cuda", choice["penalty"])
                       for example in test_examples]
    facts = refiner.test_metrics(groups["test"], predictions, e2.METHOD, target)
    excluded = collections.Counter()
    coverage = collections.Counter()
    for row, spans, example in zip(groups["test"], predictions, test_examples, strict=True):
        excluded.update(e2.exclusion_counts(row, spans))
        coverage.update(e2.exclusion_counts(row, [], example.ranks))
    if not excluded["excluded_ref"]:
        raise ValueError(f"No explicit excluded spans on held-out test: {target}")
    metrics = {"target": target, "method": e2.METHOD, "refiner_seed": SEED,
               "sat_auxiliary_seed": auxiliary["seed"],
               "sat_final_seed": final["seed"], "n_test": len(groups["test"]),
               "source_validation_selection": choice,
               "factual_exact_f1": facts["exact_f1"],
               "excluded_ref": excluded["excluded_ref"],
               "excluded_exact_recall": excluded["excluded_exact"] / excluded["excluded_ref"],
               "excluded_iou50_recall": excluded["excluded_iou50"] / excluded["excluded_ref"],
               "candidate_excluded_endpoint_coverage":
                   coverage["excluded_endpoint_available"] / coverage["excluded_ref"],
               "candidate_factual_endpoint_coverage":
                   coverage["fact_endpoint_available"] / sum(len(r["units"]) for r in groups["test"]),
               "target_used_for_selection": False, "boundary_only": True}
    prediction = folder / "predictions" / "contextual_refinement.jsonl"
    x.save_crossfit_prediction(prediction, groups["test"], predictions,
                               e2.METHOD, choice["threshold"], SEED)
    baseline_prediction = folder / "predictions" / "sat_final.jsonl"
    x.save_crossfit_prediction(baseline_prediction, groups["test"], baseline,
                               "SaT adapted to factual boundaries",
                               final["selected_threshold"], final["seed"])
    x.json_write(folder / "metrics.json", metrics)
    keep = [folder / "metrics.json", folder / "refiner" / "selection.json",
            folder / "refiner" / "best.pt", prediction, baseline_prediction,
            folder / "sat_auxiliary" / "m3_selection.json",
            folder / "sat_final" / "m3_selection.json"]
    x.json_write(folder / "complete.json", {"expected": expected,
        "files": {str(p.relative_to(folder)).replace("\\", "/"): x.digest(p) for p in keep}})
    print(json.dumps(metrics, indent=2), flush=True)


def summary(args) -> None:
    bundle, root = args.bundle.resolve(), args.out.resolve()
    manifest = verify(bundle, root)
    rows = []
    for target in sat.CORPORA:
        groups = x.bundle_groups(bundle, target)
        folder = root / f"target_{target}"
        expected = {"target": target, **manifest[target],
                    "binding_sha256": x.digest(root / "binding.json")}
        if not complete(folder, expected):
            raise RuntimeError(f"Incomplete target: {target}")
        scores = x.json_read(folder / "metrics.json")
        indexed = x.report.prediction_file(folder / "predictions" /
                                           "contextual_refinement.jsonl", groups["test"])
        if set(indexed) != {row["document_id"] for row in groups["test"]}:
            raise ValueError(f"Target predictions incomplete: {target}")
        rows.append({key: scores[key] for key in (
            "target", "method", "refiner_seed", "sat_auxiliary_seed", "sat_final_seed",
            "n_test", "factual_exact_f1", "excluded_ref", "excluded_exact_recall",
            "excluded_iou50_recall", "candidate_excluded_endpoint_coverage",
            "candidate_factual_endpoint_coverage")})
    out = root / "summary"
    out.mkdir(exist_ok=True)
    with (out / "per_domain.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    macro = {"method": e2.METHOD, "refiner_seed": SEED,
             "macro_factual_exact_f1": float(np.mean([r["factual_exact_f1"] for r in rows])),
             "macro_excluded_exact_recall": float(np.mean([r["excluded_exact_recall"] for r in rows])),
             "macro_excluded_iou50_recall": float(np.mean([r["excluded_iou50_recall"] for r in rows]))}
    with (out / "macro.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro))
        writer.writeheader()
        writer.writerow(macro)
    x.json_write(out / "manifest.json", {"version": VERSION, "seed": SEED,
        "bundle_manifest_sha256": x.digest(bundle / "manifest.json"),
        "per_domain_sha256": x.digest(out / "per_domain.csv"),
        "macro_sha256": x.digest(out / "macro.csv")})
    print(json.dumps({"macro": macro, "summary": str(out)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, action in (("prepare", prepare), ("smoke", smoke),
                         ("run", run), ("summary", summary)):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name in ("smoke", "run"):
            command.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "run":
            command.add_argument("--target-index", type=int, required=True,
                                 choices=range(len(sat.CORPORA)))
        command.set_defaults(func=action)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
