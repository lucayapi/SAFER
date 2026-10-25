#!/usr/bin/env python3
"""One-seed, source-only factual and excluded-span boundary refinement.

The upstream SaT features come from rq1_e2_crossfit.py: five grouped
out-of-fold source models and one final source model per target domain.
This command trains a boundary refiner only. It does not reject passages
or predict accident-process roles.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from contextlib import closing
from pathlib import Path

import numpy as np

import rq1_e2_crossfit as upstream
import rq1_refiner as refiner
import sat_lodo_gold_v3 as sat

VERSION = "rq1_fact_exclusion_boundaries_v2"
SEED = 13
VARIANT = "E2-B"  # The established left/right contextual representation.
METHOD = "SaT with contextual factual-and-exclusion boundary refinement"


def reference_spans(row: dict, field: str) -> list[tuple[int, int]]:
    raw = row["raw_text"]
    spans = [sat.trim_span(raw, int(item["start"]), int(item["end"]))
             for item in row.get(field, [])]
    if any(not 0 <= a < b <= len(raw) for a, b in spans):
        raise ValueError(f"Empty or invalid {field}: {upstream.key(row)}")
    return spans


def check_annotations(row: dict) -> None:
    labelled = sorted([(a, b, "fact") for a, b in reference_spans(row, "units")] +
                      [(a, b, "excluded") for a, b in reference_spans(row, "excluded_spans")])
    if any(left[1] > right[0] for left, right in zip(labelled, labelled[1:])):
        raise ValueError(f"Overlapping fact/exclusion reference: {upstream.key(row)}")


def boundary_labels(row: dict, ranks: list[int]) -> np.ndarray:
    """Observed cuts: reference endpoints=1, known span interiors=0, gaps=-1."""
    check_annotations(row)
    prefix, _, _ = refiner.nonspace_geometry(row["raw_text"])
    intervals = [(int(prefix[a]), int(prefix[b]))
                 for field in ("units", "excluded_spans")
                 for a, b in reference_spans(row, field)]
    if any(a >= b for a, b in intervals):
        raise ValueError(f"Empty nonspace reference: {upstream.key(row)}")
    total = int(prefix[-1])
    positive = {position for a, b in intervals for position in (a, b)
                if 0 < position < total}
    return np.asarray([1.0 if rank in positive else
                       0.0 if any(a < rank < b for a, b in intervals) else -1.0
                       for rank in ranks], dtype=np.float32)


def boundary_example(row: dict, feature, threshold: float) -> refiner.Example:
    example = upstream.variant_features(
        upstream.boundary_example(row, feature, threshold), VARIANT)
    example.labels = boundary_labels(row, example.ranks)
    return example


def inference_example(row: dict, feature, threshold: float) -> refiner.Example:
    """Build candidate features without reading factual or exclusion labels."""
    raw = row["raw_text"]
    probabilities, states, offsets = feature
    prefix, starts, ends = refiner.nonspace_geometry(raw)
    ranks = refiner.candidate_ranks(raw, probabilities, prefix, threshold)[1:-1]
    features = refiner.boundary_features(raw, probabilities, states, offsets,
                                         ranks, starts, ends, prefix)
    return refiner.Example(row, threshold, ranks, [],
                           np.ascontiguousarray(features[:, :256]),
                           np.full(len(ranks), -1.0, dtype=np.float32), [], [], "E2")


def exclusion_counts(row: dict, predicted: list[tuple[int, int]],
                     candidate_ranks: list[int] | None = None) -> dict[str, int]:
    raw = row["raw_text"]
    excluded = reference_spans(row, "excluded_spans")
    pred = sat.normalized_spans(raw, predicted)
    counts = {"excluded_ref": len(excluded),
              "excluded_exact": sum(e in pred for e in excluded),
              "excluded_iou50": sum(any(upstream.report.iou(p, e) >= 0.5 for p in pred)
                                    for e in excluded)}
    if candidate_ranks is not None:
        prefix, _, _ = refiner.nonspace_geometry(raw)
        available = set(candidate_ranks) | {0, int(prefix[-1])}
        counts["excluded_endpoint_available"] = sum(
            int(prefix[a]) in available and int(prefix[b]) in available
            for a, b in excluded)
        facts = reference_spans(row, "units")
        counts["fact_endpoint_available"] = sum(
            int(prefix[a]) in available and int(prefix[b]) in available
            for a, b in facts)
    return counts


def validation_scores(examples: list[refiner.Example],
                      predictions: list[list[tuple[int, int]]]) -> dict[str, float]:
    facts = collections.defaultdict(collections.Counter)
    excluded = collections.defaultdict(collections.Counter)
    for example, predicted in zip(examples, predictions, strict=True):
        domain = example.row["corpus"]
        facts[domain].update(sat.score(example.row, predicted))
        excluded[domain].update(exclusion_counts(example.row, predicted))
    if len(facts) != 5 or any(excluded[d]["excluded_ref"] == 0 for d in facts):
        raise ValueError("Each source validation domain needs factual and excluded spans")
    factual_f1 = float(np.mean([
        sat.prf(c["span_tp"], c["span_pred"], c["span_ref"])[2]
        for c in facts.values()]))
    exclusion_recall = float(np.mean([
        excluded[d]["excluded_exact"] / excluded[d]["excluded_ref"]
        for d in facts]))
    harmonic = (2 * factual_f1 * exclusion_recall /
                (factual_f1 + exclusion_recall) if factual_f1 + exclusion_recall else 0.0)
    return {"macro_factual_exact_f1": factual_f1,
            "macro_excluded_exact_recall": exclusion_recall,
            "selection_score": harmonic}


def select_on_sources(model, validation: dict[float, list[refiner.Example]],
                      device: str) -> tuple[dict, list[dict]]:
    details = []
    for threshold, examples in validation.items():
        scores = [refiner.predict_scores(model, example, device) for example in examples]
        for penalty in upstream.E2_PENALTIES:
            predictions = []
            for example, values in zip(examples, scores, strict=True):
                chosen = [rank for rank, score in zip(example.ranks, values, strict=True)
                          if score > penalty]
                total = int(refiner.nonspace_geometry(example.row["raw_text"])[0][-1])
                predictions.append(refiner.spans_from_ranks(
                    example.row["raw_text"], list(zip([0, *chosen], [*chosen, total]))))
            details.append({"threshold": threshold, "penalty": penalty,
                            **validation_scores(examples, predictions)})
    choice = max(details, key=lambda item: (
        item["selection_score"], item["macro_factual_exact_f1"],
        item["macro_excluded_exact_recall"],
        -abs(item["threshold"] - 0.05), -abs(item["penalty"])))
    return choice, details


def fit_refiner(train: list[refiner.Example],
                validation: dict[float, list[refiner.Example]],
                output: Path, device: str, mode: str) -> dict:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = upstream.boundary_network(VARIANT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    by_domain = collections.defaultdict(list)
    for index, example in enumerate(train):
        by_domain[example.row["corpus"]].append(index)
    if len(by_domain) != 5:
        raise ValueError("Expected five source-training domains")
    if not any(np.count_nonzero(example.labels == 1) for example in train):
        raise ValueError("Candidate grid contains no supervised positive boundary")
    best = None
    stale = 0
    history = []
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, upstream.MAX_E2_EPOCHS + 1):
        model.train()
        losses = []
        per_domain = math.ceil(len(train) / len(by_domain))
        order = np.concatenate([
            rng.choice(indexes, per_domain, replace=per_domain > len(indexes))
            for indexes in by_domain.values()])
        rng.shuffle(order)
        for start in range(0, len(order), 16):
            batch = [train[int(i)] for i in order[start:start + 16]
                     if len(train[int(i)].features)]
            if not batch:
                continue
            features = torch.from_numpy(np.concatenate(
                [item.features for item in batch]).astype(np.float32)).to(device)
            labels = torch.from_numpy(np.concatenate(
                [item.labels for item in batch])).to(device)
            known = labels >= 0
            if not known.any():
                continue
            positive = int((labels[known] == 1).sum().item())
            negative = int((labels[known] == 0).sum().item())
            if positive == 0:
                continue
            logits = model(features).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[known], labels[known],
                pos_weight=torch.tensor(min(10.0, negative / positive), device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("No labelled batch was used for refiner training")
        choice, grid = select_on_sources(model, validation, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **choice}
        history.append(record)
        print(json.dumps({"mode": mode, "source_selection": record}), flush=True)
        if best is None or choice["selection_score"] > best["selection_score"] + 1e-12:
            best = record
            stale = 0
            temporary = output / "best.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "seed": SEED, "mode": mode,
                        "epoch": epoch, "source_validation": choice}, temporary)
            temporary.replace(output / "best.pt")
            upstream.json_write(output / "validation_grid.json", grid)
        else:
            stale += 1
            if stale >= upstream.E2_PATIENCE:
                break
    upstream.json_write(output / "history.json", history)
    upstream.json_write(output / "selection.json", best)
    return best


def binding(bundle: Path, source: Path) -> dict:
    return {"version": VERSION, "seed": SEED,
            "bundle_manifest_sha256": upstream.digest(bundle / "manifest.json"),
            "upstream_binding_sha256": upstream.digest(source / "binding.json"),
            "upstream_split_sha256": upstream.digest(source / "split_manifest.json"),
            "script_sha256": upstream.digest(Path(__file__)),
            "upstream_code_sha256": upstream.digest(Path(upstream.__file__)),
            "refiner_code_sha256": upstream.digest(Path(refiner.__file__)),
            "sat_code_sha256": upstream.digest(Path(sat.__file__)),
            "candidate_thresholds": list(upstream.E2_THRESHOLDS),
            "penalties": list(upstream.E2_PENALTIES),
            "selection": "source-macro harmonic mean of factual exact F1 and excluded exact recall"}


def check_binding(bundle: Path, source: Path, output: Path) -> None:
    upstream.verify_campaign_binding(source, bundle)
    path = output / "binding.json"
    if not path.is_file() or upstream.json_read(path) != binding(bundle, source):
        raise ValueError(f"Exclusion-refiner provenance differs or is missing: {path}")


def prepare(args) -> None:
    bundle, source, output = args.bundle.resolve(), args.source.resolve(), args.out.resolve()
    upstream.verified_bundle(bundle)
    upstream.verify_campaign_binding(source, bundle)
    desired = binding(bundle, source)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "binding.json"
    if marker.is_file():
        if upstream.json_read(marker) != desired:
            raise ValueError(f"Existing run has different code or data: {output}")
    elif any(output.iterdir()):
        raise ValueError(f"Unbound run directory is not empty: {output}")
    else:
        upstream.json_write(marker, desired)
    counts = {}
    for target in sat.CORPORA:
        groups = upstream.bundle_groups(bundle, target)
        counts[target] = {}
        for part, rows in groups.items():
            for row in rows:
                check_annotations(row)
            counts[target][part] = {"narratives": len(rows),
                                    "facts": sum(len(row["units"]) for row in rows),
                                    "excluded": sum(len(row.get("excluded_spans", []))
                                                    for row in rows)}
        if any(counts[target][part]["excluded"] == 0 for part in ("train", "val", "test")):
            raise ValueError(f"No excluded spans in a required partition: {target}")
    upstream.json_write(output / "reference_counts.json", counts)
    print(json.dumps({"ready": True, "seed": SEED, "targets": counts}, indent=2), flush=True)


def smoke(args) -> None:
    """Check the GPU feature path and exclusion labels before costly folds."""
    import torch

    bundle, source, output = args.bundle.resolve(), args.source.resolve(), args.out.resolve()
    check_binding(bundle, source, output)
    marker = output / "smoke_ok.json"
    if marker.is_file():
        saved = upstream.json_read(marker)
        if (saved.get("binding_sha256") != upstream.digest(output / "binding.json") or
                saved.get("passed") is not True):
            raise ValueError("Stored exclusion smoke belongs to another campaign")
        print(f"Exclusion smoke already passed: {marker}", flush=True)
        return
    adapter = upstream.adapter_path(source / "smoke" / "trainer")
    if not upstream.checkpoint.complete_adapter(adapter):
        raise FileNotFoundError(f"Upstream GPU smoke adapter is missing: {adapter}")
    target = sat.CORPORA[0]
    groups = upstream.bundle_groups(bundle, target)
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    sample = None
    for row in (row for row in groups["train"] if row["units"] and row.get("excluded_spans")):
        feature = refiner.extract_document(model, row["raw_text"])
        supervised = boundary_example(row, feature, 0.02)
        if np.any(supervised.labels == 1):
            sample = (row, feature, supervised)
            break
    if sample is None:
        raise ValueError("No source smoke example has an available labelled boundary")
    row, feature, supervised = sample
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    label_free = {field: row[field] for field in ("corpus", "document_id", "raw_text")}
    inferred = inference_example(label_free, feature, 0.02)
    if (supervised.features.shape[1] != 256 or
            supervised.ranks != inferred.ranks or
            not np.array_equal(supervised.features, inferred.features) or
            not np.all(inferred.labels == -1) or
            not np.all(np.isin(supervised.labels, (-1, 0, 1)))):
        raise ValueError("Exclusion boundary features or labels failed GPU smoke")
    upstream.json_write(marker, {"binding_sha256": upstream.digest(output / "binding.json"),
                                 "sample_corpus": row["corpus"],
                                 "sample_document_id": row["document_id"],
                                 "candidate_cuts": len(inferred.ranks),
                                 "observed_positive_cuts": int(np.count_nonzero(supervised.labels == 1)),
                                 "passed": True})
    print(json.dumps(upstream.json_read(marker), indent=2), flush=True)


def completed(folder: Path, expected: dict) -> bool:
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    saved = upstream.json_read(marker)
    if saved.get("expected") != expected:
        raise ValueError(f"Completed target has different provenance: {folder}")
    for relative, sha in saved["files"].items():
        if upstream.digest(folder / relative) != sha:
            raise ValueError(f"Completed result changed: {folder / relative}")
    return True


def run(args) -> None:
    import torch

    bundle, source, output = args.bundle.resolve(), args.source.resolve(), args.out.resolve()
    check_binding(bundle, source, output)
    if not (output / "smoke_ok.json").is_file():
        raise RuntimeError("Exclusion boundary GPU smoke must pass first")
    target = sat.CORPORA[args.target_index]
    groups = upstream.bundle_groups(bundle, target)
    source_target = source / f"target_{target}"
    final_path = source_target / "m3_final.json"
    if not final_path.is_file():
        raise FileNotFoundError(f"Final source-only SaT model missing: {final_path}")
    final = upstream.json_read(final_path)
    expected = {"target": target, "seed": SEED,
                "bundle_manifest_sha256": upstream.digest(bundle / "manifest.json"),
                "source_final_sha256": upstream.digest(final_path),
                "heldout_ids_sha256": upstream.ids_hash(groups["test"])}
    folder = output / f"target_{target}"
    if completed(folder, expected):
        print(f"Already complete: {folder}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Boundary refiner training requires a CUDA GPU")
    folder.mkdir(parents=True, exist_ok=True)
    split = upstream.json_read(source / "split_manifest.json")["targets"][target]
    train = []
    for fold in range(5):
        fold_root = source_target / "crossfit" / f"fold_{fold+1}"
        marker = upstream.json_read(fold_root / "complete.json")
        fold_ids = {(item["corpus"], item["document_id"])
                    for item in split["fold_keys"][str(fold)]}
        heldout = [row for row in groups["train"] if upstream.key(row) in fold_ids]
        if (marker["heldout_ids_sha256"] != upstream.ids_hash(heldout) or
                marker["bundle_sha256"] != expected["bundle_manifest_sha256"]):
            raise ValueError(f"Out-of-fold features changed: {target}/{fold}")
        with closing(upstream.feature_db(fold_root / "features.sqlite3")) as db:
            train.extend(boundary_example(row, upstream.feature_tuple(db, row), 0.02)
                         for row in heldout)
    if len(train) != len(groups["train"]):
        raise ValueError("Out-of-fold training examples do not cover source training")
    final_features = source_target / "features.sqlite3"
    with closing(upstream.feature_db(final_features)) as db:
        validation = {threshold: [inference_example(row, upstream.feature_tuple(db, row), threshold)
                                  for row in groups["val"]]
                      for threshold in upstream.E2_THRESHOLDS}
        exclusion_choice = fit_refiner(train, validation, folder / "refiner",
                                       "cuda", "factual_and_excluded_boundaries")
        state = torch.load(folder / "refiner" / "best.pt",
                           map_location="cpu", weights_only=False)
        model = upstream.boundary_network(VARIANT).to("cuda")
        model.load_state_dict(state["state_dict"])
        exclusion_tests = [inference_example(row, upstream.feature_tuple(db, row),
                                             exclusion_choice["threshold"])
                           for row in groups["test"]]
        exclusion_refined = [upstream.prediction_spans(model, example, "cuda",
                                                       exclusion_choice["penalty"])
                             for example in exclusion_tests]
    fact = refiner.test_metrics(groups["test"], exclusion_refined, METHOD, target)
    excluded = collections.Counter()
    coverage = collections.Counter()
    for row, spans, example in zip(groups["test"], exclusion_refined,
                                   exclusion_tests, strict=True):
        excluded.update(exclusion_counts(row, spans))
        coverage.update(exclusion_counts(row, [], example.ranks))
    metrics = {"factual_exact_f1": fact["exact_f1"],
               "n_factual": sum(len(row["units"]) for row in groups["test"]),
               **dict(excluded),
               "excluded_exact_recall": excluded["excluded_exact"] / excluded["excluded_ref"],
               "excluded_iou50_recall": excluded["excluded_iou50"] / excluded["excluded_ref"]}
    coverage = {"excluded_both_endpoints": coverage["excluded_endpoint_available"],
                "excluded_ref": coverage["excluded_ref"],
                "factual_both_endpoints": coverage["fact_endpoint_available"],
                "factual_ref": sum(len(row["units"]) for row in groups["test"])}
    prediction = folder / "predictions" / "contextual_refinement.jsonl"
    upstream.save_crossfit_prediction(prediction, groups["test"], exclusion_refined,
                                      METHOD, exclusion_choice["threshold"], SEED)
    upstream.json_write(folder / "metrics.json", {
        "target": target, "refiner_seed": SEED,
        "sat_training_seed": final["seed"], "n_test": len(groups["test"]),
        "method": METHOD, "source_validation_selection": exclusion_choice,
        "candidate_coverage": coverage,
        "scores": metrics, "boundary_only": True,
        "target_used_for_selection": False})
    keep = [folder / "metrics.json", folder / "refiner" / "selection.json",
            folder / "refiner" / "best.pt", prediction]
    upstream.json_write(folder / "complete.json", {"expected": expected,
        "files": {str(path.relative_to(folder)).replace("\\", "/"): upstream.digest(path)
                  for path in keep}})
    print(json.dumps({"target": target, "metrics": metrics,
                      "coverage": coverage}, indent=2), flush=True)


def summary(args) -> None:
    bundle, source, output = args.bundle.resolve(), args.source.resolve(), args.out.resolve()
    check_binding(bundle, source, output)
    rows_out = []
    for target in sat.CORPORA:
        rows = upstream.bundle_groups(bundle, target)["test"]
        folder = output / f"target_{target}"
        final = source / f"target_{target}" / "m3_final.json"
        expected = {"target": target, "seed": SEED,
                    "bundle_manifest_sha256": upstream.digest(bundle / "manifest.json"),
                    "source_final_sha256": upstream.digest(final),
                    "heldout_ids_sha256": upstream.ids_hash(rows)}
        if not completed(folder, expected):
            raise RuntimeError(f"Incomplete held-out domain: {target}")
        metrics = upstream.json_read(folder / "metrics.json")
        indexed = upstream.report.prediction_file(
            folder / "predictions" / "contextual_refinement.jsonl", rows)
        if set(indexed) != {row["document_id"] for row in rows}:
            raise ValueError(f"Predictions do not cover the held-out domain: {target}")
        if metrics["method"] != METHOD:
            raise ValueError(f"Wrong refinement method: {target}")
        item = metrics["scores"]
        coverage = metrics["candidate_coverage"]
        rows_out.append({"target": target, "method": METHOD,
                         "sat_training_seed": metrics["sat_training_seed"],
                         "refiner_seed": SEED, "n_test": metrics["n_test"],
                         "factual_exact_f1": item["factual_exact_f1"],
                         "excluded_ref": item["excluded_ref"],
                         "excluded_exact_recall": item["excluded_exact_recall"],
                         "excluded_iou50_recall": item["excluded_iou50_recall"],
                         "candidate_excluded_endpoint_coverage": (
                             coverage["excluded_both_endpoints"] / coverage["excluded_ref"]),
                         "candidate_factual_endpoint_coverage": (
                             coverage["factual_both_endpoints"] / coverage["factual_ref"])})
    summary_dir = output / "summary"
    summary_dir.mkdir(exist_ok=True)
    with (summary_dir / "per_domain.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    macro = [{"method": METHOD, "refiner_seed": SEED,
              "macro_factual_exact_f1": float(np.mean([r["factual_exact_f1"] for r in rows_out])),
              "macro_excluded_exact_recall": float(np.mean([r["excluded_exact_recall"] for r in rows_out])),
              "macro_excluded_iou50_recall": float(np.mean([r["excluded_iou50_recall"] for r in rows_out]))}]
    with (summary_dir / "macro.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    upstream.json_write(summary_dir / "manifest.json", {
        "version": VERSION, "seed": SEED, "boundary_only": True,
        "bundle_manifest_sha256": upstream.digest(bundle / "manifest.json"),
        "per_domain_sha256": upstream.digest(summary_dir / "per_domain.csv"),
        "macro_sha256": upstream.digest(summary_dir / "macro.csv")})
    print(json.dumps({"per_domain": rows_out, "macro": macro,
                      "summary_dir": str(summary_dir)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, action in (("prepare", prepare), ("smoke", smoke),
                         ("run", run), ("summary", summary)):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--source", type=Path, required=True,
                             help="Completed or running rq1_e2_crossfit_v3 output root")
        command.add_argument("--out", type=Path, required=True)
        if name == "run":
            command.add_argument("--target-index", type=int, required=True,
                                 choices=range(len(sat.CORPORA)))
        command.set_defaults(func=action)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
