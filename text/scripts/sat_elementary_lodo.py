#!/usr/bin/env python3
"""One-seed LODO SaT-LoRA adaptation to annotated elementary-unit boundaries.

An elementary unit is either a factual ``unit`` or an explicitly annotated
``excluded_span``. Both become literal training chunks for the official SaT
trainer. This is boundary supervision only: it does not predict facthood or a
role, and unannotated gaps are never promoted to excluded examples.

The source-only partitions, original target narratives, backbone revisions and
LoRA hyperparameters come from the immutable narrative_v3 RQ1 bundle. This is
a new training run, not a modification of the historical M3 checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import torch

import sat_lodo_gold_v3 as base

ROOT = base.ROOT
BASE_BUNDLE = base.BUNDLE
PREPARED = ROOT / "annotation" / "lodo_sat_transfer" / "sat12l_sm_elementary_v1"
VERSION = "sat_elementary_lodo_v1"
METHOD = "SaT+LoRA elementary-unit boundaries"
COUNT_FIELDS = ("fact_tp", "fact_pred", "fact_ref", "excluded_tp", "excluded_ref",
                "elementary_tp", "elementary_pred", "elementary_ref")


def code_digest() -> str:
    """Bind the code across Windows and Linux newline conventions."""
    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def elementary_chunks(row: dict) -> tuple[list[str] | None, str, int]:
    """Use ordered, literal fact and exclusion intervals; omit unknown gaps."""
    base.validate_gold(row)
    raw = row["raw_text"]
    intervals = sorted([(item["start"], item["end"])
                        for field in ("units", "excluded_spans")
                        for item in row[field]])
    if not intervals:
        return None, "no_annotated_units", 0
    chunks = [raw[a:b] for a, b in intervals]
    if any("\n" in chunk or "\r" in chunk for chunk in chunks):
        return None, "newline_inside_elementary_unit", 0
    if any(not chunk or not chunk.strip() for chunk in chunks):
        raise ValueError(f"Empty elementary unit: {row['corpus']}/{row['document_id']}")
    return chunks, "literal_elementary_units", len(raw) - sum(map(len, chunks))


def prepared_signature(bundle: Path) -> dict:
    return {"version": VERSION, "method": METHOD,
            "base_bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "gold_snapshot_sha256": base.check_bundle(bundle)["gold_snapshot_sha256"],
            "script_sha256": code_digest(),
            "template_sha256": base.digest(base.legacy.TEMPLATE),
            "fold_order": list(base.CORPORA), "seed": base.SEED,
            "training_unit": "sorted literal units + explicitly annotated excluded_spans",
            "selection": "source-domain macro exact F1 of all annotated elementary units"}


def prepare(args: argparse.Namespace) -> None:
    bundle, output = args.bundle.resolve(), args.out.resolve()
    base.check_bundle(bundle)
    if output.exists():
        raise FileExistsError(f"Prepared folder exists: {output}")
    staging = output.with_name(output.name + ".preparing")
    if staging.exists():
        raise FileExistsError(f"Incomplete preparation: {staging}")
    staging.mkdir(parents=True)
    files, report = {}, {}
    for target in base.CORPORA:
        folder = bundle / "folds" / f"target_{target}"
        examples, counts = {}, {}
        for group_name in ("train", "val"):
            rows = base.read_jsonl(folder / f"{group_name}.jsonl")
            chunks_list, skipped, omitted = [], collections.Counter(), 0
            n_facts = n_excluded = included_excluded = 0
            for row in rows:
                chunks, reason, n = elementary_chunks(row)
                if chunks is None:
                    skipped[reason] += 1
                    continue
                chunks_list.append(chunks)
                omitted += n
                n_facts += len(row["units"])
                n_excluded += len(row["excluded_spans"])
                included_excluded += bool(row["excluded_spans"])
            if not chunks_list or n_excluded == 0:
                raise ValueError(f"No trainable elementary/excluded units: {target}/{group_name}")
            examples[group_name] = chunks_list
            counts[group_name] = {"source_documents": len(rows),
                                  "trainer_documents": len(chunks_list),
                                  "facts": n_facts, "excluded": n_excluded,
                                  "documents_with_exclusions": included_excluded,
                                  "omitted_gap_characters": omitted,
                                  "skipped": dict(skipped)}
        data = {"fr": {"sentence": {base.DATASET_NAME: {
            "meta": {"train_data": examples["train"]}, "data": examples["val"]}}}}
        path = staging / "folds" / f"target_{target}" / "training.pth"
        path.parent.mkdir(parents=True)
        torch.save(data, path)
        files[path.relative_to(staging).as_posix()] = base.digest(path)
        report[target] = counts
        print(f"{target}: train={counts['train']['trainer_documents']} docs, "
              f"{counts['train']['excluded']} exclusions; "
              f"val={counts['val']['trainer_documents']} docs, "
              f"{counts['val']['excluded']} exclusions", flush=True)
    manifest = {**prepared_signature(bundle), "files": files, "source_report": report,
                "created_at_utc": base.stamp()}
    base.write_json(staging / "manifest.json", manifest)
    base.write_json(staging / "complete.json",
                    {"manifest_sha256": base.digest(staging / "manifest.json")})
    check_prepared(bundle, staging)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    print(json.dumps({"prepared": str(output), "targets": len(base.CORPORA)}))


def check_prepared(bundle: Path, prepared: Path) -> dict:
    base.check_bundle(bundle)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((prepared / "complete.json").read_text(encoding="utf-8"))
    if (complete.get("manifest_sha256") != base.digest(prepared / "manifest.json")
            or any(manifest.get(key) != value for key, value in
                   prepared_signature(bundle).items())):
        raise ValueError("Elementary bundle has different code, labels or RQ1 source")
    if set(manifest["files"]) != {
            f"folds/target_{target}/training.pth" for target in base.CORPORA}:
        raise ValueError("Elementary bundle does not contain exactly six fold datasets")
    for relative, expected in manifest["files"].items():
        path = (prepared / relative).resolve()
        if (not path.is_relative_to(prepared.resolve()) or not path.is_file()
                or base.digest(path) != expected):
            raise ValueError(f"Changed elementary training dataset: {relative}")
    return manifest


def binding(bundle: Path, prepared: Path) -> dict:
    return {"version": VERSION,
            "base_bundle_manifest_sha256": base.digest(bundle / "manifest.json"),
            "elementary_manifest_sha256": base.digest(prepared / "manifest.json"),
            "script_sha256": code_digest(),
            "template_sha256": base.digest(base.legacy.TEMPLATE)}


def bind_run(path: Path, expected: dict) -> None:
    marker = path / "binding.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Different model/data already bound to {path}")
    elif path.exists() and any(path.iterdir()):
        raise ValueError(f"Unbound nonempty output folder: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        base.write_json(marker, expected)


def verify_smoke(root: Path, expected: dict) -> None:
    marker = root / "smoke_ok.json"
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8"))["binding"] != expected:
        raise RuntimeError("Run the elementary SaT GPU smoke first")


def smoke(args: argparse.Namespace) -> None:
    bundle, prepared, root = args.bundle.resolve(), args.prepared.resolve(), args.run_root.resolve()
    check_prepared(bundle, prepared)
    expected = binding(bundle, prepared)
    if (root / "smoke_ok.json").is_file():
        verify_smoke(root, expected)
        print(f"Smoke already passed: {root / 'smoke_ok.json'}")
        return
    repo = args.wtpsplit_repo.resolve()
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    fold = base.CORPORA[0]
    source = torch.load(prepared / "folds" / f"target_{fold}" / "training.pth",
                        map_location="cpu", weights_only=False)["fr"]["sentence"][base.DATASET_NAME]
    work = root / "smoke"
    bind_run(work, expected)
    sample = {"fr": {"sentence": {base.DATASET_NAME: {
        "meta": {"train_data": source["meta"]["train_data"][:8]},
        "data": source["data"][:4]}}}}
    path = work / "training.pth"
    torch.save(sample, path)
    config = base.trainer_config(bundle, fold, work / "trainer", model_dir, tokenizer_dir)
    config.update({"text_path": str(path), "num_train_epochs": 1, "max_steps": 1,
                   "save_steps": 1, "per_device_train_batch_size": 2,
                   "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 1})
    base.legacy.run_trainer(repo, config, work / "config.json")
    adapter = base.legacy.adapter_folder(work / "trainer")
    if not base.legacy.adapter_exists(adapter):
        raise RuntimeError("Smoke did not save a loadable LoRA adapter")
    model = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    raw = base.read_jsonl(bundle / "folds" / f"target_{fold}" / "val.jsonl")[0]["raw_text"]
    base.legacy.pieces_to_spans(raw, model.split(raw, threshold=base.DEFAULT))
    base.write_json(root / "smoke_ok.json", {"binding": expected,
                    "adapter": str(adapter), "completed_at_utc": base.stamp()})
    print(json.dumps({"smoke_ok": True, "adapter": str(adapter)}))


def score(row: dict, predicted: list[tuple[int, int]]) -> dict[str, int]:
    raw = row["raw_text"]
    fact = base.score(row, predicted)
    predictions = base.normalized_spans(raw, predicted)
    excluded = base.normalized_spans(raw, [(item["start"], item["end"])
                                           for item in row["excluded_spans"]])
    e_tp = len(predictions & excluded)
    return {"fact_tp": fact["span_tp"], "fact_pred": fact["span_pred"],
            "fact_ref": fact["span_ref"], "excluded_tp": e_tp,
            "excluded_ref": len(excluded),
            "elementary_tp": fact["span_tp"] + e_tp,
            "elementary_pred": len(predictions),
            "elementary_ref": fact["span_ref"] + len(excluded)}


def rates(counts: dict[str, int]) -> dict[str, float]:
    factual_recall = (counts["fact_tp"] / counts["fact_ref"]
                      if counts["fact_ref"] else 0.0)
    e_recall = (counts["excluded_tp"] / counts["excluded_ref"]
                if counts["excluded_ref"] else 0.0)
    elementary_f1 = base.prf(counts["elementary_tp"], counts["elementary_pred"],
                             counts["elementary_ref"])[2]
    return {"factual_exact_recall": factual_recall,
            "excluded_exact_recall": e_recall,
            "elementary_exact_f1": elementary_f1}


def select_threshold(val: list[dict], db: sqlite3.Connection) -> tuple[float, list[dict]]:
    sources = sorted({row["corpus"] for row in val})
    if len(sources) != 5:
        raise ValueError("Exactly five source validation domains are required")
    counts = collections.defaultdict(collections.Counter)
    for row in val:
        probs = base.get_probs(db, row)
        for threshold in base.THRESHOLDS:
            pred = base.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            counts[row["corpus"], threshold].update(score(row, pred))
    curve = []
    for threshold in base.THRESHOLDS:
        values = [rates(counts[source, threshold]) for source in sources]
        if any(counts[source, threshold]["excluded_ref"] == 0 for source in sources):
            raise ValueError("A source validation domain has no excluded spans")
        fact = sum(v["factual_exact_recall"] for v in values) / len(values)
        excluded = sum(v["excluded_exact_recall"] for v in values) / len(values)
        elementary = sum(v["elementary_exact_f1"] for v in values) / len(values)
        curve.append({"threshold": threshold, "macro_factual_exact_recall": fact,
                      "macro_excluded_exact_recall": excluded,
                      "macro_elementary_exact_f1": elementary})
    best = min(curve, key=lambda r: (-r["macro_elementary_exact_f1"],
                                      abs(r["threshold"] - base.DEFAULT), r["threshold"]))
    for row in curve:
        row["selected"] = row is best
    return best["threshold"], curve


def evaluate(bundle: Path, target: str, run: Path, adapter: Path,
             model_dir: Path, tokenizer_dir: Path) -> None:
    folder = bundle / "folds" / f"target_{target}"
    val = base.read_jsonl(folder / "val.jsonl")
    test = base.read_jsonl(folder / "test.jsonl")
    model = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    with closing(base._probability_connection(run / "adapted_probabilities.sqlite3", create=True)) as db:
        base.infer_adapted(model, db, val + test)
        selected, curve = select_threshold(val, db)
        base.verify_native(model, db, val + test, selected)
        base.write_csv(run / "source_validation_thresholds.csv", curve,
                       ("threshold", "macro_elementary_exact_f1",
                        "macro_factual_exact_recall", "macro_excluded_exact_recall",
                        "selected"))
        total = collections.Counter()
        prediction_path = run / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for row in test:
                pred = base.legacy.spans_from_probabilities(
                    row["raw_text"], base.get_probs(db, row), selected)
                total.update(score(row, pred))
                stream.write(json.dumps({"corpus": target,
                    "document_id": row["document_id"], "threshold": selected,
                    "segments": [{"start": a, "end": b} for a, b in pred]},
                    ensure_ascii=False) + "\n")
    base.write_json(run / "metrics.json", {"version": VERSION, "method": METHOD,
                    "target": target, "n_test": len(test), "seed": base.SEED,
                    "threshold": selected, "selection": "source validation only",
                    "counts": dict(total), "metrics": rates(total),
                    "completed_at_utc": base.stamp()})
    print(json.dumps({"target": target, "threshold": selected,
                      **rates(total)}, indent=2), flush=True)


def fit(args: argparse.Namespace) -> None:
    bundle, prepared, root = args.bundle.resolve(), args.prepared.resolve(), args.run_root.resolve()
    check_prepared(bundle, prepared)
    expected = binding(bundle, prepared)
    verify_smoke(root, expected)
    if not 0 <= args.task_index < len(base.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = base.CORPORA[args.task_index]
    repo = args.wtpsplit_repo.resolve()
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    run = root / f"target_{target}"
    bind_run(run, expected)
    config = base.trainer_config(bundle, target, run / "trainer", model_dir, tokenizer_dir)
    config["text_path"] = str(prepared / "folds" / f"target_{target}" / "training.pth")
    adapter = base.legacy.adapter_folder(run / "trainer")
    if not base.legacy.adapter_exists(adapter):
        checkpoints = sorted((run / "trainer").glob("checkpoint-*"),
                             key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        if checkpoints:
            config["resume_from_checkpoint"] = str(checkpoints[-1])
        base.legacy.run_trainer(repo, config, run / "config.json")
        if not base.legacy.adapter_exists(adapter):
            raise RuntimeError(f"LoRA adapter was not saved: {adapter}")
    state = run / "trainer" / "trainer_state.json"
    if state.is_file():
        base.legacy.export_loss_history(state, run / "loss_history.csv")
    base.write_json(run / "training_done.json", {"target": target,
                    "binding": expected, "adapter": str(adapter),
                    "config_sha256": base.digest(run / "config.json"),
                    "completed_at_utc": base.stamp()})
    if (run / "metrics.json").is_file() and not args.force_evaluate:
        print(f"Already evaluated: {run / 'metrics.json'}")
        return
    evaluate(bundle, target, run, adapter, model_dir, tokenizer_dir)


def summarize(args: argparse.Namespace) -> None:
    bundle, prepared, root = args.bundle.resolve(), args.prepared.resolve(), args.run_root.resolve()
    check_prepared(bundle, prepared)
    base_manifest = base.check_bundle(bundle)
    expected = binding(bundle, prepared)
    rows = []
    for target in base.CORPORA:
        run = root / f"target_{target}"
        if json.loads((run / "binding.json").read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Incompatible run: {target}")
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if result["version"] != VERSION or result["target"] != target:
            raise ValueError(f"Incompatible metrics: {target}")
        if result["n_test"] != base_manifest["source_report"][target]["gold_documents"]:
            raise ValueError(f"Incomplete target evaluation: {target}")
        rows.append({"target": target, "n_test": result["n_test"],
                     "threshold": result["threshold"], **result["metrics"],
                     **result["counts"]})
    metrics = ("elementary_exact_f1", "factual_exact_recall", "excluded_exact_recall")
    macro = {key: sum(row[key] for row in rows) / len(rows) for key in metrics}
    rows.append({"target": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in rows),
                 "threshold": "by_target_source_validation", **macro,
                 **{key: sum(r.get(key, 0) for r in rows) for key in COUNT_FIELDS}})
    base.write_csv(root / "summary.csv", rows,
                   ("target", "n_test", "threshold", *metrics, *COUNT_FIELDS))
    base.write_json(root / "summary.json", {"version": VERSION, "method": METHOD,
                    "binding": expected, "seed": base.SEED,
                    "results": rows, "completed_at_utc": base.stamp()})
    print(json.dumps({"summary": str(root / "summary.csv"), "macro": macro}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("check", None), ("smoke", smoke),
                           ("fit", fit), ("summarize", summarize)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bundle", type=Path, default=BASE_BUNDLE)
        if name == "prepare":
            cmd.add_argument("--out", type=Path, default=PREPARED)
        else:
            cmd.add_argument("--prepared", type=Path, default=PREPARED)
        if name in {"smoke", "fit", "summarize"}:
            cmd.add_argument("--run-root", type=Path, required=True)
        if name in {"smoke", "fit"}:
            cmd.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "fit":
            cmd.add_argument("--task-index", type=int, required=True)
            cmd.add_argument("--force-evaluate", action="store_true")
        cmd.set_defaults(func=function)
    args = parser.parse_args()
    if args.command == "check":
        report = check_prepared(args.bundle.resolve(), args.prepared.resolve())
        print(json.dumps({"ready": True, "method": report["method"],
                          "targets": report["fold_order"]}))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
