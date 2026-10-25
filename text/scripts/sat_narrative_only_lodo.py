#!/usr/bin/env python3
"""SaT+LoRA LODO on cleaned accident narratives, with edge-normalized exact F1.

Training examples contain factual units only. Validation and target test use the
cleaned narrative text from narrative_only_v1. Exact span scoring trims Unicode
whitespace and punctuation at both span edges, while preserving all internal
characters and punctuation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import tempfile
import unicodedata
from contextlib import closing
from pathlib import Path

import torch

import sat_lodo_gold_v3 as base

ROOT = base.ROOT
BASE_BUNDLE = base.BUNDLE
DATASET = ROOT / "text" / "dataset" / "narrative_only_v1"
PREPARED = ROOT / "annotation" / "lodo_sat_transfer" / "sat12l_sm_narrative_only_v1"
VERSION = "sat_narrative_only_lodo_v1"
METHOD = "SaT+LoRA; factual units only; narrative-only input"
FIELDS = ("span_tp", "span_pred", "span_ref")


def digest(path: Path) -> str:
    return base.digest(path)


def code_digest() -> str:
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def is_edge_char(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def trim_edges(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and is_edge_char(text[start]):
        start += 1
    while end > start and is_edge_char(text[end - 1]):
        end -= 1
    return start, end


def score(row: dict, predicted: list[tuple[int, int]]) -> dict[str, int]:
    raw = row["raw_text"]
    gold = {span for unit in row["units"]
            if (span := trim_edges(raw, int(unit["start"]), int(unit["end"])))[0] < span[1]}
    pred = {span for a, b in predicted
            if (span := trim_edges(raw, int(a), int(b)))[0] < span[1]}
    return {"span_tp": len(gold & pred), "span_pred": len(pred), "span_ref": len(gold)}


def prf(counts: dict[str, int]) -> tuple[float, float, float]:
    tp, predicted, reference = counts["span_tp"], counts["span_pred"], counts["span_ref"]
    p = tp / predicted if predicted else 0.0
    r = tp / reference if reference else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def runtime_signature(bundle: Path) -> dict:
    return {
        "version": VERSION,
        "method": METHOD,
        "base_bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "script_sha256": code_digest(),
        "template_sha256": digest(base.legacy.TEMPLATE),
        "fold_order": list(base.CORPORA),
        "seed": base.SEED,
        "training_target": "sorted literal factual units only; excluded spans removed from input",
        "selection_metric": "macro source micro exact-span F1 after trimming edge whitespace and Unicode punctuation",
    }


def prepared_signature(bundle: Path, dataset: Path) -> dict:
    return {**runtime_signature(bundle),
            "narrative_only_manifest_sha256": digest(dataset / "narrative_only_manifest.json"),
            "clean_snapshot_sha256": digest(dataset / "gold_snapshot.jsonl")}


def read_clean_snapshot(dataset: Path) -> dict[tuple[str, str], dict]:
    rows = base.read_jsonl(dataset / "gold_snapshot.jsonl")
    result = {}
    for row in rows:
        validate_clean_row(row)
        if row.get("excluded_spans"):
            raise ValueError("The narrative-only snapshot still contains excluded spans")
        key = row["corpus"], row["document_id"]
        if key in result:
            raise ValueError(f"Duplicate clean narrative: {key}")
        result[key] = row
    if len(result) != 17927:
        raise ValueError(f"Expected 17,927 cleaned narratives, got {len(result)}")
    return result


def validate_clean_row(row: dict) -> None:
    # Removing the only documentary content can legitimately leave an empty
    # input document. It remains in the split, contributes no gold/predicted
    # spans, and is skipped by model inference.
    if row["raw_text"]:
        base.validate_gold(row)
    else:
        surrogate = {**row, "raw_text": " "}
        base.validate_gold(surrogate)


def prepare(args: argparse.Namespace) -> None:
    bundle, dataset, output = args.bundle.resolve(), args.dataset.resolve(), args.out.resolve()
    base.check_bundle(bundle)
    clean = read_clean_snapshot(dataset)
    signature = prepared_signature(bundle, dataset)
    if output.exists():
        raise FileExistsError(f"Prepared folder exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".preparing-", dir=output.parent))
    files, reports = {}, {}
    try:
        shutil.copy2(dataset / "narrative_only_manifest.json", stage / "source_dataset_manifest.json")
        files["source_dataset_manifest.json"] = digest(stage / "source_dataset_manifest.json")
        for target in base.CORPORA:
            original_fold = bundle / "folds" / f"target_{target}"
            fold_dir = stage / "folds" / f"target_{target}"
            fold_dir.mkdir(parents=True)
            reports[target] = {}
            for split in ("train", "val", "test"):
                old_rows = base.read_jsonl(original_fold / f"{split}.jsonl")
                new_rows = []
                for old in old_rows:
                    key = old["corpus"], old["document_id"]
                    row = clean.get(key)
                    if row is None:
                        raise ValueError(f"Missing cleaned narrative {key}")
                    if len(old["units"]) != len(row["units"]):
                        raise ValueError(f"GOLD unit count changed for {key}")
                    for old_unit, new_unit in zip(old["units"], row["units"], strict=True):
                        old_text = old["raw_text"][old_unit["start"]:old_unit["end"]]
                        new_text = row["raw_text"][new_unit["start"]:new_unit["end"]]
                        if old_text != new_text or old_unit["role"] != new_unit["role"]:
                            raise ValueError(f"Factual unit text or role changed for {key}")
                    new_rows.append(row)
                path = fold_dir / f"{split}.jsonl"
                base.legacy.write_jsonl(path, new_rows)
                files[path.relative_to(stage).as_posix()] = digest(path)
                reports[target][f"{split}_documents"] = len(new_rows)
                reports[target][f"{split}_units"] = sum(len(r["units"]) for r in new_rows)

            train_rows = base.read_jsonl(fold_dir / "train.jsonl")
            val_rows = base.read_jsonl(fold_dir / "val.jsonl")
            train_chunks, val_chunks = [], []
            skipped_train = collections.Counter()
            skipped_val = collections.Counter()
            omitted_train = 0
            omitted_val = 0
            for rows, dest, skipped, omitted_ref in (
                (train_rows, train_chunks, skipped_train, "train"),
                (val_rows, val_chunks, skipped_val, "val"),
            ):
                omitted = 0
                for row in rows:
                    chunks, reason, n = base.trainer_chunks(row)
                    if chunks is None:
                        skipped[reason] += 1
                    else:
                        dest.append(chunks)
                        omitted += n
                if not dest:
                    raise ValueError(f"No trainable factual units for {target}/{omitted_ref}")
                if omitted_ref == "train":
                    omitted_train = omitted
                else:
                    omitted_val = omitted
            data = {"fr": {"sentence": {base.DATASET_NAME: {
                "meta": {"train_data": train_chunks}, "data": val_chunks}}}}
            path = fold_dir / "training.pth"
            torch.save(data, path)
            files[path.relative_to(stage).as_posix()] = digest(path)
            reports[target]["trainer"] = {
                "train_documents": len(train_chunks), "val_documents": len(val_chunks),
                "train_factual_units": sum(len(r["units"]) for r in train_rows),
                "val_factual_units": sum(len(r["units"]) for r in val_rows),
                "train_skipped": dict(skipped_train), "val_skipped": dict(skipped_val),
                "train_characters_omitted_from_trainer_copy": omitted_train,
                "val_characters_omitted_from_trainer_copy": omitted_val,
            }
            print(f"{target}: train={len(train_chunks)} docs, val={len(val_chunks)} docs, "
                  f"test={len(base.read_jsonl(fold_dir / 'test.jsonl'))} docs", flush=True)
        manifest = {**signature, "files": files, "fold_report": reports,
                    "created_at_utc": base.stamp()}
        readme = (
            "SaT+LoRA narrative_only_v1 transfer bundle\n"
            "\n"
            "Training chunks contain factual units only. All explicit excluded_spans were removed from the raw narrative before split export; validation and target test use the cleaned narrative. The six folds reuse the original RQ1 source/validation/target assignments.\n"
            "\n"
            "Primary metric: edge-normalized exact-span F1. Both predicted and gold offsets are trimmed over Unicode whitespace and Unicode punctuation at their outer edges only; internal punctuation and characters remain exact. Threshold choice uses only the five source validation corpora. One seed and the existing SaT+LoRA template are used.\n"
            "\n"
            "Transfer this entire folder to $SCRATCH/safer_private/sat12l_sm_narrative_only_v1/ on the cluster. The pre-existing sat12l_sm_gold_v3_rq1 base bundle and the updated SAFER code/jobs are also required. No separate narrative_only_v1 dataset transfer is required.\n"
        )
        (stage / "README.txt").write_text(readme, encoding="utf-8")
        files["README.txt"] = digest(stage / "README.txt")
        manifest["files"] = files
        base.write_json(stage / "manifest.json", manifest)
        base.write_json(stage / "complete.json", {"manifest_sha256": digest(stage / "manifest.json")})
        check_prepared(bundle, stage)
        stage.rename(output)
        print(json.dumps({"prepared": str(output), "targets": len(base.CORPORA),
                          "clean_documents": len(clean)}, ensure_ascii=False))
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def check_prepared(bundle: Path, prepared: Path) -> dict:
    base.check_bundle(bundle)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((prepared / "complete.json").read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != digest(prepared / "manifest.json"):
        raise ValueError("Prepared complete marker is invalid")
    if any(manifest.get(k) != v for k, v in runtime_signature(bundle).items()):
        raise ValueError("Prepared bundle has different source, data or code")
    source_manifest = prepared / "source_dataset_manifest.json"
    if (digest(source_manifest) != manifest.get("narrative_only_manifest_sha256")
            or json.loads(source_manifest.read_text(encoding="utf-8")).get("version") != "narrative_only_v1"):
        raise ValueError("Prepared bundle is not bound to narrative_only_v1")
    if not manifest.get("clean_snapshot_sha256"):
        raise ValueError("Prepared bundle lacks the clean source snapshot hash")
    expected_files = {f"folds/target_{t}/{s}.jsonl" for t in base.CORPORA for s in ("train", "val", "test")}
    expected_files |= {f"folds/target_{t}/training.pth" for t in base.CORPORA}
    expected_files |= {"source_dataset_manifest.json", "README.txt"}
    if set(manifest["files"]) != expected_files:
        raise ValueError("Prepared bundle does not contain the expected fold files")
    for relative, sha in manifest["files"].items():
        path = (prepared / relative).resolve()
        if not path.is_relative_to(prepared.resolve()) or not path.is_file() or digest(path) != sha:
            raise ValueError(f"Prepared file missing or changed: {relative}")
    for target in base.CORPORA:
        fold = prepared / "folds" / f"target_{target}"
        groups = {s: base.read_jsonl(fold / f"{s}.jsonl") for s in ("train", "val", "test")}
        keys = {s: {(r["corpus"], r["document_id"]) for r in rows} for s, rows in groups.items()}
        if any(keys[a] & keys[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Partition overlap in target {target}")
        if any(r["corpus"] == target for r in groups["train"] + groups["val"]):
            raise ValueError(f"Target leakage in target {target}")
        if any(r["corpus"] != target for r in groups["test"]):
            raise ValueError(f"Invalid test fold for target {target}")
        for split, rows in groups.items():
            for row in rows:
                validate_clean_row(row)
                if row.get("excluded_spans"):
                    raise ValueError("Excluded text remains in prepared narratives")
    return manifest


def binding(bundle: Path, prepared: Path) -> dict:
    return {"version": VERSION, "base_bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "prepared_manifest_sha256": digest(prepared / "manifest.json"),
            "script_sha256": code_digest(), "template_sha256": digest(base.legacy.TEMPLATE)}


def verify_smoke(root: Path, expected: dict) -> None:
    marker = root / "smoke_ok.json"
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("binding") != expected:
        raise RuntimeError("Run the narrative-only SaT GPU smoke first")


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
    target = base.CORPORA[0]
    source = torch.load(prepared / "folds" / f"target_{target}" / "training.pth",
                        map_location="cpu", weights_only=False)["fr"]["sentence"][base.DATASET_NAME]
    work = root / "smoke"
    work.mkdir(parents=True, exist_ok=True)
    path = work / "training.pth"
    torch.save({"fr": {"sentence": {base.DATASET_NAME: {
        "meta": {"train_data": source["meta"]["train_data"][:8]},
        "data": source["data"][:4]}}}}, path)
    config = base.trainer_config(bundle, target, work / "trainer", model_dir, tokenizer_dir)
    config.update({"text_path": str(path), "num_train_epochs": 1, "max_steps": 1,
                   "save_steps": 1, "per_device_train_batch_size": 2,
                   "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 1})
    base.legacy.run_trainer(repo, config, work / "config.json")
    adapter = base.legacy.adapter_folder(work / "trainer")
    if not base.legacy.adapter_exists(adapter):
        raise RuntimeError("Smoke did not save a loadable LoRA adapter")
    model = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    test_row = next(r for r in base.read_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl")
                    if r["raw_text"])
    base.legacy.pieces_to_spans(test_row["raw_text"], model.split(test_row["raw_text"], threshold=base.DEFAULT))
    root.mkdir(parents=True, exist_ok=True)
    base.write_json(root / "smoke_ok.json", {"binding": expected, "adapter": str(adapter),
                                              "completed_at_utc": base.stamp()})
    print(json.dumps({"smoke_ok": True, "adapter": str(adapter)}, ensure_ascii=False))


def select_threshold(val: list[dict], db) -> tuple[float, list[dict]]:
    source_domains = sorted({r["corpus"] for r in val})
    if len(source_domains) != 5:
        raise ValueError("Expected five source validation domains")
    counts = collections.defaultdict(collections.Counter)
    for row in val:
        if not row["raw_text"]:
            continue
        probs = base.get_probs(db, row)
        for threshold in base.THRESHOLDS:
            pred = base.legacy.spans_from_probabilities(row["raw_text"], probs, threshold)
            counts[row["corpus"], threshold].update(score(row, pred))
    curve = []
    for threshold in base.THRESHOLDS:
        by_source = {source: prf(counts[source, threshold])[2] for source in source_domains}
        macro = sum(by_source.values()) / len(by_source)
        curve.append({"threshold": threshold, "source_macro_edge_normalized_exact_f1": macro,
                      **{f"f1_{source}": by_source[source] for source in source_domains}})
    best = min(curve, key=lambda r: (-r["source_macro_edge_normalized_exact_f1"],
                                      abs(r["threshold"] - base.DEFAULT), r["threshold"]))
    for row in curve:
        row["selected"] = row is best
    return best["threshold"], curve


def evaluate(bundle: Path, prepared: Path, target: str, run: Path, adapter: Path,
             model_dir: Path, tokenizer_dir: Path) -> None:
    fold = prepared / "folds" / f"target_{target}"
    val, test = base.read_jsonl(fold / "val.jsonl"), base.read_jsonl(fold / "test.jsonl")
    model = base.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    with closing(base._probability_connection(run / "adapted_probabilities.sqlite3", create=True)) as db:
        nonempty = [r for r in val + test if r["raw_text"]]
        base.infer_adapted(model, db, nonempty)
        threshold, curve = select_threshold(val, db)
        base.verify_native(model, db, nonempty, threshold)
        base.write_csv(run / "source_validation_thresholds.csv", curve,
                       ("threshold", "source_macro_edge_normalized_exact_f1",
                        *(f"f1_{s}" for s in base.CORPORA if s != target), "selected"))
        total = collections.Counter()
        with (run / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in test:
                spans = (base.legacy.spans_from_probabilities(row["raw_text"], base.get_probs(db, row), threshold)
                         if row["raw_text"] else [])
                total.update(score(row, spans))
                stream.write(json.dumps({"corpus": target, "document_id": row["document_id"],
                    "threshold": threshold, "segments": [{"start": a, "end": b} for a, b in spans]},
                    ensure_ascii=False) + "\n")
    p, r, f1 = prf(total)
    base.write_json(run / "metrics.json", {"version": VERSION, "method": METHOD, "target": target,
        "n_test": len(test), "seed": base.SEED, "threshold": threshold,
        "selection": "source validation only; five-domain macro edge-normalized exact F1",
        "normalization": "trim Unicode whitespace and punctuation at span edges only",
        "counts": dict(total), "metrics": {"precision": p, "recall": r, "edge_normalized_exact_f1": f1},
        "completed_at_utc": base.stamp()})
    print(json.dumps({"target": target, "threshold": threshold,
                      "precision": p, "recall": r, "edge_normalized_exact_f1": f1}, indent=2), flush=True)


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
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "binding.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"Run directory belongs to different inputs: {run}")
    if not marker.exists():
        if any(run.iterdir()):
            raise ValueError(f"Unbound nonempty run folder: {run}")
        base.write_json(marker, expected)
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
    base.write_json(run / "training_done.json", {"target": target, "binding": expected,
        "adapter": str(adapter), "config_sha256": digest(run / "config.json"),
        "completed_at_utc": base.stamp()})
    if (run / "metrics.json").is_file() and not args.force_evaluate:
        print(f"Already evaluated: {run / 'metrics.json'}")
        return
    evaluate(bundle, prepared, target, run, adapter, model_dir, tokenizer_dir)


def summarize(args: argparse.Namespace) -> None:
    bundle, prepared, root = args.bundle.resolve(), args.prepared.resolve(), args.run_root.resolve()
    manifest = check_prepared(bundle, prepared)
    expected = binding(bundle, prepared)
    rows = []
    for target in base.CORPORA:
        run = root / f"target_{target}"
        if json.loads((run / "binding.json").read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Incompatible run: {target}")
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        expected_n = manifest["fold_report"][target]["test_documents"]
        if result["version"] != VERSION or result["target"] != target or result["n_test"] != expected_n:
            raise ValueError(f"Incomplete or incompatible target result: {target}")
        rows.append({"target": target, "n_test": expected_n, "threshold": result["threshold"],
                     **result["metrics"], **result["counts"]})
    metrics = ("precision", "recall", "edge_normalized_exact_f1")
    macro = {key: sum(r[key] for r in rows) / len(rows) for key in metrics}
    rows.append({"target": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in rows),
                 "threshold": "source-selected per target", **macro,
                 **{key: sum(r.get(key, 0) for r in rows) for key in FIELDS}})
    base.write_csv(root / "summary.csv", rows,
                   ("target", "n_test", "threshold", *metrics, *FIELDS))
    base.write_json(root / "summary.json", {"version": VERSION, "method": METHOD,
        "binding": expected, "seed": base.SEED, "results": rows,
        "normalization": "trim Unicode whitespace and punctuation at span edges only",
        "completed_at_utc": base.stamp()})
    print(json.dumps({"summary": str(root / "summary.csv"), "macro": macro}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("check", None), ("smoke", smoke), ("fit", fit), ("summarize", summarize)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bundle", type=Path, default=BASE_BUNDLE)
        if name == "prepare":
            cmd.add_argument("--dataset", type=Path, default=DATASET)
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
        print(json.dumps({"ready": True, "method": METHOD, "targets": report["fold_order"]}))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
