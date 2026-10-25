#!/usr/bin/env python3
"""Three-epoch, full-parameter SaT fine-tuning on elementary units.

Six leave-one-domain-out fits use the existing immutable facts-plus-exclusions
training bundle. Epoch and threshold are selected on the five source validation
domains; each target is read only after this choice is frozen.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from contextlib import closing
from pathlib import Path

import torch

import sat_elementary_lodo as elementary
import sat_lodo_checkpoint_select as checkpoint
import sat_lodo_gold_v3 as base

VERSION = "sat_elementary_full3_lodo_v1"
METHOD = "SaT full fine-tuning, three epochs, elementary units"
CURVE_FIELDS = ("candidate", "step", "epoch", "threshold",
                "macro_elementary_exact_f1", "macro_factual_exact_recall",
                "macro_excluded_exact_recall", "selected")
METRICS = ("elementary_exact_f1", "factual_exact_recall", "excluded_exact_recall")


def signature(bundle: Path, prepared: Path) -> dict:
    return {"version": VERSION,
            "bundle_sha256": base.digest(bundle / "manifest.json"),
            "elementary_sha256": base.digest(prepared / "manifest.json"),
            "vendor_sha256": base.digest(base.ROOT / "text" / "vendor" /
                                         "wtpsplit" / "SAFER_VENDOR.json"),
            "script_sha256": base.digest(Path(__file__)),
            "template_sha256": base.digest(base.legacy.TEMPLATE),
            "seed": base.SEED, "epochs": 3, "train_adapter": False}


def inputs(args):
    bundle, prepared, root = args.bundle.resolve(), args.prepared.resolve(), args.run_root.resolve()
    base.check_bundle(bundle)
    elementary.check_prepared(bundle, prepared)
    return bundle, prepared, root, signature(bundle, prepared)


def config_for(bundle, target, trainer, model_dir, tokenizer_dir, data_path):
    config = base.trainer_config(bundle, target, trainer, model_dir, tokenizer_dir)
    config.update({"text_path": str(data_path), "train_adapter": False,
                   "num_train_epochs": 3, "learning_rate": 2e-5,
                   "per_device_train_batch_size": 4,
                   "per_device_eval_batch_size": 4,
                   "gradient_accumulation_steps": 16,
                   "bf16": True, "fp16": False,
                   "save_strategy": "epoch", "save_total_limit": 3,
                   "load_best_model_at_end": False})
    if config.get("freeze_classifier") or config.get("clf_from_scratch"):
        raise ValueError("Full fine-tuning must include the pretrained classifier")
    return config


def weight_file(folder: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = folder / name
        if path.is_file() and path.stat().st_size:
            return path
    raise FileNotFoundError(f"Missing full model weights in {folder}")


def hashes(folder: Path) -> dict:
    return {name: base.digest(folder / name) for name in
            ("config.json", weight_file(folder).name)}


def load_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def export_checkpoint(folder: Path, output: Path, model_dir: Path, tokenizer_dir: Path,
                      epoch: float) -> dict:
    """Convert HF Trainer's Model(backbone) state to a loadable SaT backbone."""
    source = weight_file(folder)
    state_path = folder / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(folder.name.removeprefix("checkpoint-"))
    if int(state["global_step"]) != step or abs(float(state["epoch"]) - epoch) > 0.02:
        raise ValueError(f"Checkpoint is not the completed epoch {epoch}: {folder}")
    fingerprint = base.digest(source)
    metadata_path = output / "candidate.json"
    if metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (saved["checkpoint_sha256"] != fingerprint or saved["epoch"] != epoch or
                saved["weights"] != hashes(output)):
            raise ValueError(f"Existing epoch export changed: {output}")
        return saved
    if output.exists():
        raise ValueError(f"Incomplete epoch export: {output}")
    weights = load_state(source)
    if not weights or any(not key.startswith("backbone.") for key in weights):
        raise ValueError("Trainer checkpoint is not a complete wrapped backbone")
    weights = {key.removeprefix("backbone."): value for key, value in weights.items()}
    num_labels = int(weights["classifier.weight"].shape[0])
    from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
    import adapters

    cfg = SubwordXLMConfig.from_pretrained(str(model_dir), num_labels=num_labels)
    cfg.base_model = str(tokenizer_dir)
    backbone = SubwordXLMForTokenClassification(cfg)
    original_type = backbone.config.model_type
    backbone.config.model_type = "xlm-roberta"
    adapters.init(backbone)
    backbone.config.model_type = original_type
    backbone.load_state_dict(weights, strict=True)
    staging = output.with_name(output.name + ".preparing")
    if staging.exists():
        if staging.resolve().parent != output.resolve().parent:
            raise ValueError(f"Unsafe temporary export path: {staging}")
        shutil.rmtree(staging)  # An interrupted model-only export has no valid marker.
    staging.mkdir(parents=True)
    backbone.save_pretrained(staging, safe_serialization=True)
    del backbone, weights
    result = {"candidate": output.name, "step": step, "epoch": epoch,
              "checkpoint_sha256": fingerprint, "weights": hashes(staging)}
    checkpoint.write_json_atomic(staging / "candidate.json", result)
    staging.rename(output)
    return result


def epoch_checkpoints(trainer: Path) -> list[tuple[int, Path]]:
    folders = sorted((p for p in trainer.glob("checkpoint-*") if p.is_dir() and
                      p.name.removeprefix("checkpoint-").isdigit()),
                     key=lambda p: int(p.name.removeprefix("checkpoint-")))
    results = []
    for folder in folders:
        state_path = folder / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        epoch = round(float(state["epoch"]))
        if (1 <= epoch <= 3 and abs(float(state["epoch"]) - epoch) <= 0.02 and
                int(state["global_step"]) == int(folder.name.removeprefix("checkpoint-"))):
            weight_file(folder)
            results.append((epoch, folder))
    if [e for e, _ in results] != [1, 2, 3]:
        raise ValueError(f"Expected checkpoints at epochs 1, 2, 3; got {results}")
    return results


def smoke(args):
    bundle, prepared, root, expected = inputs(args)
    marker = root / "smoke_ok.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Smoke marker belongs to a different experiment")
        print(f"Full fine-tuning smoke already passed: {marker}", flush=True)
        return
    repo = args.wtpsplit_repo.resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    target = base.CORPORA[0]
    work = root / "smoke"
    elementary.bind_run(work, expected)
    source = torch.load(prepared / "folds" / f"target_{target}" / "training.pth",
                        map_location="cpu", weights_only=False)["fr"]["sentence"][base.DATASET_NAME]
    sample = {"fr": {"sentence": {base.DATASET_NAME: {
        "meta": {"train_data": source["meta"]["train_data"][:16]},
        "data": source["data"][:4]}}}}
    data_path = work / "training.pth"
    torch.save(sample, data_path)
    trainer = work / "trainer"
    config = config_for(bundle, target, trainer, model_dir, tokenizer_dir, data_path)
    config.update({"num_train_epochs": 1, "max_steps": 2,
                   "save_strategy": "steps", "save_steps": 1,
                   "per_device_train_batch_size": 2,
                   "per_device_eval_batch_size": 2,
                   "gradient_accumulation_steps": 1})
    if not (trainer / "checkpoint-2" / "optimizer.pt").is_file():
        base.legacy.run_trainer(repo, config, work / "config.json")
    folder = trainer / "checkpoint-1"
    if not (folder / "optimizer.pt").is_file():
        raise ValueError("Smoke did not save resumable full optimizer state")
    from wtpsplit.models import SubwordXLMForTokenClassification
    initial = SubwordXLMForTokenClassification.from_pretrained(str(model_dir))
    trained = load_state(weight_file(folder))
    encoder_keys = [key for key in initial.state_dict()
                    if key.startswith("roberta.encoder.layer.0.") and
                    "weight" in key and "lora" not in key]
    if not encoder_keys or not any(
            not torch.equal(initial.state_dict()[key], trained["backbone." + key])
            for key in encoder_keys):
        raise ValueError("Smoke did not update a pretrained encoder layer")
    del initial, trained
    smoke_epoch = float(json.loads((folder / "trainer_state.json").read_text(
        encoding="utf-8"))["epoch"])
    export = work / "exported_checkpoint"
    exported = export_checkpoint(folder, export, model_dir, tokenizer_dir, smoke_epoch)
    if exported["weights"] != hashes(export):
        raise ValueError("Checkpoint export changed during smoke")
    final = trainer / base.DATASET_NAME / "fr"
    hashes(final)
    raw = base.read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")[0]["raw_text"]
    for path in (export, final):
        model = base.load_pinned_sat(path, tokenizer_dir)
        base.legacy.pieces_to_spans(raw, model.split(raw, threshold=base.DEFAULT))
        del model
    state = json.loads((trainer / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) < 3:
        config["max_steps"] = 3
        config["resume_from_checkpoint"] = str(trainer / "checkpoint-2")
        base.legacy.run_trainer(repo, config, work / "resume_config.json")
        state = json.loads((trainer / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != 3:
        raise ValueError("Full-parameter optimizer resume did not reach step 3")
    # Smoke is diagnostic only; free its temporary optimizer and model copies.
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.resolve().parent == trainer.resolve():
            shutil.rmtree(folder)
    shutil.rmtree(final)
    shutil.rmtree(export)
    checkpoint.write_json_atomic(marker, expected)
    print(f"Full-model save/reload smoke passed: {marker}", flush=True)


def infer_to_cache(model_path, tokenizer_dir, rows, path, binding):
    marker = path.with_suffix(".binding.json")
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != binding:
            raise ValueError(f"Changed probability cache: {path}")
    elif path.exists():
        raise ValueError(f"Unbound probability cache: {path}")
    else:
        checkpoint.write_json_atomic(marker, binding)
    with closing(base._probability_connection(path, create=True)) as db:
        keys = {(r["corpus"], r["document_id"]) for r in rows}
        observed = set(db.execute("SELECT corpus,document_id FROM probabilities"))
        if not observed.issubset(keys):
            raise ValueError(f"Unexpected narratives in {path}")
        if len(observed) < len(keys):
            model = base.load_pinned_sat(model_path, tokenizer_dir)
            base.infer_adapted(model, db, rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if set(db.execute("SELECT corpus,document_id FROM probabilities")) != keys:
            raise ValueError(f"Incomplete probability cache: {path}")


def best_candidate(curve):
    if not curve:
        raise ValueError("No source-validation candidates")
    return min(curve, key=lambda row: (-row["macro_elementary_exact_f1"],
                                       abs(row["threshold"] - base.DEFAULT),
                                       row["epoch"], row["threshold"]))


def fit(args):
    bundle, prepared, root, expected = inputs(args)
    if json.loads((root / "smoke_ok.json").read_text(encoding="utf-8")) != expected:
        raise ValueError("Run the full fine-tuning GPU smoke first")
    if not 0 <= args.task_index < len(base.CORPORA):
        raise ValueError("task-index must be 0..5")
    target = base.CORPORA[args.task_index]
    repo = args.wtpsplit_repo.resolve()
    base.legacy.git_revision(repo)
    base.legacy.gpu_runtime(repo)
    model_dir, tokenizer_dir = base.pinned_backbone(bundle)
    run = root / f"target_{target}"
    elementary.bind_run(run, expected)
    trainer = run / "trainer"
    data_path = prepared / "folds" / f"target_{target}" / "training.pth"
    config = config_for(bundle, target, trainer, model_dir, tokenizer_dir, data_path)
    final = trainer / base.DATASET_NAME / "fr"
    if not (run / "metrics.json").is_file():
        if not (final / "config.json").is_file():
            # A resumed fit keeps all three epoch checkpoints until validation ends.
            checkpoints = sorted((p for p in trainer.glob("checkpoint-*") if p.is_dir() and
                                  (p / "optimizer.pt").is_file() and
                                  (p / "trainer_state.json").is_file()),
                                 key=lambda p: int(p.name.removeprefix("checkpoint-")))
            if checkpoints:
                config["resume_from_checkpoint"] = str(checkpoints[-1])
            base.legacy.run_trainer(repo, config, run / "config.json")
        hashes(final)
        base.legacy.export_loss_history(trainer / "trainer_state.json", run / "loss_history.csv")
        candidates = []
        for epoch, folder in epoch_checkpoints(trainer):
            output = run / "epoch_models" / f"epoch_{epoch}"
            candidates.append((export_checkpoint(folder, output, model_dir, tokenizer_dir, epoch), output))
        source_rows = base.read_jsonl(bundle / "folds" / f"target_{target}" / "val.jsonl")
        curve = []
        for candidate, model_path in candidates:
            cache = run / "source_cache" / f"{candidate['candidate']}.sqlite3"
            infer_to_cache(model_path, tokenizer_dir, source_rows, cache,
                           {"experiment": expected, "candidate": candidate})
            with closing(base._probability_connection(cache)) as db:
                _, thresholds = elementary.select_threshold(source_rows, db)
            curve.extend({"candidate": candidate["candidate"],
                          "step": candidate["step"], "epoch": candidate["epoch"],
                          **row, "selected": False} for row in thresholds)
        selected = best_candidate(curve)
        selected["selected"] = True
        base.write_csv(run / "source_validation.csv", curve, CURVE_FIELDS)
        selected_path = next(
            path for candidate, path in candidates if candidate["candidate"] == selected["candidate"])
        selection = {"experiment": expected, "target": target,
                     "criterion": "macro exact elementary F1 on five source domains",
                     "candidate": selected["candidate"], "epoch": selected["epoch"],
                     "step": selected["step"], "threshold": selected["threshold"],
                     "weights": hashes(selected_path),
                     "curve_sha256": base.digest(run / "source_validation.csv")}
        selection_path = run / "selection.json"
        if selection_path.is_file():
            if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
                raise ValueError("Source selection changed on resume")
        else:
            checkpoint.write_json_atomic(selection_path, selection)
        chosen_model = run / "selected_model"
        if chosen_model.exists():
            if hashes(chosen_model) != selection["weights"]:
                raise ValueError("Selected model changed on resume")
        else:
            shutil.copytree(selected_path, chosen_model,
                            ignore=shutil.ignore_patterns("candidate.json"))
            if hashes(chosen_model) != selection["weights"]:
                raise ValueError("Copied model differs from source-selected epoch")
        test_rows = base.read_jsonl(bundle / "folds" / f"target_{target}" / "test.jsonl")
        cache = run / "test_probabilities.sqlite3"
        infer_to_cache(chosen_model, tokenizer_dir, test_rows, cache,
                       {"experiment": expected, "selection": selection})
        model = base.load_pinned_sat(chosen_model, tokenizer_dir)
        with closing(base._probability_connection(cache)) as db:
            base.verify_native(model, db, test_rows, selection["threshold"])
        del model
        counts = collections.Counter()
        predictions = run / "predictions.jsonl"
        temporary = predictions.with_name(predictions.name + ".tmp")
        with closing(base._probability_connection(cache)) as db, temporary.open(
                "w", encoding="utf-8") as stream:
            for row in test_rows:
                pred = base.legacy.spans_from_probabilities(
                    row["raw_text"], base.get_probs(db, row), selection["threshold"])
                counts.update(elementary.score(row, pred))
                stream.write(json.dumps({"corpus": target, "document_id": row["document_id"],
                                         "threshold": selection["threshold"],
                                         "segments": [{"start": a, "end": b} for a, b in pred]},
                                        ensure_ascii=False) + "\n")
        temporary.replace(predictions)
        metrics = {"version": VERSION, "method": METHOD, "target": target,
                   "n_test": len(test_rows), "seed": base.SEED,
                   "epoch": selection["epoch"], "threshold": selection["threshold"],
                   "selection_sha256": base.digest(run / "selection.json"),
                   "predictions_sha256": base.digest(predictions),
                   "counts": dict(counts), "metrics": elementary.rates(counts)}
        checkpoint.write_json_atomic(run / "metrics.json", metrics)
        print(json.dumps({"target": target, **metrics["metrics"]}), flush=True)
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if (metrics["selection_sha256"] != base.digest(run / "selection.json") or
            metrics["predictions_sha256"] != base.digest(run / "predictions.jsonl") or
            hashes(run / "selected_model") != selection["weights"]):
        raise ValueError("Cannot clean unverified run")
    released = 0
    for folder in trainer.glob("checkpoint-*"):
        if folder.is_dir() and folder.resolve().parent == trainer.resolve() and \
                folder.name.removeprefix("checkpoint-").isdigit():
            released += sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            shutil.rmtree(folder)
    for folder in (run / "epoch_models").glob("epoch_*"):
        if folder.is_dir() and folder.resolve().parent == (run / "epoch_models").resolve() and \
                folder.name.removeprefix("epoch_").isdigit():
            released += sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            shutil.rmtree(folder)
    final = trainer / base.DATASET_NAME / "fr"
    if final.is_dir() and final.resolve().is_relative_to(trainer.resolve()):
        released += sum(p.stat().st_size for p in final.rglob("*") if p.is_file())
        shutil.rmtree(final)
    source_cache = run / "source_cache"
    if source_cache.is_dir() and source_cache.resolve().parent == run.resolve():
        released += sum(p.stat().st_size for p in source_cache.rglob("*") if p.is_file())
        shutil.rmtree(source_cache)
    checkpoint.write_json_atomic(run / "storage_cleanup.json", {"released_bytes": released})


def summarize(args):
    bundle, prepared, root, expected = inputs(args)
    manifest = base.check_bundle(bundle)
    rows = []
    for target in base.CORPORA:
        run = root / f"target_{target}"
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (json.loads((run / "binding.json").read_text(encoding="utf-8")) != expected or
                selection["experiment"] != expected or metrics["version"] != VERSION or
                metrics["target"] != target or
                metrics["n_test"] != manifest["source_report"][target]["gold_documents"] or
                metrics["selection_sha256"] != base.digest(run / "selection.json") or
                metrics["predictions_sha256"] != base.digest(run / "predictions.jsonl") or
                hashes(run / "selected_model") != selection["weights"]):
            raise ValueError(f"Incomplete or incompatible fold: {target}")
        rows.append({"target": target, "n_test": metrics["n_test"],
                     "epoch": metrics["epoch"], "threshold": metrics["threshold"],
                     **metrics["metrics"], **metrics["counts"]})
    macro = {name: sum(row[name] for row in rows) / len(rows) for name in METRICS}
    rows.append({"target": "MACRO_6_CORPUS", "n_test": sum(r["n_test"] for r in rows),
                 "epoch": "by_source_validation", "threshold": "by_source_validation",
                 **macro, **{key: sum(r.get(key, 0) for r in rows)
                            for key in elementary.COUNT_FIELDS}})
    base.write_csv(root / "summary.csv", rows,
                   ("target", "n_test", "epoch", "threshold", *METRICS,
                    *elementary.COUNT_FIELDS))
    checkpoint.write_json_atomic(root / "summary.json",
                                 {"version": VERSION, "method": METHOD,
                                  "experiment": expected, "results": rows})
    print(json.dumps({"summary": str(root / "summary.csv"), "macro": macro}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("check", None), ("smoke", smoke), ("fit", fit),
                           ("summarize", summarize)):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bundle", type=Path, default=elementary.BASE_BUNDLE)
        cmd.add_argument("--prepared", type=Path, default=elementary.PREPARED)
        cmd.add_argument("--run-root", type=Path, required=True)
        if name in {"smoke", "fit"}:
            cmd.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "fit":
            cmd.add_argument("--task-index", type=int, required=True)
        cmd.set_defaults(func=function)
    args = parser.parse_args()
    if args.command == "check":
        _, _, _, expected = inputs(args)
        print(json.dumps({"ready": True, "experiment": expected}), flush=True)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
