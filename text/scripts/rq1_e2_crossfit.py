#!/usr/bin/env python3
"""Cross-fitted SaT+LoRA features and E2 ablations for the RQ1 LODO study.

The source 80/20 split and target tests are taken verbatim from the frozen
RQ1 bundle. OOF features for E2 training are produced by a model that excludes
the corresponding fold. See text/annotation/README_rq1_e2_crossfit.md.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

import rq1_final as campaign
import rq1_final_lodo as final_lodo
import rq1_final_report as report
import sat_lodo_checkpoint_select as checkpoint
import sat_lodo_gold_v3 as sat
import rq1_refiner as refiner

VERSION = "rq1_e2_crossfit_v1"
SEED = 20261019
INTERNAL_VALIDATION_FRACTION = 0.10
SAVE_STEPS = checkpoint.SAVE_STEPS
E2_SEEDS = (13, 23, 37)
E2_VARIANTS = ("E2-A", "E2-B", "E2-C", "E2-D")
E2_INPUT_DIMS = {"E2-A": 1, "E2-B": 256, "E2-C": 257, "E2-D": 258}
E2_FEATURE_SLICES = {"E2-A": (256, 257), "E2-B": (0, 256),
                     "E2-C": (0, 257), "E2-D": (0, 258)}
E2_THRESHOLDS = (0.02, 0.05, 0.10)
E2_PENALTIES = (-2.0, -1.0, 0.0, 1.0, 2.0)
MAX_E2_EPOCHS = 8
E2_PATIENCE = 2
BOOTSTRAP_REPS = 2000
_VERIFIED_BUNDLE: tuple[str, str] | None = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict) -> tuple[str, str]:
    return row["corpus"], row["document_id"]


def normalized_text_hash(row: dict) -> str:
    return hashlib.sha256(" ".join(row["raw_text"].split()).casefold().encode("utf-8")).hexdigest()


def binding(bundle: Path) -> dict:
    return {"version": VERSION, "bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "script_sha256": digest(Path(__file__)),
            "refiner_sha256": digest(Path(refiner.__file__)),
            "sat_module_sha256": digest(Path(sat.__file__)),
            "report_module_sha256": digest(Path(report.__file__)),
            "checkpoint_module_sha256": digest(Path(checkpoint.__file__)),
            "lodo_module_sha256": digest(Path(final_lodo.__file__)),
            "trainer_template_sha256": digest(sat.legacy.TEMPLATE),
            "seed": SEED, "internal_validation_fraction": INTERNAL_VALIDATION_FRACTION,
            "projection_seed": 20261018, "projection_dim": refiner.PROJECTION_DIM}


def verify_campaign_binding(root: Path, bundle: Path) -> None:
    path = root / "binding.json"
    if not path.is_file() or json_read(path) != binding(bundle):
        raise ValueError(f"Campaign binding is missing or changed: {root}")


def verified_bundle(bundle: Path) -> dict:
    global _VERIFIED_BUNDLE
    manifest_sha = digest(bundle / "manifest.json")
    identity = (str(bundle.resolve()), manifest_sha)
    if _VERIFIED_BUNDLE != identity:
        campaign.check_bundle(bundle)
        _VERIFIED_BUNDLE = identity
    return json_read(bundle / "manifest.json")


def bundle_groups(bundle: Path, target: str) -> dict[str, list[dict]]:
    verified_bundle(bundle)
    folder = bundle / "folds" / f"target_{target}"
    groups = {name: sat.read_jsonl(folder / f"{name}.jsonl")
              for name in ("train", "val", "test")}
    ids = [{key(row) for row in groups[name]} for name in groups]
    if any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError(f"Train/validation/test overlap for {target}")
    if any(row["corpus"] == target for part in ("train", "val") for row in groups[part]):
        raise ValueError(f"Target used as source for {target}")
    if any(row["corpus"] != target for row in groups["test"]):
        raise ValueError(f"Wrong target test rows for {target}")
    train_texts = {normalized_text_hash(row) for row in groups["train"]}
    val_texts = {normalized_text_hash(row) for row in groups["val"]}
    test_texts = {normalized_text_hash(row) for row in groups["test"]}
    if train_texts & val_texts:
        raise ValueError(f"Textually duplicate source narratives cross A/V for {target}")
    if test_texts & (train_texts | val_texts):
        raise ValueError(f"Target-test text duplicates a source narrative for {target}")
    for rows in groups.values():
        for row in rows:
            sat.validate_gold(row)
    return groups


def hash_group_split(rows: list[dict], n_parts: int, seed: int, *, fraction: float | None = None):
    """Stratify by corpus while keeping identical normalized texts globally grouped."""
    by_hash: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_hash[normalized_text_hash(row)].append(row)
    assignment: dict[tuple[str, str], int] = {}
    totals = collections.Counter(row["corpus"] for row in rows)
    ordered = sorted(by_hash.items(), key=lambda item: hashlib.sha256(
        f"{seed}:{item[0]}".encode()).digest())
    if fraction is None:
        groups_per_corpus = collections.Counter(
            row["corpus"] for members in by_hash.values()
            for row in {key(item): item for item in members}.values())
        if any(groups_per_corpus[corpus] < n_parts for corpus in totals):
            raise ValueError(f"A source corpus has fewer than {n_parts} unique text groups")
        counts = {h: collections.Counter(row["corpus"] for row in members)
                  for h, members in by_hash.items()}
        bins = {corpus: [0] * n_parts for corpus in totals}
        desired = {corpus: max(1.0, totals[corpus] / n_parts) for corpus in totals}
        ordered = sorted(ordered, key=lambda item: (-sum(counts[item[0]].values()),
            hashlib.sha256(f"{seed}:{item[0]}".encode()).digest()))
        for text_hash, members in ordered:
            group_counts = counts[text_hash]
            def cost(fold: int) -> float:
                return sum(((bins[c][fold] + n) / desired[c]) ** 2
                           for c, n in group_counts.items())
            fold = min(range(n_parts), key=lambda index: (cost(index), index))
            for corpus, n in group_counts.items():
                bins[corpus][fold] += n
            for row in members:
                assignment[key(row)] = fold
        if any(not all(bins[corpus]) for corpus in totals):
            raise ValueError("A corpus was not represented in every cross-fit fold")
    else:
        if not 0 < fraction < 1:
            raise ValueError("Internal validation fraction must be between 0 and 1")
        counts = {h: collections.Counter(row["corpus"] for row in members)
                  for h, members in by_hash.items()}
        targets = {corpus: max(1.0, totals[corpus] * fraction) for corpus in totals}
        validation = collections.Counter()
        for text_hash, members in ordered:
            group_counts = counts[text_hash]
            before = sum(((validation[c] - targets[c]) / targets[c]) ** 2 for c in totals)
            after = sum(((validation[c] + group_counts[c] - targets[c]) / targets[c]) ** 2
                        for c in totals)
            if after < before or any(validation[c] == 0 for c in group_counts):
                for corpus, n in group_counts.items():
                    validation[corpus] += n
                part = 1
            else:
                part = 0
            for row in members:
                assignment[key(row)] = part
        for corpus in totals:
            if not 0 < validation[corpus] < totals[corpus]:
                raise ValueError(f"Invalid inner split for source corpus {corpus}")
    return assignment


def make_split_manifest(bundle: Path, out: Path) -> dict:
    manifest = verified_bundle(bundle)
    splits = {}
    for target in sat.CORPORA:
        groups = bundle_groups(bundle, target)
        train = groups["train"]
        folds = hash_group_split(train, 5, SEED + sat.CORPORA.index(target))
        if set(folds) != {key(row) for row in train}:
            raise ValueError(f"Fold assignment incomplete: {target}")
        fold_ids = {str(i): sorted(row["document_id"] for row in train
                                   if folds[key(row)] == i) for i in range(5)}
        fold_keys = {str(i): sorted(({"corpus":row["corpus"],"document_id":row["document_id"]}
                                     for row in train if folds[key(row)] == i),
                                    key=lambda item:(item["corpus"],item["document_id"]))
                     for i in range(5)}
        if any(not value for value in fold_ids.values()):
            raise ValueError(f"Empty cross-fit fold: {target}")
        fold_ids_by_corpus = {str(i): {
            corpus: sorted(row["document_id"] for row in train
                           if row["corpus"] == corpus and folds[key(row)] == i)
            for corpus in sat.CORPORA if corpus != target}
            for i in range(5)}
        grouped = collections.defaultdict(list)
        for row in train:
            grouped[normalized_text_hash(row)].append(row)
        duplicate_groups = []
        for text_hash, members in sorted(grouped.items()):
            if len(members) > 1:
                member_folds = {folds[key(row)] for row in members}
                if len(member_folds) != 1:
                    raise AssertionError("Duplicate narratives crossed cross-fit folds")
                duplicate_groups.append({"normalized_text_sha256": text_hash,
                    "members": sorted(({"corpus":row["corpus"],"document_id":row["document_id"]}
                                       for row in members),
                                       key=lambda item:(item["corpus"],item["document_id"])),
                    "fold": next(iter(member_folds))})
        split_digest = hashlib.sha256(json.dumps(fold_keys, sort_keys=True).encode()).hexdigest()
        splits[target] = {"A_ids_by_corpus": {
            corpus: [row["document_id"] for row in train if row["corpus"] == corpus]
            for corpus in sat.CORPORA if corpus != target},
            "V_ids_by_corpus": {corpus: [row["document_id"] for row in groups["val"]
                                           if row["corpus"] == corpus]
                                for corpus in sat.CORPORA if corpus != target},
            "fold_ids": fold_ids, "fold_keys": fold_keys,
            "fold_ids_by_corpus": fold_ids_by_corpus,
            "duplicate_text_groups": duplicate_groups,
            "fold_assignment_sha256": split_digest,
            "counts": {name: len(rows) for name, rows in groups.items()}}
    result = {"version": VERSION, "reference": sat.REFERENCE,
              "bundle_manifest_sha256": digest(bundle / "manifest.json"),
              "gold_snapshot_sha256": manifest["gold_snapshot_sha256"], "seed": SEED,
              "split_description": "reuse frozen source train=80% as A and val=20% as V; five grouped folds inside A",
              "groups": "casefolded whitespace-normalized raw_text hashes, kept in one fold across corpora",
              "targets": splits}
    return result


def prepare(args) -> None:
    bundle, out = args.bundle.resolve(), args.out.resolve()
    manifest = make_split_manifest(bundle, out)
    out.mkdir(parents=True, exist_ok=True)
    expected = binding(bundle)
    bp = out / "binding.json"
    if bp.is_file() and json_read(bp) != expected:
        raise ValueError("Existing campaign is bound to different code or GOLD")
    if not bp.exists():
        if any(out.iterdir()):
            raise ValueError(f"Unbound campaign output directory: {out}")
        json_write(bp, expected)
    mp = out / "split_manifest.json"
    if mp.is_file():
        if json_read(mp) != manifest:
            raise ValueError("Existing split manifest differs; refusing to rewrite partitions")
    else:
        json_write(mp, manifest)
    estimates = []
    for target in sat.CORPORA:
        completed = out / f"target_{target}" / "refiner" / "complete.json"
        if completed.is_file():
            marker = json_read(completed)
            metrics_path = completed.parent / "metrics.json"
            if (marker.get("target") == target and metrics_path.is_file() and
                    marker.get("metrics_sha256") == digest(metrics_path) and
                    marker.get("bundle_manifest_sha256") == digest(bundle / "manifest.json")):
                continue
        groups = bundle_groups(bundle, target)
        n_chars = sum(len(row["raw_text"]) for name in ("train", "val", "test")
                       for row in groups[name])
        # Six independent target rotations retain different M3 features for A/V/test.
        # A conservative 200 bytes per source character includes projected token
        # states, character probabilities, SQLite overhead, and sidecars.
        estimates.append({"target": target, "characters": n_chars,
                          "feature_bytes_estimate": int(n_chars * 200)})
    all_rotations = sum(x["feature_bytes_estimate"] for x in estimates)
    free = shutil.disk_usage(out).free
    reserve = 2 * 1024**3
    # The upstream checkpoint harvester retains only the newest full trainer
    # state per rotation and exports compact adapters for older checkpoints.
    trainer_headroom = 8 * 1024**3
    required = all_rotations + reserve + trainer_headroom
    if free < required:
        raise OSError(f"Insufficient campaign storage: {free/1024**3:.1f} GiB free, "
                      f"estimated {all_rotations/1024**3:.1f} GiB for retained features, "
                      f"{trainer_headroom/1024**3:.0f} GiB for two trainers, and "
                      f"{reserve/1024**3:.0f} GiB reserve")
    json_write(out / "preflight.json", {"checked_at_utc": sat.stamp(), "disk_free_bytes": free,
                "feature_estimates_by_target": estimates,
                "required_bytes_for_all_retained_features_plus_two_trainers_and_reserve": required,
                "target_test_documents": sum(manifest["targets"][t]["counts"]["test"]
                                              for t in sat.CORPORA)})
    print(json.dumps({"ready": True, "campaign": str(out),
                      "targets": {k: v["counts"] for k, v in manifest["targets"].items()},
                      "estimated_feature_gib": round(all_rotations/1024**3, 2),
                      "required_free_gib": round(required/1024**3, 2),
                      "disk_free_gib": round(free/1024**3, 2)}, ensure_ascii=False, indent=2))


def read_partition(bundle: Path, target: str, name: str) -> list[dict]:
    return sat.read_jsonl(bundle / "folds" / f"target_{target}" / f"{name}.jsonl")


def rows_by_ids(rows: list[dict], ids: set[tuple[str, str]]) -> list[dict]:
    return [row for row in rows if key(row) in ids]


def training_data(rows: list[dict], validation_rows: list[dict], path: Path) -> int:
    import torch
    train_chunks, val_chunks = [], []
    excluded = collections.Counter()
    for row in rows:
        chunks, reason, _ = sat.trainer_chunks(row)
        if chunks is None:
            excluded[reason] += 1
        else:
            train_chunks.append(chunks)
    for row in validation_rows:
        chunks, reason, _ = sat.trainer_chunks(row)
        if chunks is None:
            excluded["validation_" + reason] += 1
        else:
            val_chunks.append(chunks)
    if not train_chunks or not val_chunks:
        raise ValueError(f"No compatible training or internal validation narratives: {excluded}")
    sat.legacy.write_json(path.with_suffix(".counts.json"),
                          {"train_rows": len(rows), "train_examples": len(train_chunks),
                           "validation_rows": len(validation_rows),
                           "validation_examples": len(val_chunks), "skipped": dict(excluded)})
    torch.save({"fr": {"sentence": {sat.DATASET_NAME: {
        "meta": {"train_data": train_chunks}, "data": val_chunks}}}}, path)
    return len(train_chunks)


def make_inner_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    assigned = hash_group_split(rows, 2, seed, fraction=INTERNAL_VALIDATION_FRACTION)
    fit = [row for row in rows if assigned[key(row)] == 0]
    val = [row for row in rows if assigned[key(row)] == 1]
    if not fit or not val:
        raise ValueError("Empty internal training/validation split")
    return fit, val


def adapter_path(trainer: Path) -> Path:
    return sat.legacy.adapter_folder(trainer)


def ids_hash(rows: list[dict]) -> str:
    payload = "\n".join("\t".join(item) for item in sorted(key(row) for row in rows))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train_m3_select_and_refit(bundle: Path, repo: Path, root: Path, rows: list[dict],
                              target: str, model_tag: str, seed: int) -> dict:
    """Select epoch/threshold on an inner split, then refit on all permitted rows."""
    import torch
    from tqdm import tqdm
    allowed = {key(row) for row in rows}
    if any(row["corpus"] == target for row in rows):
        raise ValueError("Target narratives reached M3 trainer")
    stage = root / model_tag
    stage.mkdir(parents=True, exist_ok=True)
    selection_path = stage / "m3_selection.json"
    refit_adapter = stage / "selected_adapter"
    if selection_path.is_file() and checkpoint.complete_adapter(refit_adapter):
        selected = json_read(selection_path)
        if selected["allowed_ids_sha256"] != ids_hash(rows):
            raise ValueError(f"Allowed training IDs changed: {model_tag}")
        return selected

    fit_rows, val_rows = make_inner_split(rows, SEED + seed)
    for split in (fit_rows, val_rows):
        if any(key(row) not in allowed for row in split):
            raise ValueError("Inner split escaped allowed training narratives")
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    selection_root = stage / "selection"
    selection_trainer = selection_root / "trainer"
    pth = selection_root / "training.pth"
    training_data(fit_rows, val_rows, pth)
    config = sat.trainer_config(bundle, target, selection_trainer, model_dir, tokenizer_dir)
    config.update({"text_path": str(pth), "save_steps": SAVE_STEPS,
                   "save_total_limit": None, "load_best_model_at_end": False,
                   "seed": SEED + seed})
    template = adapter_path(root / "smoke" / "trainer")
    if not checkpoint.complete_adapter(template):
        raise RuntimeError("Cross-fit GPU smoke did not save the reference LoRA adapter")
    if not (selection_root / "training_done.json").is_file():
        trainer_state = selection_trainer / "trainer_state.json"
        if not (trainer_state.is_file() and checkpoint.complete_adapter(adapter_path(selection_trainer))):
            resumable = checkpoint.latest_resumable_checkpoint(selection_trainer)
            if resumable is not None:
                config["resume_from_checkpoint"] = str(resumable)
            final_lodo.run_trainer_with_harvest(repo, config, selection_root / "config.json",
                                                selection_root, template)
        sat.legacy.export_loss_history(selection_trainer / "trainer_state.json",
                                       selection_root / "loss_history.csv")
        json_write(selection_root / "training_done.json", {"config_sha256": digest(selection_root / "config.json"),
                   "allowed_ids_sha256": ids_hash(rows),
                   "inner_fit_ids_sha256": ids_hash(fit_rows),
                   "inner_validation_ids_sha256": ids_hash(val_rows),
                   "inner_fit_ids": [{"corpus":r["corpus"],"document_id":r["document_id"]}
                                     for r in fit_rows],
                   "validation_ids": [{"corpus":r["corpus"],"document_id":r["document_id"]}
                                      for r in val_rows]})

    candidates = final_lodo.candidates_from_exports(selection_root, selection_trainer)
    curves = []
    candidate_fingerprints = []
    best = None
    best_candidate_sha256 = None
    for item in tqdm(candidates, desc=f"Checkpoint selection {target}/{model_tag}", unit="checkpoint"):
        adapter = item["adapter"]
        if not checkpoint.complete_adapter(adapter):
            raise ValueError(f"Invalid candidate adapter: {adapter}")
        candidate_hashes = final_lodo.adapter_hashes(adapter)
        candidate_fingerprints.append({"candidate":item["candidate"],"step":item["step"],
            "epoch":item["epoch"],"adapter_sha256":candidate_hashes})
        cache_path = selection_root / "candidate_cache.sqlite3"
        if cache_path.exists():
            cache_path.unlink()
        with closing(sat._probability_connection(cache_path, create=True)) as db:
            model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
            sat.infer_adapted(model, db, val_rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            curve = checkpoint.validation_curve(val_rows, db, item["candidate"],
                                                 item["step"], item["epoch"])
        curves.extend(curve)
        candidate_best = checkpoint.best_row(curve)
        if best is None or (-candidate_best["source_macro_span_f1"],
                abs(candidate_best["threshold"] - sat.DEFAULT), candidate_best["step"],
                candidate_best["threshold"]) < (-best["source_macro_span_f1"],
                abs(best["threshold"] - sat.DEFAULT), best["step"], best["threshold"]):
            best = candidate_best
            best_candidate_sha256 = candidate_hashes
        cache_path.unlink(missing_ok=True)
    sat.write_csv(selection_root / "validation_curve.csv", curves, checkpoint.CURVE_FIELDS)
    json_write(selection_root / "candidate_adapters.json", candidate_fingerprints)
    if best is None:
        raise ValueError("No M3 checkpoint selected from inner validation")

    chosen_epoch = float(best["epoch"])
    if chosen_epoch <= 0:
        raise ValueError("Selected checkpoint has a nonpositive epoch")
    refit_root = stage / "refit"
    refit_trainer = refit_root / "trainer"
    refit_data = refit_root / "training.pth"
    training_data(rows, rows[:max(1, min(len(rows), 8))], refit_data)
    refit_config = sat.trainer_config(bundle, target, refit_trainer, model_dir, tokenizer_dir)
    refit_config.update({"text_path": str(refit_data), "num_train_epochs": chosen_epoch,
                         "save_strategy": "steps", "save_steps": SAVE_STEPS,
                         "save_total_limit": 2, "load_best_model_at_end": False,
                         "seed": SEED + seed + 100000})
    if not (refit_root / "training_done.json").is_file():
        final_adapter = adapter_path(refit_trainer)
        trainer_state = refit_trainer / "trainer_state.json"
        if not (trainer_state.is_file() and checkpoint.complete_adapter(final_adapter)):
            resumable = checkpoint.latest_resumable_checkpoint(refit_trainer)
            if resumable is not None:
                refit_config["resume_from_checkpoint"] = str(resumable)
            final_lodo.run_trainer_with_harvest(repo, refit_config, refit_root / "config.json",
                                                refit_root, template)
        sat.legacy.export_loss_history(refit_trainer / "trainer_state.json",
                                       refit_root / "loss_history.csv")
        if not checkpoint.complete_adapter(final_adapter):
            raise RuntimeError("Full-data refit did not save a valid LoRA adapter")
        shutil.copytree(final_adapter, refit_adapter, dirs_exist_ok=True)
        weights = final_lodo.adapter_hashes(refit_adapter)
        json_write(refit_root / "training_done.json", {"config_sha256": digest(refit_root / "config.json"),
                   "selected_epoch": chosen_epoch, "selected_adapter_sha256": weights,
                   "completed_at_utc": sat.stamp()})
        # Full states are removed only after the compact refit adapter is hashed.
        for cp in list(refit_trainer.glob("checkpoint-*")):
            if cp.is_dir() and cp.name.removeprefix("checkpoint-").isdigit():
                final_lodo.remove_checkpoint(refit_trainer, cp)
    if not checkpoint.complete_adapter(refit_adapter):
        raise RuntimeError("Selected refit adapter is missing during cleanup")
    selected_weights = final_lodo.adapter_hashes(refit_adapter)
    removed = []
    for cp in list(refit_trainer.glob("checkpoint-*")):
        if cp.is_dir() and cp.name.removeprefix("checkpoint-").isdigit():
            final_lodo.remove_checkpoint(refit_trainer, cp)
    adapter_snapshots = refit_root / "checkpoint_adapters"
    if adapter_snapshots.is_dir():
        for folder in list(adapter_snapshots.glob("checkpoint-*")):
            if folder.is_dir():
                if not checkpoint.complete_adapter(folder):
                    raise ValueError(f"Incomplete unselected refit adapter: {folder}")
                removed.append({"path":folder.name,
                    "adapter_sha256":final_lodo.adapter_hashes(folder),
                    "bytes":sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())})
                shutil.rmtree(folder)
    json_write(refit_root / "checkpoint_cleanup.json", {"selected_adapter_sha256":selected_weights,
                "removed_unselected_adapters":removed,
                "reason":"selected epoch fixed before full-A refit; no refit checkpoint is validation-selected"})
    selected = {"target": target, "model_tag": model_tag,
                "allowed_ids_sha256": ids_hash(rows),
                "selection_candidate": best["candidate"], "selection_step": best["step"],
                "selection_candidate_adapter_sha256": best_candidate_sha256,
                "selected_epoch": chosen_epoch, "selected_threshold": best["threshold"],
                "inner_validation_macro_f1": best["source_macro_span_f1"],
                "selection_validation_sha256": digest(selection_root / "validation_curve.csv"),
                "inner_fit_ids_sha256": ids_hash(fit_rows),
                "inner_validation_ids_sha256": ids_hash(val_rows),
                "inner_validation_ids": [{"corpus":r["corpus"],"document_id":r["document_id"]}
                                         for r in val_rows],
                "refit_adapter_sha256": final_lodo.adapter_hashes(refit_adapter),
                "adapter_path": str(refit_adapter), "inner_validation_n": len(val_rows),
                "refit_n_allowed": len(rows), "seed": SEED + seed}
    json_write(selection_path, selected)
    # Remove selector full-state files and unselected adapter snapshots; keep curves, logs and selection record.
    for cp in list(selection_trainer.glob("checkpoint-*")):
        if cp.is_dir() and cp.name.removeprefix("checkpoint-").isdigit():
            final_lodo.remove_checkpoint(selection_trainer, cp)
    for folder in (selection_root / "checkpoint_adapters").glob("checkpoint-*"):
        if folder.is_dir():
            shutil.rmtree(folder)
    selector_adapter = adapter_path(selection_trainer)
    if selector_adapter.is_dir() and selector_adapter.resolve() != refit_adapter.resolve():
        shutil.rmtree(selector_adapter)
    for path in (pth, refit_data):
        path.unlink(missing_ok=True)
    refit_training_adapter = adapter_path(refit_trainer)
    if refit_training_adapter.is_dir() and refit_training_adapter.resolve() != refit_adapter.resolve():
        if final_lodo.adapter_hashes(refit_training_adapter) != final_lodo.adapter_hashes(refit_adapter):
            raise ValueError("Cannot remove refit trainer adapter: selected copy differs")
        shutil.rmtree(refit_training_adapter)
    return selected


def feature_db(path: Path) -> sqlite3.Connection:
    return refiner._open_feature_db(path)


def validate_feature_tuple(row: dict, value) -> None:
    probs, states, offsets = value
    n = len(row["raw_text"])
    if (len(probs) != n or not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or
            states.ndim != 2 or states.shape[1] != refiner.PROJECTION_DIM or
            len(states) != len(offsets) or not np.isfinite(states).all() or
            offsets.ndim != 2 or offsets.shape[1] != 2 or
            np.any(offsets[:, 0] < 0) or np.any(offsets[:, 1] < offsets[:, 0]) or
            not np.any(offsets[:, 1] > offsets[:, 0]) or np.any(offsets[:, 1] > n) or
            np.any(offsets[1:, 0] < offsets[:-1, 0])):
        raise ValueError(f"Invalid feature/probability/token offsets for {key(row)}")


def add_features(bundle: Path, adapter: Path, rows: list[dict], db_path: Path,
                 description: str, *, limit: int | None = None) -> None:
    import torch
    from tqdm import tqdm
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    model.model.eval()
    with closing(feature_db(db_path)) as db:
        for row in tqdm(rows[:limit] if limit else rows, desc=description, unit="récit"):
            existing = refiner._read_feature(db, row)
            if existing is not None:
                validate_feature_tuple(row, existing)
                continue
            probs, states, offsets = refiner.extract_document(model, row["raw_text"])
            validate_feature_tuple(row, (probs, states, offsets))
            db.execute("INSERT INTO features VALUES (?,?,?,?,?,?,?)",
                       (*key(row), hashlib.sha256(row["raw_text"].encode()).hexdigest(),
                        probs.tobytes(), states.tobytes(), offsets.tobytes(), len(offsets)))
            db.commit()
        if not limit:
            expected = {key(row) for row in rows}
            found = set(db.execute("SELECT corpus,document_id FROM features"))
            if found != expected:
                raise ValueError(f"Feature IDs mismatch for {description}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def task_target_fold(task_index: int) -> tuple[str, int]:
    if not 0 <= task_index < len(sat.CORPORA) * 5:
        raise ValueError("Cross-fit task index must be 0..29")
    return sat.CORPORA[task_index // 5], task_index % 5


def crossfit(args) -> None:
    target, fold = task_target_fold(args.task_index)
    bundle, root = args.bundle.resolve(), args.out.resolve()
    verify_campaign_binding(root, bundle)
    manifest = json_read(root / "split_manifest.json")
    if manifest["bundle_manifest_sha256"] != digest(bundle / "manifest.json"):
        raise ValueError("Split manifest and input GOLD bundle differ")
    groups = bundle_groups(bundle, target)
    fold_ids = {(item["corpus"], item["document_id"])
                for item in manifest["targets"][target]["fold_keys"][str(fold)]}
    heldout = rows_by_ids(groups["train"], fold_ids)
    allowed = [row for row in groups["train"] if key(row) not in fold_ids]
    if set(map(key, heldout)) & set(map(key, allowed)) or not heldout or not allowed:
        raise ValueError("OOF fold is empty or overlapping")
    fold_dir = root / f"target_{target}" / "crossfit" / f"fold_{fold+1}"
    marker = fold_dir / "complete.json"
    expected = {"bundle_sha256": digest(bundle / "manifest.json"),
                "fold_assignment_sha256": manifest["targets"][target]["fold_assignment_sha256"],
                "allowed_ids_sha256": ids_hash(allowed),
                "heldout_ids_sha256": ids_hash(heldout),
                "script_sha256": digest(Path(__file__))}
    if marker.is_file():
        saved = json_read(marker)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing OOF fold belongs to another split/code")
        with closing(sqlite3.connect(fold_dir / "features.sqlite3")) as db:
            ids = set(db.execute("SELECT corpus,document_id FROM features"))
        if ids != set(map(key, heldout)):
            raise ValueError("Completed OOF feature cache is corrupt")
        with closing(feature_db(fold_dir / "features.sqlite3")) as db:
            for row in heldout:
                validate_feature_tuple(row, refiner._read_feature(db, row))
        print(f"OOF fold already complete: {target}/{fold+1}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    selection = train_m3_select_and_refit(bundle, args.wtpsplit_repo.resolve(), fold_dir,
                                          allowed, target, "m3", fold + sat.CORPORA.index(target)*10)
    add_features(bundle, Path(selection["adapter_path"]), heldout,
                 fold_dir / "features.sqlite3", f"OOF features {target}/fold{fold+1}", limit=args.limit)
    if args.limit:
        print("OOF feature smoke subset done (no completion marker written)", flush=True)
        return
    json_write(marker, {**expected, "target": target, "fold": fold,
                "adapter_sha256": selection["refit_adapter_sha256"],
                "n_heldout": len(heldout), "n_allowed_training": len(allowed),
                "selected_epoch": selection["selected_epoch"],
                "selected_threshold": selection["selected_threshold"]})
    print(f"Completed OOF {target}/fold{fold+1}: {len(heldout)} narratives", flush=True)


def combine_oof(root: Path, target: str, expected_rows: list[dict], output: Path) -> None:
    expected = {key(row): row for row in expected_rows}
    with closing(feature_db(output)) as out:
        for fold in range(5):
            path = root / f"target_{target}" / "crossfit" / f"fold_{fold+1}"
            marker = json_read(path / "complete.json")
            if marker["target"] != target or marker["fold"] != fold:
                raise ValueError(f"Wrong OOF fold marker: {path}")
            with closing(sqlite3.connect((path / "features.sqlite3").resolve().as_uri() +
                                         "?mode=ro", uri=True)) as src:
                ids = set(src.execute("SELECT corpus,document_id FROM features"))
                expected_ids = {(item["corpus"],item["document_id"])
                                for item in json_read(root / "split_manifest.json")["targets"][target]
                                    ["fold_keys"][str(fold)]}
                if ids != expected_ids:
                    raise ValueError(f"OOF partition mismatch for {target}/fold{fold+1}")
                for record in src.execute("SELECT corpus,document_id,text_sha256,prob,states,offsets,n_tokens FROM features"):
                    k = (record[0], record[1])
                    row = expected.get(k)
                    if row is None or record[2] != hashlib.sha256(row["raw_text"].encode()).hexdigest():
                        raise ValueError("OOF cache contains unknown/changed text")
                    out.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?)", record)
            out.commit()
        observed = set(out.execute("SELECT corpus,document_id FROM features"))
        if observed != set(expected):
            raise ValueError("Aggregated OOF features do not cover A exactly once")


def final_model(args) -> None:
    import torch
    target = sat.CORPORA[args.target_index]
    bundle, root = args.bundle.resolve(), args.out.resolve()
    verify_campaign_binding(root, bundle)
    groups = bundle_groups(bundle, target)
    target_root = root / f"target_{target}"
    saved_final = target_root / "m3_final.json"
    saved_features = target_root / "features.sqlite3"
    if saved_final.is_file() and saved_features.is_file():
        saved = json_read(saved_final)
        if saved.get("feature_cache_sha256") == digest(saved_features):
            with closing(feature_db(saved_features)) as db:
                if set(db.execute("SELECT corpus,document_id FROM features")) == {
                        key(row) for row in groups["val"] + groups["test"]}:
                    print(f"Final M3 features already complete: {target}", flush=True)
                    return
    for fold in range(5):
        marker = target_root / "crossfit" / f"fold_{fold+1}" / "complete.json"
        if not marker.is_file():
            raise RuntimeError(f"Missing OOF model/features: {marker}")
    final_selection = train_m3_select_and_refit(bundle, args.wtpsplit_repo.resolve(),
            target_root, groups["train"], target, "m3_final", 500 + args.target_index)
    features = target_root / "features.sqlite3"
    add_features(bundle, Path(final_selection["adapter_path"]), groups["val"] + groups["test"],
                 features, f"Final M3 features {target}", limit=args.limit)
    if args.limit:
        print("Final-model feature smoke subset done", flush=True)
        return
    # Require native probability equivalence against a second inference path on fixed samples.
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    model = sat.load_pinned_sat(model_dir, tokenizer_dir, Path(final_selection["adapter_path"]))
    with closing(feature_db(features)) as db:
        for row in (groups["val"][:3] + groups["test"][:3]):
            cached = refiner._read_feature(db, row)[0]
            native = np.asarray(model.predict_proba(row["raw_text"], stride=64), dtype=np.float32)
            if len(cached) != len(native) or np.max(np.abs(cached-native)) > 2e-3:
                raise ValueError(f"M3 hidden extraction probabilities disagree for {key(row)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # E0's threshold is selected inside A; E1 is independently recalibrated on V.
    with closing(feature_db(features)) as db:
        e1, curve = refiner.baseline_threshold(db, groups["val"])
    json_write(target_root / "E1_validation.json", {"selected": e1, "curve": curve,
                "selection_ids_sha256": ids_hash(groups["val"])})
    json_write(target_root / "m3_final.json", {**final_selection,
                "E0_threshold": final_selection["selected_threshold"],
                "E1_threshold": e1["threshold"],
                "feature_cache_sha256": digest(features),
                "validation_n": len(groups["val"]), "test_n": len(groups["test"])})
    print(f"Final M3 ready for {target}; E0={final_selection['selected_threshold']}, E1={e1['threshold']}", flush=True)


def feature_tuple(db: sqlite3.Connection, row: dict):
    value = refiner._read_feature(db, row)
    if value is None:
        raise ValueError(f"Missing features for {key(row)}")
    validate_feature_tuple(row, value)
    return value


def boundary_example(row: dict, feature, threshold: float) -> refiner.Example:
    example = refiner.make_example(row, feature, threshold, 0, "E2", training=False)
    raw = row["raw_text"]
    prefix, _, _ = refiner.nonspace_geometry(raw)
    positive = set(v for pair in refiner.gold_rank_spans(row, prefix) for v in pair
                   if 0 < v < int(prefix[-1]))
    units = [(int(prefix[u["start"]]), int(prefix[u["end"]])) for u in row["units"]]
    labels = []
    for rank in example.ranks:
        if rank in positive:
            labels.append(1.0)
            continue
        # Rank k is the position between non-space characters k and k+1.
        # It is a known negative only strictly inside a GOLD factual unit.
        # All other non-GOLD boundaries are unknown (including excluded text).
        labels.append(0.0 if any(a < rank < b for a, b in units) else -1.0)
    example.labels = np.asarray(labels, dtype=np.float32)
    return example


def variant_features(example: refiner.Example, variant: str) -> refiner.Example:
    start, end = E2_FEATURE_SLICES[variant]
    return refiner.Example(example.row, example.threshold, example.ranks,
        example.spans, np.ascontiguousarray(example.features[:, start:end]),
        example.labels.copy(), list(example.merges), list(example.fragments), "E2")


def boundary_network(variant: str):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(E2_INPUT_DIMS[variant], 256), torch.nn.ReLU(),
        torch.nn.Dropout(0.2), torch.nn.Linear(256, 64), torch.nn.ReLU(),
        torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))


def fit_e2_variant(variant: str, seed: int, train: list[refiner.Example],
                   validation: dict[float, list[refiner.Example]], out: Path, device: str) -> dict:
    import torch
    from tqdm import tqdm
    torch.manual_seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = boundary_network(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    by_domain = collections.defaultdict(list)
    for i, item in enumerate(train):
        by_domain[item.row["corpus"]].append(i)
    if len(by_domain) != 5:
        raise ValueError("E2 training needs five source domains")
    best, stale, history = None, 0, []
    for epoch in range(1, MAX_E2_EPOCHS+1):
        model.train(); losses=[]
        per_domain = math.ceil(len(train)/5)
        order = np.concatenate([rng.choice(indexes, per_domain, replace=per_domain>len(indexes))
                                for indexes in by_domain.values()])
        rng.shuffle(order)
        for offset in tqdm(range(0,len(order),16), desc=f"{variant} seed {seed} epoch {epoch}", leave=False):
            batch=[train[int(i)] for i in order[offset:offset+16] if len(train[int(i)].features)]
            if not batch: continue
            x=torch.from_numpy(np.concatenate([e.features for e in batch]).astype(np.float32)).to(device)
            y=torch.from_numpy(np.concatenate([e.labels for e in batch]).astype(np.float32)).to(device)
            known=y>=0
            if not known.any(): continue
            scores=model(x).squeeze(-1)
            positives=int((y[known]==1).sum().item()); negatives=int((y[known]==0).sum().item())
            if positives == 0: continue
            loss=torch.nn.functional.binary_cross_entropy_with_logits(scores[known], y[known],
                pos_weight=torch.tensor(min(10.,negatives/max(1,positives)),device=device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        chosen, grid=refiner.evaluate_grid(model, validation, "E2", device)
        record={"epoch":epoch,"train_loss":float(np.mean(losses)) if losses else None,**chosen}
        history.append(record); print(json.dumps({"variant":variant,"seed":seed,**record}),flush=True)
        if best is None or chosen["macro_exact_f1"] > best["macro_exact_f1"]+1e-12:
            best=record; stale=0; out.mkdir(parents=True,exist_ok=True)
            tmp=out/"best.pt.tmp"; torch.save({"state_dict":model.state_dict(),"variant":variant,
                "seed":seed,"epoch":epoch,"source_validation":chosen},tmp); tmp.replace(out/"best.pt")
            json_write(out/"validation_grid.json",grid)
        else:
            stale+=1
            if stale>=E2_PATIENCE: break
    if best is None: raise ValueError(f"No E2 checkpoint selected for {variant}/{seed}")
    json_write(out/"history.json",history); json_write(out/"selection.json",best)
    return best


def prediction_spans(model, example: refiner.Example, device: str, penalty: float):
    scores=refiner.predict_scores(model,example,device)
    chosen=[r for r,s in zip(example.ranks,scores,strict=True) if s>penalty]
    total=int(refiner.nonspace_geometry(example.row["raw_text"])[0][-1])
    bounds=[0,*chosen,total]
    return refiner.spans_from_ranks(example.row["raw_text"],list(zip(bounds,bounds[1:])))


def evaluate_variant_grid(model, validation, device):
    details=[]
    for threshold,examples in validation.items():
        scores=[refiner.predict_scores(model,e,device) for e in examples]
        for penalty in E2_PENALTIES:
            preds=[]
            for example,sc in zip(examples,scores,strict=True):
                chosen=[r for r,s in zip(example.ranks,sc,strict=True) if s>penalty]
                total=int(refiner.nonspace_geometry(example.row["raw_text"])[0][-1])
                preds.append(refiner.spans_from_ranks(example.row["raw_text"],
                    list(zip([0,*chosen],[*chosen,total]))))
            details.append({"threshold":threshold,"penalty":penalty,
                            "macro_exact_f1":refiner.macro_f1(examples,preds)})
    best=max(details,key=lambda r:(r["macro_exact_f1"],-abs(r["threshold"]-.05),
                                   -abs(r["penalty"]),-r["threshold"],-r["penalty"]))
    return best,details


def save_crossfit_prediction(path: Path, rows: list[dict], predictions, method: str,
                             threshold: float | None, seed: int | None):
    refiner.save_predictions(path,rows,predictions,model=method,threshold=threshold,seed=seed)


def refine_target(args) -> None:
    import torch
    target=sat.CORPORA[args.target_index]; bundle=args.bundle.resolve(); root=args.out.resolve()
    verify_campaign_binding(root,bundle)
    groups=bundle_groups(bundle,target); target_root=root/f"target_{target}"
    output=target_root/"refiner"
    existing_complete=output/"complete.json"
    if existing_complete.is_file():
        complete=json_read(existing_complete); metrics_file=output/"metrics.json"
        if complete.get("metrics_sha256") != digest(metrics_file):
            raise ValueError(f"Completed E2 metrics changed: {target}")
        for relative, sha in complete["predictions"].items():
            if digest(output/relative) != sha:
                raise ValueError(f"Completed E2 predictions changed: {target}/{relative}")
        print(f"E2 rotation already complete: {target}", flush=True)
        return
    final_path=target_root/"m3_final.json"
    if not final_path.is_file(): raise RuntimeError(f"Final M3 missing: {final_path}")
    final=json_read(final_path)
    oof_db=target_root/"oof_features.sqlite3"
    if not (target_root/"features.sqlite3").is_file(): raise FileNotFoundError(target_root/"features.sqlite3")
    combine_oof(root,target,groups["train"],oof_db)
    output.mkdir(parents=True,exist_ok=True)
    # Link/cache references are content bound; features are merged row by row below.
    combined=output/"all_features.sqlite3"
    with closing(feature_db(combined)) as db:
        for source_db, rows in ((oof_db,groups["train"]),(target_root/"features.sqlite3",groups["val"]+groups["test"])):
            with closing(sqlite3.connect(source_db.resolve().as_uri()+"?mode=ro",uri=True)) as src:
                for rec in src.execute("SELECT corpus,document_id,text_sha256,prob,states,offsets,n_tokens FROM features"):
                    db.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?)",rec)
            db.commit()
        expected={key(r) for part in groups.values() for r in part}
        if set(db.execute("SELECT corpus,document_id FROM features")) != expected:
            raise ValueError("Joined OOF and final-model features do not cover exact target rotation")
        # Source baseline calibration and test predictions.
        val_db=sqlite3.connect(combined.resolve().as_uri()+"?mode=ro",uri=True)
        e1,curve=refiner.baseline_threshold(val_db,groups["val"])
        json_write(output/"E1_validation.json",{"selected":e1,"curve":curve})
        test_predictions={}; test_metrics={}
        for label,threshold in (("E0",final["E0_threshold"]),("E1",e1["threshold"])):
            preds=[sat.legacy.spans_from_probabilities(r["raw_text"],sat.get_probs(val_db,r),threshold)
                   for r in groups["test"]]
            save_crossfit_prediction(output/"predictions"/f"{label}.jsonl",groups["test"],preds,label,threshold,None)
            test_predictions[label]=preds; test_metrics[label]=refiner.test_metrics(groups["test"],preds,label,target)
        val_db.close()
        max_len=refiner._max_source_gold_length(groups)
        # Training examples are built only from OOF predictions/states.
        train_examples=[boundary_example(row,feature_tuple(db,row),.02)
                        for row in groups["train"]]
        validation={}
        for threshold in E2_THRESHOLDS:
            validation[threshold]=[boundary_example(row,feature_tuple(db,row),threshold)
                                   for row in groups["val"]]
        test_by_threshold={threshold:[boundary_example(row,feature_tuple(db,row),threshold)
                                      for row in groups["test"]] for threshold in E2_THRESHOLDS}
        device="cuda" if torch.cuda.is_available() else "cpu"
        if device!="cuda": raise RuntimeError("Full E2 training requires a GPU")
        json_write(output/"parameters.json",{"feature_provenance":"M3 OOF for A; final M3 trained only on A for V/test",
            "target_test_used_for_selection":False,"candidate_thresholds":E2_THRESHOLDS,
            "penalties":E2_PENALTIES,"seeds":E2_SEEDS,"variants":E2_VARIANTS,
            "input_dims":E2_INPUT_DIMS,"hidden_widths":[256,64],"dropout":.2,
            "negative_boundary_rule":"known negative only inside one annotated factual unit; excluded/uncovered unknown",
            "projection_seed":20261018,"projection_dim":refiner.PROJECTION_DIM,
            "torch_version":torch.__version__,"target":target})
        for variant in E2_VARIANTS:
            start,end=E2_FEATURE_SLICES[variant]
            train=[variant_features(e,variant) for e in train_examples]
            val={t:[variant_features(e,variant) for e in examples]
                 for t,examples in validation.items()}
            for seed in E2_SEEDS:
                out=output/variant/f"seed_{seed}"
                if all((out/name).is_file() for name in ("best.pt","selection.json","history.json")):
                    choice=json_read(out/"selection.json")
                else:
                    choice=fit_e2_variant(variant,seed,train,val,out,device)
                state=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
                model=boundary_network(variant).to(device); model.load_state_dict(state["state_dict"])
                test_examples=[variant_features(e,variant)
                    for e in test_by_threshold[choice["threshold"]]]
                preds=[prediction_spans(model,e,device,choice["penalty"]) for e in test_examples]
                label=f"{variant}_seed_{seed}"
                save_crossfit_prediction(output/"predictions"/f"{label}.jsonl",groups["test"],preds,
                                         label,choice["threshold"],seed)
                test_metrics[label]=refiner.test_metrics(groups["test"],preds,label,target)
                del model
        val_db.close()
    metrics_path=output/"metrics.json"
    parameters_path=output/"parameters.json"
    split_hash=json_read(root/"split_manifest.json")["targets"][target]["fold_assignment_sha256"]
    json_write(metrics_path,{"target":target,"reference":sat.REFERENCE,
        "target_test_n":len(groups["test"]),"methods":test_metrics,
        "M3_selection_sha256":digest(final_path),"E1_threshold":e1["threshold"],
        "split_assignment_sha256":split_hash,
        "parameters_sha256":digest(parameters_path),
        "E1_validation_sha256":digest(output/"E1_validation.json"),
        "feature_provenance":"OOF A; final refit M3 for validation and target test",
        "target_labels_used_for_selection":False})
    predictions={str(p.relative_to(output)):digest(p) for p in (output/"predictions").glob("*.jsonl")}
    json_write(output/"complete.json",{"metrics_sha256":digest(metrics_path),"predictions":predictions,
        "parameters_sha256":digest(parameters_path),"n_methods":len(test_metrics),
        "target":target,"bundle_manifest_sha256":digest(bundle/"manifest.json"),
        "split_assignment_sha256":split_hash})
    print(json.dumps({"target":target,"completed_methods":list(test_metrics),"test_n":len(groups["test"])},indent=2),flush=True)


def smoke(args) -> None:
    """Run one official one-step LoRA training, reload it, then test our feature path."""
    import torch
    bundle, root, repo = args.bundle.resolve(), args.out.resolve(), args.wtpsplit_repo.resolve()
    verified_bundle(bundle)
    verify_campaign_binding(root, bundle)
    marker = root / "crossfit_smoke_ok.json"
    wanted = {"bundle_sha256": digest(bundle / "manifest.json"),
              "script_sha256": digest(Path(__file__)),
              "refiner_sha256": digest(Path(refiner.__file__))}
    if marker.is_file():
        saved = json_read(marker)
        if any(saved.get(k) != v for k, v in wanted.items()):
            raise ValueError("Existing cross-fit smoke marker belongs to other inputs/code")
        print(f"Cross-fit smoke already passed: {marker}", flush=True)
        return
    sat.smoke(argparse.Namespace(bundle=str(bundle), run_root=str(root),
                                  wtpsplit_repo=str(repo)))
    adapter = adapter_path(root / "smoke" / "trainer")
    if not checkpoint.complete_adapter(adapter):
        raise RuntimeError("Official smoke did not leave a reloadable LoRA adapter")
    target = sat.CORPORA[0]
    groups = bundle_groups(bundle, target)
    sample = next(row for row in groups["val"] if row["raw_text"].strip())
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    model = sat.load_pinned_sat(model_dir, tokenizer_dir, adapter)
    probs, states, offsets = refiner.extract_document(model, sample["raw_text"])
    native = np.asarray(model.predict_proba(sample["raw_text"], stride=64), dtype=np.float32)
    if (len(probs) != len(sample["raw_text"]) or len(probs) != len(native) or
            np.max(np.abs(probs - native)) > 2e-3 or states.shape[1] != refiner.PROJECTION_DIM or
            offsets.shape != (len(states), 2) or np.any(offsets < 0) or
            np.any(offsets[:, 1] > len(sample["raw_text"]))):
        raise ValueError("Smoke failed probabilities/token/character offset alignment")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    json_write(marker, {**wanted, "adapter_sha256": final_lodo.adapter_hashes(adapter),
        "sample": {"corpus": sample["corpus"], "document_id": sample["document_id"]},
        "probability_max_abs_error": float(np.max(np.abs(probs-native))),
        "projected_token_count": len(states), "projection_dim": int(states.shape[1]),
        "passed": True})
    print(json.dumps({"smoke_ok": True, "marker": str(marker)}, indent=2), flush=True)


def summary(args) -> None:
    bundle, root = args.bundle.resolve(), args.out.resolve()
    verify_campaign_binding(root, bundle)
    manifest = verified_bundle(bundle)
    targets = ["btp"] if args.scope == "pilot" else sat.CORPORA
    all_metrics, doc_archive = [], {}
    for target in targets:
        folder = root / f"target_{target}" / "refiner"
        complete_path = folder / "complete.json"
        metrics_path = folder / "metrics.json"
        if not complete_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"Cannot summarize incomplete target {target}: {complete_path}")
        complete, stored = json_read(complete_path), json_read(metrics_path)
        if (complete["bundle_manifest_sha256"] != digest(bundle / "manifest.json") or
                complete["metrics_sha256"] != digest(metrics_path) or
                complete.get("parameters_sha256") != stored.get("parameters_sha256") or
                complete.get("split_assignment_sha256") != json_read(root / "split_manifest.json")
                    ["targets"][target]["fold_assignment_sha256"] or
                stored["target"] != target):
            raise ValueError(f"Target summary binding/checksum mismatch: {target}")
        rows = bundle_groups(bundle, target)["test"]
        by_method = {}
        for relative, sha in complete["predictions"].items():
            path = folder / relative
            if digest(path) != sha:
                raise ValueError(f"Prediction file changed: {path}")
            method = path.stem
            pred = report.prediction_file(path, rows)
            records = report.metric_rows(rows, pred, VERSION, target, method)
            metric = report.aggregate(records, experiment=VERSION, source="", target=target,
                                      model=method)
            by_method[method] = (records, metric)
            all_metrics.append(metric)
        if set(by_method) != set(stored["methods"]):
            raise ValueError(f"Metrics and prediction methods differ for {target}")
        doc_archive[target] = by_method

    methods = sorted({m for target in targets for m in doc_archive[target]})
    method_rows = [r for r in all_metrics]
    macro_rows = []
    for method in methods:
        values = [m["exact_f1"] for m in method_rows if m["model"] == method]
        if len(values) != len(targets):
            continue
        macro_rows.append({"model": method, "n_domains": len(values),
                           "macro_exact_f1": float(np.mean(values)),
                           "sd_exact_f1": float(np.std(values, ddof=1)) if len(values)>1 else 0.0,
                           "macro_soft_f1": float(np.mean([m["soft_f1"] for m in method_rows if m["model"] == method])),
                           "macro_soft75_f1": float(np.mean([m["soft75_f1"] for m in method_rows if m["model"] == method])),
                           "macro_boundary_f1": float(np.mean([m["boundary_f1"] for m in method_rows if m["model"] == method]))})
    report.write_csv(root / f"{args.scope}_metrics_by_target.csv", method_rows)
    report.write_csv(root / f"{args.scope}_macro_by_method.csv", macro_rows)

    # Paired narrative bootstrap. Methods share precisely the same target test rows.
    rng = np.random.default_rng(20261019)
    comparisons = []
    reference_method = "E1"
    candidate_methods = [m for m in methods if m != reference_method]
    per_target_boot = {}
    for target in targets:
        by_method = doc_archive[target]
        if reference_method not in by_method:
            raise ValueError(f"Missing paired reference method {reference_method}/{target}")
        ref_records = by_method[reference_method][0]
        sampled_index = rng.integers(0, len(ref_records), size=(BOOTSTRAP_REPS, len(ref_records)))
        ref_stats = np.asarray([[x["span_tp"],x["span_pred"],x["span_ref"],
                                 x["soft_credit"],x["boundary_tp"],x["boundary_pred"],x["boundary_ref"]]
                                for x in ref_records], dtype=float)
        def three_f1(sums):
            return np.column_stack((
                np.divide(2*sums[:,0],sums[:,1]+sums[:,2],out=np.zeros(len(sums)),where=sums[:,1]+sums[:,2]!=0),
                np.divide(2*sums[:,3],sums[:,1]+sums[:,2],out=np.zeros(len(sums)),where=sums[:,1]+sums[:,2]!=0),
                np.divide(2*sums[:,4],sums[:,5]+sums[:,6],out=np.zeros(len(sums)),where=sums[:,5]+sums[:,6]!=0)))
        ref_boot = three_f1(ref_stats[sampled_index].sum(axis=1))
        per_target_boot[target] = {}
        ref_point = three_f1(ref_stats.sum(axis=0, keepdims=True))[0]
        for method in candidate_methods:
            records = by_method[method][0]
            if [r["document_id"] for r in records] != [r["document_id"] for r in ref_records]:
                raise ValueError(f"Bootstrap is not paired for {target}/{method}")
            stats = np.asarray([[x["span_tp"],x["span_pred"],x["span_ref"],
                                 x["soft_credit"],x["boundary_tp"],x["boundary_pred"],x["boundary_ref"]]
                                for x in records], dtype=float)
            boot_delta = three_f1(stats[sampled_index].sum(axis=1)) - ref_boot
            point = three_f1(stats.sum(axis=0, keepdims=True))[0] - ref_point
            per_target_boot[target][method] = boot_delta
            for metric_index, metric_name in enumerate(("exact_f1", "soft_f1", "boundary_f1")):
                comparisons.append({"target":target,"method":method,"reference":reference_method,
                    "metric":metric_name,"difference":float(point[metric_index]),
                    "ci95_low":float(np.quantile(boot_delta[:,metric_index],.025)),
                    "ci95_high":float(np.quantile(boot_delta[:,metric_index],.975)),
                    "bootstrap_reps":BOOTSTRAP_REPS})
    if len(targets) > 1:
        for method in candidate_methods:
            macro_boot = np.mean(np.stack([per_target_boot[t][method] for t in targets]), axis=0)
            point_rows = [row for row in comparisons if row["method"]==method and row["target"] in targets]
            for metric_index, metric_name in enumerate(("exact_f1","soft_f1","boundary_f1")):
                values = [r["difference"] for r in point_rows if r["metric"]==metric_name]
                comparisons.append({"target":"MACRO","method":method,"reference":reference_method,
                    "metric":metric_name,"difference":float(np.mean(values)),
                    "ci95_low":float(np.quantile(macro_boot[:,metric_index],.025)),
                    "ci95_high":float(np.quantile(macro_boot[:,metric_index],.975)),
                    "bootstrap_reps":BOOTSTRAP_REPS})
    report.write_csv(root / f"{args.scope}_paired_bootstrap.csv", comparisons)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{120+len(method_rows)*48}" viewBox="0 0 1100 {120+len(method_rows)*48}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#183047;font-size:14px}</style>',
           f'<text x="25" y="32" font-size="20">Cross-fit E2 {args.scope}: exact-span F1</text>']
    for idx, metric in enumerate(method_rows):
        y=75+idx*48; value=metric["exact_f1"]
        svg.extend([f'<text x="25" y="{y+18}">{metric["target"]} / {metric["model"]}</text>',
                    f'<rect x="260" y="{y}" width="{700*value:.2f}" height="25" fill="#376995"/>',
                    f'<text x="{270+700*value:.2f}" y="{y+18}">{value:.4f}</text>'])
    svg.append('</svg>')
    (root / f"{args.scope}_exact_f1.svg").write_text("\n".join(svg), encoding="utf-8")
    title = "BTP pilote" if args.scope == "pilot" else "Six rotations LODO"
    lines = [f"# Résultats cross-fit E2 — {title}", "",
             "Les résultats tests sont descriptifs : aucun choix de checkpoint, seuil ou pénalité n’a utilisé les tests.",
             "Les caractéristiques d’entraînement E2 proviennent de M3 hors pli; validation et test utilisent M3 refit sur A.",
             "Le choix d’architecture E2 a déjà été informé par des tests exploratoires antérieurs; cette campagne n’est donc pas une confirmation totalement indépendante.", "",
             "| Cible | Méthode | Exact-span F1 | Soft Span F1 | Soft Span F1 IoU≥0,75 | Boundary F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for m in method_rows:
        lines.append(f"| {m['target']} | {m['model']} | {m['exact_f1']:.4f} | {m['soft_f1']:.4f} | {m['soft75_f1']:.4f} | {m['boundary_f1']:.4f} |")
    lines += ["", "Les intervalles appariés figurent dans le CSV dédié; les comparaisons opposent chaque méthode à E1.",
              "Les fichiers CSV conservent également les diagnostics de fusion, fragmentation, frontière décalée, unité manquée et segment superflu.", ""]
    (root / f"{args.scope.upper()}_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"scope":args.scope,"targets":targets,"method_rows":len(method_rows),
                      "macro_by_method":macro_rows,"report":str(root / f"{args.scope.upper()}_RESULTS.md")},
                     ensure_ascii=False,indent=2),flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "crossfit", "final", "refine", "summary"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name in {"smoke", "crossfit", "final"}:
            command.add_argument("--wtpsplit-repo", type=Path, required=True)
        if name == "crossfit":
            command.add_argument("--task-index", type=int, required=True)
        if name in {"final", "refine"}:
            command.add_argument("--target-index", type=int, required=True)
        if name == "summary":
            command.add_argument("--scope", choices=("pilot", "all"), required=True)
        if name in {"crossfit", "final"}:
            command.add_argument("--limit", type=int)
    actions = {"prepare":prepare,"smoke":smoke,"crossfit":crossfit,
               "final":final_model,"refine":refine_target,"summary":summary}
    args = parser.parse_args()
    actions[args.command](args)


if __name__ == "__main__":
    main()
