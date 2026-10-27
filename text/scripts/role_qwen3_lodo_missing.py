#!/usr/bin/env python3
"""Finish CE and batch-hard Triplet on the frozen narrative-only LODO campaign.

The original campaign script and its 48 completed results remain immutable.
This extension records its own code hash and writes only under
``missing_ce_batch_triplet_v1`` inside the existing campaign run directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import role_qwen3_lodo_campaign as base

METHODS = ("ce", "batch_triplet")
SEEDS = base.SEEDS
CORPORA = base.CORPORA
EXTENSION = "missing_ce_batch_triplet_v1"


def extension_root() -> Path:
    return base.root() / EXTENSION


def extension_manifest(*, create: bool = False) -> Path:
    parent = base.campaign()  # verifies old script, dataset, Qwen export and splits
    path = extension_root() / "manifest.json"
    expected = {
        "version": EXTENSION,
        "parent_campaign_sha256": base.digest(parent),
        "extension_script_sha256": base.digest(Path(__file__)),
        "methods": list(METHODS),
        "targets": list(CORPORA),
        "seeds": list(SEEDS),
        "epochs": base.EPOCHS,
        "selection": "mean balanced accuracy on the five source validations",
        "triplet": {"mining": "batch-hard", "distance": "cosine",
                    "loss": "soft-margin", "batch_size": 10,
                    "projection_dim": 128},
        "ce_weight_dtype": "torch.float32",
    }
    if not path.is_file():
        if not create:
            raise FileNotFoundError(f"Run the CE/Triplet smoke first: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        base.write(path, expected)
    if base.read(path) != expected:
        trained = extension_root() / "trained"
        if create and not any(trained.rglob("complete.json")):
            # A failed smoke may require a code fix; no completed training
            # result exists to preserve under the old extension hash.
            base.write(path, expected)
        else:
            raise ValueError(f"CE/Triplet extension binding changed: {path}")
    return path


def result_dir(target: str, method: str, seed: int, *, smoke: bool = False) -> Path:
    if smoke:
        return extension_root() / "smoke" / method
    return extension_root() / "trained" / target / method / f"seed_{seed}"


def verified_result(dest: Path, parent: Path, extension: Path) -> bool:
    if not base.verified_done(dest, parent):
        return False
    try:
        return base.read(dest / "complete.json").get("extension_sha256") == base.digest(extension)
    except (OSError, ValueError):
        return False


def safe_remove_checkpoint(dest: Path) -> None:
    checkpoint = dest / "best_model"
    if checkpoint.is_dir():
        if checkpoint.is_symlink() or checkpoint.resolve().parent != dest.resolve():
            raise RuntimeError(f"Unsafe checkpoint cleanup path: {checkpoint}")
        shutil.rmtree(checkpoint)


def train_one(target: str, seed: int, method: str, *, smoke: bool = False) -> None:
    import torch
    from joblib import dump, load
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
    from contrastive_methods.hf_training_common import (
        load_contrastive_checkpoint, save_contrastive_checkpoint,
    )
    from contrastive_methods.losses.triplet_st import BatchTripletEmbeddingLoss
    from contrastive_methods.samplers.pk_batch_sampler import (
        build_pk_batch_sampler, resolve_balanced_pk_params,
    )
    from supervised_macro_ft.checkpoint_io import load_checkpoint, save_checkpoint
    from supervised_macro_ft.model import SupervisedMacroModel

    if target not in CORPORA or seed not in SEEDS or method not in METHODS:
        raise ValueError("Invalid target, seed or method")
    parent = base.campaign()
    extension = extension_manifest()
    dest = result_dir(target, method, seed, smoke=smoke)
    if verified_result(dest, parent, extension):
        safe_remove_checkpoint(dest)
        print(f"[skip] {dest}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Full Qwen3 fine-tuning needs CUDA")
    frames, splits, _ = base.load_data(check_qwen=False)
    train, val, test = base.selection(target, frames, splits)
    if smoke:
        train = pd.concat([train[train.pred_label.eq(role)].head(4)
                           for role in base.ROLES], ignore_index=True)
        val = pd.concat([val[(val.corpus == corpus) & val.pred_label.eq(role)].head(1)
                         for corpus in CORPORA if corpus != target for role in base.ROLES],
                        ignore_index=True)
        test = test.head(8).copy()

    base.old_role.seed_all(seed)
    device = torch.device("cuda")
    model_ref = base.old_qwen.backbone_path()
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if method == "ce":
        model = SupervisedMacroModel(
            backbone_name=model_ref, num_classes=5, pooling="mean",
            backbone_trainable=True, train_last_n_layers=None,
            gradient_checkpointing=True, projection=None, dropout=0.1,
        ).to(device)
        counts = np.bincount(train.label_id.to_numpy(), minlength=5).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError("CE training data are missing a role")
        # NumPy promotes integer / float32 division to float64. The old CE
        # loss therefore received double weights and failed on float logits.
        weights = torch.as_tensor(len(train) / (5 * counts), dtype=torch.float32,
                                  device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ], weight_decay=0.01)
        sampler = None
    else:
        model = ContrastiveEncoder(EncoderConfig(
            backbone_name=model_ref, max_seq_length=base.MAX_LENGTH,
            backbone_trainable=True, train_last_n_layers=None,
            cache_backbone_embeddings=False, use_projector=True,
            projection="linear", hiddim=128,
        )).to(device)
        loss_fn = BatchTripletEmbeddingLoss(distance_metric="cosine",
                                            soft_margin=True).to(device)
        sampler = build_pk_batch_sampler(
            train.label_id.tolist(), resolve_balanced_pk_params(
                train.label_id.tolist(), 10, seed=seed))
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": 2e-5},
            {"params": model.projector.parameters(), "lr": 1e-3},
        ], weight_decay=0.01)
    if not all(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Qwen3 backbone is not fully trainable")
    layers = model.backbone._get_transformer_layers()
    witness = next(p for p in layers[-1].parameters() if p.requires_grad)
    initial = witness.detach().float().cpu().clone()
    loader = base.text_loader(train, tokenizer,
                              batch_size=10 if method == "batch_triplet" else 4,
                              shuffle=sampler is None, sampler=sampler)
    val_loader = base.text_loader(val, tokenizer, batch_size=16)
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / "best_model"
    best, history = None, []
    for epoch in range(1 if smoke else base.EPOCHS):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total, count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accumulation = 1 if method == "batch_triplet" else 4
        for batch_index, batch in enumerate(loader):
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if method == "ce":
                    logits = model.forward_logits(x, mask)
                    loss = loss_fn(logits.float(), labels)
                else:
                    embeddings = model({"input_ids": x, "attention_mask": mask})
                    loss = loss_fn(embeddings.float(), labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {method} loss at epoch {epoch + 1}")
            group_size = min(accumulation, len(loader) -
                             (batch_index // accumulation) * accumulation)
            (loss / group_size).backward()
            total += float(loss.detach().cpu())
            count += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if method == "ce":
            val_pred, _ = base.old_role.ce_predictions(model, val_loader, device)
            score, per_source = base.macro_source_ba(val, val_pred, target)
            candidates = [({"epoch": epoch + 1, "C": None,
                            "source_macro_ba": score, "source_ba": per_source,
                            "train_loss": total / count}, None)]
        else:
            x_train = base.metric_embeddings(model, train, tokenizer, device, 32)
            x_val = base.metric_embeddings(model, val, tokenizer, device, 32)
            candidates = []
            for c in base.CS:
                classifier = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c, class_weight="balanced", max_iter=1000, random_state=seed))
                classifier.fit(x_train, train.label_id.to_numpy())
                score, per_source = base.macro_source_ba(
                    val, classifier.predict(x_val).astype(int), target)
                candidates.append(({"epoch": epoch + 1, "C": c,
                                    "source_macro_ba": score,
                                    "source_ba": per_source,
                                    "train_loss": total / count}, classifier))
        for row, classifier in candidates:
            history.append(row)
            if base.choose(best, row):
                best = row
                if method == "ce":
                    save_checkpoint(model, checkpoint, config={
                        "backbone_name": model_ref, "num_classes": 5,
                        "pooling": "mean", "projection": None,
                        "input_prefix": "", "max_seq_length": base.MAX_LENGTH})
                else:
                    save_contrastive_checkpoint(model, checkpoint)
                    dump(classifier, checkpoint / "classifier.joblib")
        print(json.dumps({"method": method, "seed": seed, "epoch": epoch + 1,
                          "best_source_ba": best["source_macro_ba"]}), flush=True)
    if torch.equal(initial, witness.detach().float().cpu()):
        raise RuntimeError(f"Failed to update full Qwen3 encoder: {method}")
    del layers, witness, initial, model, optimizer, loss_fn
    gc.collect()
    torch.cuda.empty_cache()
    if method == "ce":
        selected = load_checkpoint(checkpoint, device="cuda")
        pred, probs = base.old_role.ce_predictions(
            selected, base.text_loader(test, tokenizer, batch_size=16), device)
    else:
        cfg = ContrastiveConfig(method_name="batch_triplet", dataset_path=Path("unused"),
            backbone_name=model_ref, input_prefix="", backbone_trainable=True,
            train_last_n_layers=None, cache_backbone_embeddings=False,
            max_seq_length=base.MAX_LENGTH, use_projector=True,
            projection="linear", hiddim=128, use_amp=True,
            encode_batch_size=32, seed=seed)
        selected = load_contrastive_checkpoint(cfg, checkpoint, "cuda")
        classifier = load(checkpoint / "classifier.joblib")
        x_test = base.metric_embeddings(selected, test, tokenizer, device, 32)
        pred, probs = base.old_role.predict_lr(classifier, x_test)
    del selected
    gc.collect()
    torch.cuda.empty_cache()
    base.save_result(dest, test, pred, probs, target, method, seed, best, history)
    marker = base.read(dest / "complete.json")
    marker["extension_sha256"] = base.digest(extension)
    base.write(dest / "complete.json", marker)
    if not verified_result(dest, parent, extension):
        raise RuntimeError(f"Invalid CE/Triplet output: {dest}")
    safe_remove_checkpoint(dest)
    if not verified_result(dest, parent, extension):
        raise RuntimeError(f"Result changed after checkpoint cleanup: {dest}")
    print(f"[done] {dest}", flush=True)


def task(index: int) -> tuple[str, int, str]:
    if not 0 <= index < len(CORPORA) * len(SEEDS) * len(METHODS):
        raise ValueError("Expected task index 0..35")
    method = METHODS[index % 2]
    target = CORPORA[(index // 2) % len(CORPORA)]
    seed = SEEDS[index // (2 * len(CORPORA))]
    return target, seed, method


def smoke() -> None:
    parent = base.campaign()
    extension = extension_manifest(create=True)
    for method in METHODS:
        train_one(CORPORA[0], SEEDS[0], method, smoke=True)
    base.write(extension_root() / "smoke_complete.json", {
        "parent_campaign_sha256": base.digest(parent),
        "extension_sha256": base.digest(extension),
        "methods": list(METHODS)})


def train(index: int) -> None:
    parent = base.campaign()
    extension = extension_manifest()
    marker = base.read(extension_root() / "smoke_complete.json")
    if (marker.get("parent_campaign_sha256") != base.digest(parent) or
            marker.get("extension_sha256") != base.digest(extension) or
            marker.get("methods") != list(METHODS)):
        raise RuntimeError("Run the CE/Triplet GPU smoke first")
    train_one(*task(index))


def summary() -> None:
    parent = base.campaign()
    extension = extension_manifest()
    frames, _, _ = base.load_data(check_qwen=False)
    rows = []
    observed_methods = ("tfidf_lr", "qwen3_frozen_lr", "supcon", "softtriple", *METHODS)
    for target in CORPORA:
        for method in observed_methods:
            for seed in ((None,) if method.endswith("_lr") else SEEDS):
                if method in METHODS:
                    dest = result_dir(target, method, seed)
                    complete = verified_result(dest, parent, extension)
                else:
                    dest = (base.root() / "baselines" / target / method if seed is None
                            else base.root() / "trained" / target / method / f"seed_{seed}")
                    complete = base.verified_done(dest, parent)
                if not complete:
                    raise ValueError(f"Missing or changed result: {dest}")
                metrics = base.read(dest / "metrics.json")
                predictions = pd.read_csv(dest / "predictions.csv", dtype={
                    "accident_id": str, "doc_id": str, "fact_id": str})
                expected = frames[target]
                keys = ("corpus", "accident_id", "doc_id", "fact_id", "pred_label")
                if len(predictions) != len(expected) or any(
                    predictions[key].astype(str).tolist() != expected[key].astype(str).tolist()
                    for key in keys):
                    raise ValueError(f"Prediction/test units differ: {dest}")
                predicted_ids = predictions.prediction.map(base.LABEL2ID)
                if predicted_ids.isna().any():
                    raise ValueError(f"Unknown predicted role: {dest}")
                recomputed = base.old_role.scores(
                    expected.label_id.to_numpy(), predicted_ids.to_numpy(dtype=int))
                if any(not math.isclose(metrics[key], recomputed[key], abs_tol=1e-10)
                       for key in ("balanced_accuracy", "macro_f1")):
                    raise ValueError(f"Metric/predictions mismatch: {dest}")
                rows.append({"target": target, "method": method, "seed": seed,
                             "balanced_accuracy": metrics["balanced_accuracy"],
                             "macro_f1": metrics["macro_f1"],
                             "n_units": metrics["n_units"]})
    if len(rows) != 84:
        raise ValueError("Expected 84 six-target results")
    out = extension_root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "scores_by_run.csv", index=False)
    per_target = table.groupby(["target", "method"], as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        n_runs=("balanced_accuracy", "size"))
    per_target.to_csv(out / "scores_by_target.csv", index=False)
    per_seed = table.groupby(["method", "seed"], dropna=False, as_index=False).agg(
        mean_ba=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        worst_ba=("balanced_accuracy", "min"),
        n_targets=("target", "nunique"))
    if not per_seed.n_targets.eq(6).all():
        raise ValueError("Incomplete target set for a seed")
    per_seed.to_csv(out / "macro_by_seed.csv", index=False)
    macro = per_seed.groupby("method", as_index=False).agg(
        mean_ba=("mean_ba", "mean"), sd_ba=("mean_ba", "std"),
        mean_macro_f1=("mean_macro_f1", "mean"),
        sd_macro_f1=("mean_macro_f1", "std"),
        worst_ba=("worst_ba", "mean"), sd_worst_ba=("worst_ba", "std"),
        n_seeds=("mean_ba", "size"))
    macro.to_csv(out / "macro_six_targets.csv", index=False)
    base.write(out / "manifest.json", {
        "parent_campaign_sha256": base.digest(parent),
        "extension_sha256": base.digest(extension),
        "runs": len(rows), "methods": list(observed_methods),
        "note": "Per-seed six-domain mean and worst domain, then mean/sample SD across seeds"})
    print(f"[done] {out}: {len(rows)} verified results", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--task-index", type=int, required=True)
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "train":
        train(args.task_index)
    else:
        summary()


if __name__ == "__main__":
    main()
