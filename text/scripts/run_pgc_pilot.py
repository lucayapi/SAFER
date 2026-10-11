"""Run one stage of the source-only CE versus CE+PGC pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgc_pilot.core import EDGES, ROLES, accident_batches, balanced_accuracy_from_confusions, confusion_by_accident, pgc_loss
from safer_core.io import load_yaml
from safer_core.kfold_eval import group_kfold_splits
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.batch_utils import batch_to_device
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg
from supervised_macro_ft.train_loop import build_class_weights, build_optimizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_and_dir(config_path: Path, run_dir: Path):
    cfg = load_yaml(config_path)
    if cfg["model"]["backbone_name"] != "Qwen/Qwen3-Embedding-0.6B" or cfg["model"]["projection"] is not None:
        raise ValueError("Pilot requires selected Qwen3 full-encoder/no-projector configuration")
    if cfg["model"]["train_last_n_layers"] is not None or int(cfg["training"]["epochs"]) != 3:
        raise ValueError("Pilot requires full encoder and three epochs")
    if tuple(tuple(ROLES.index(role) for role in edge) for edge in cfg["edges"]) != EDGES:
        raise ValueError("Unexpected process-role contrasts")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return cfg, run_dir


def source_dataset(cfg):
    data = cfg["data"]
    path = ROOT / cfg["source_csv"]
    return TextRawDataset(
        str(path), label_col=data["label_col"], pred_ok_col=data["pred_ok_col"],
        group_col=data["group_col"], text_col=data["text_col"],
    )


def target_dataset(cfg, target: str):
    spec = resolve_test_corpus(target, require_files=True, require_emb_csv=False)
    data = cfg["data"]
    return TextRawDataset(
        str(spec.data_csv), label_col=data["label_col"], pred_ok_col=data["pred_ok_col"],
        group_col=data["group_col"], text_col=data["text_col"],
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preflight(cfg, run_dir: Path, config_path: Path):
    dataset = source_dataset(cfg)
    groups = dataset.get_groups()
    for target in cfg["targets"]:
        resolve_test_corpus(target, require_files=True, require_emb_csv=False)
    folds = group_kfold_splits(groups, int(cfg["n_folds"]), int(cfg["split_seed"]))
    partition_rows = []
    for fold, (train, validation) in enumerate(folds):
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("Accident leakage across source folds")
        partition_rows.extend((fold, "train", int(i), str(groups[i])) for i in train)
        partition_rows.extend((fold, "validation", int(i), str(groups[i])) for i in validation)
    pd.DataFrame(partition_rows, columns=["fold", "partition", "row_index", "accident_id"]).to_csv(run_dir / "source_partitions.csv", index=False)
    np.savez_compressed(run_dir / "source_folds.npz", **{
        f"{kind}_{fold}": indices for fold, pair in enumerate(folds)
        for kind, indices in zip(("train", "validation"), pair)
    })
    batches = accident_batches(np.arange(len(dataset)), groups,
        batch_size=int(cfg["training"]["batch_size"]),
        chunk_size=int(cfg["training"]["accident_chunk_size"]), seed=int(cfg["cv_seed"]))
    edge_pairs = Counter()
    edge_batches = Counter()
    active_batches = 0
    for batch in batches:
        roles_by_accident = {}
        for index in batch:
            roles_by_accident.setdefault(str(groups[index]), set()).add(int(dataset.label_ids[index]))
        active = False
        for source, target in EDGES:
            name = f"{ROLES[source]}->{ROLES[target]}"
            eligible = sum(source in roles and target in roles for roles in roles_by_accident.values())
            pairs = eligible * (eligible - 1) // 2
            edge_pairs[name] += pairs
            edge_batches[name] += int(pairs > 0)
            active |= pairs > 0
        active_batches += int(active)
    if not active_batches:
        raise RuntimeError("No eligible source batch for graph consistency")
    sampler_diagnostics = {"batches": len(batches), "active_batches": active_batches,
        "mean_batch_size": len(dataset) / len(batches), "pairs_by_edge": dict(edge_pairs),
        "active_batches_by_edge": dict(edge_batches)}
    (run_dir / "sampler_diagnostics.json").write_text(json.dumps(sampler_diagnostics, indent=2), encoding="utf-8")
    seeds = random.Random(int(cfg["seed_generator"])).sample(range(1, 2_147_483_647), int(cfg["n_seeds"]))
    source_hash = file_sha256(ROOT / cfg["source_csv"])
    config_hash = file_sha256(config_path)
    manifest = {"source_rows": len(dataset), "source_accidents": len(set(groups)), "source_sha256": source_hash,
                "config_sha256": config_hash, "final_seeds": seeds, "split_seed": int(cfg["split_seed"]),
                "lambda_grid": cfg["lambda_grid"], "source_only_selection": True,
                "sampler_diagnostics": sampler_diagnostics}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_bytes(config_path.read_bytes())
    print(json.dumps(manifest, indent=2), flush=True)


def require_manifest(cfg, run_dir, config_path):
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != file_sha256(config_path):
        raise ValueError("Pilot config changed after preflight")
    if manifest["source_sha256"] != file_sha256(ROOT / cfg["source_csv"]):
        raise ValueError("Source corpus changed after preflight")
    return manifest


def tokenizer_for(cfg):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone_name"], trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def fit(cfg, dataset, indices: np.ndarray, *, lam: float, seed: int, out_dir: Path,
        diagnostics_every: int = 0):
    if not torch.cuda.is_available():
        raise RuntimeError("Full Qwen3 fine-tuning requires a CUDA GPU")
    set_seed(seed)
    model = SupervisedMacroModel(**model_kwargs_from_cfg(cfg["model"])).to("cuda")
    tokenizer = tokenizer_for(cfg)
    training = cfg["training"]
    collate = make_text_collate_fn(tokenizer, int(cfg["model"]["max_seq_length"]))
    selected_labels = dataset.label_ids[indices].tolist()
    class_weights = build_class_weights(selected_labels, len(ROLES), training["class_weight"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to("cuda"))
    optimizer = build_optimizer(model, training)
    if bool(training["use_amp"]):
        raise ValueError("Pilot currently requires AMP disabled to match the selected CE recipe")
    group_values = dataset.get_groups()
    history = []
    batch_diagnostics = []
    total_graph_pairs = Counter()
    for epoch in range(int(training["epochs"])):
        batches = accident_batches(
            indices, group_values, batch_size=int(training["batch_size"]),
            chunk_size=int(training["accident_chunk_size"]), seed=seed + epoch,
        )
        loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate)
        model.train()
        totals = Counter()
        edge_pairs = Counter()
        edge_batches = Counter()
        for batch_number, batch in enumerate(loader, start=1):
            original_indices = batch["indices"].tolist()
            accident_ids = [str(group_values[i]) for i in original_indices]
            batch = batch_to_device(batch, torch.device("cuda"))
            optimizer.zero_grad(set_to_none=True)
            logits, z, _ = model.forward_with_latents(batch)
            ce = criterion(logits, batch["label_ids"])
            graph, counts = pgc_loss(z, batch["label_ids"], accident_ids)
            if not torch.isfinite(ce) or not torch.isfinite(graph):
                raise FloatingPointError(f"Non-finite CE/PGC loss: epoch={epoch + 1}, batch={batch_number}")
            loss = ce + float(lam) * graph
            if diagnostics_every and batch_number % diagnostics_every == 0:
                ce_grad = torch.autograd.grad(ce, z, retain_graph=True)[0]
                pgc_grad = torch.autograd.grad(graph, z, retain_graph=True)[0]
                ce_value = float(ce.detach())
                graph_value = float(graph.detach())
                ce_norm = float(torch.linalg.vector_norm(ce_grad.detach()))
                graph_norm = float(torch.linalg.vector_norm(pgc_grad.detach()))
                batch_diagnostics.append({
                    "epoch": epoch + 1, "batch": batch_number, "lambda": lam,
                    "n_units": len(original_indices), "ce": ce_value, "pgc": graph_value,
                    "weighted_pgc": lam * graph_value,
                    "weighted_loss_ratio": lam * graph_value / max(ce_value, 1e-12),
                    "ce_z_grad_norm": ce_norm, "pgc_z_grad_norm": graph_norm,
                    "weighted_z_grad_ratio": lam * graph_norm / max(ce_norm, 1e-12),
                    "active_edges": sum(count > 0 for count in counts.values()),
                    **{f"pairs_{edge}": count for edge, count in counts.items()},
                })
            loss.backward()
            optimizer.step()
            totals["units"] += len(original_indices)
            totals["batches"] += 1
            totals["ce_sum"] += float(ce.detach())
            totals["pgc_sum"] += float(graph.detach())
            if any(counts.values()):
                totals["active_batches"] += 1
            for edge, count in counts.items():
                edge_pairs[edge] += count
                edge_batches[edge] += int(count > 0)
        total_graph_pairs.update(edge_pairs)
        row = {"epoch": epoch + 1, "lambda": lam, "seed": seed, **totals,
               "mean_ce": totals["ce_sum"] / totals["batches"],
               "mean_pgc": totals["pgc_sum"] / totals["batches"]}
        row.update({f"pairs_{edge}": count for edge, count in edge_pairs.items()})
        row.update({f"batches_{edge}": count for edge, count in edge_batches.items()})
        history.append(row)
        print(json.dumps(row), flush=True)
    if lam > 0 and sum(total_graph_pairs.values()) == 0:
        raise RuntimeError("PGC had no eligible accident pairs in any epoch")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if diagnostics_every:
        pd.DataFrame(batch_diagnostics).to_csv(out_dir / "batch_diagnostics.csv", index=False)
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, dataset, cfg):
    model.eval()
    collate = make_text_collate_fn(tokenizer, int(cfg["model"]["max_seq_length"]))
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["eval_batch_size"]), shuffle=False, collate_fn=collate)
    predictions = []
    probabilities = []
    for batch in loader:
        batch = batch_to_device(batch, torch.device("cuda"))
        probs = torch.softmax(model(batch), dim=1).float().cpu().numpy()
        probabilities.append(probs)
        predictions.extend(probs.argmax(axis=1).tolist())
    probabilities = np.concatenate(probabilities, axis=0)
    metadata = dataset.get_metadata_df()
    frame = pd.DataFrame({"row_id": dataset.row_ids, "accident_id": dataset.get_groups(),
                          "true_id": dataset.label_ids, "pred_id": predictions,
                          "true_macro": [ROLES[i] for i in dataset.label_ids],
                          "pred_macro": [ROLES[i] for i in predictions]})
    for i, role in enumerate(ROLES):
        frame[f"prob_{role}"] = probabilities[:, i]
    if len(frame) != len(metadata):
        raise AssertionError("Prediction count mismatch")
    return frame


def metrics(frame):
    y, p = frame["true_id"], frame["pred_id"]
    if set(y.astype(int)) != set(range(len(ROLES))):
        raise ValueError("Evaluation corpus must contain all four roles")
    return {"n_units": len(frame), "n_accidents": frame["accident_id"].nunique(),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, labels=list(range(4)), average="macro", zero_division=0),
            "a1_support": int((y == 1).sum()),
            "a1_precision": precision_score(y == 1, p == 1, zero_division=0),
            "a1_recall": recall_score(y == 1, p == 1, zero_division=0)}


def cv_task(cfg, run_dir, task: int):
    grid = [float(value) for value in cfg["lambda_grid"]]
    folds_count = int(cfg["n_folds"])
    if not 0 <= task < len(grid) * folds_count:
        raise ValueError("Invalid CV task index")
    lam_idx, fold = divmod(task, folds_count)
    folds = np.load(run_dir / "source_folds.npz")
    dataset = source_dataset(cfg)
    train = folds[f"train_{fold}"]
    validation = folds[f"validation_{fold}"]
    if set(dataset.get_groups()[train]) & set(dataset.get_groups()[validation]):
        raise AssertionError("Accident leakage across source folds")
    out_dir = run_dir / "cv" / f"lambda_{lam_idx}_fold_{fold}"
    model, tokenizer = fit(cfg, dataset, train, lam=grid[lam_idx], seed=int(cfg["cv_seed"]) + fold, out_dir=out_dir)
    val_frame = predict_subset(model, tokenizer, dataset, validation, cfg)
    result = {"lambda_index": lam_idx, "lambda": grid[lam_idx], "fold": fold,
              "n_train": len(train), "n_validation": len(validation), **metrics(val_frame)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


@torch.no_grad()
def predict_subset(model, tokenizer, dataset, indices, cfg):
    subset = Subset(dataset, indices.tolist())
    model.eval()
    collate = make_text_collate_fn(tokenizer, int(cfg["model"]["max_seq_length"]))
    loader = DataLoader(subset, batch_size=int(cfg["training"]["eval_batch_size"]), shuffle=False, collate_fn=collate)
    predicted = []
    for batch in loader:
        batch = batch_to_device(batch, torch.device("cuda"))
        predicted.extend(model(batch).argmax(dim=1).cpu().tolist())
    return pd.DataFrame({"accident_id": dataset.get_groups()[indices], "true_id": dataset.label_ids[indices], "pred_id": predicted})


def select_lambda(cfg, run_dir):
    rows = []
    for index, lam in enumerate(cfg["lambda_grid"]):
        for fold in range(int(cfg["n_folds"])):
            path = run_dir / "cv" / f"lambda_{index}_fold_{fold}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["lambda_index"] != index or row["fold"] != fold or float(row["lambda"]) != float(lam):
                raise ValueError(f"Mismatched CV result: {path}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame["balanced_accuracy"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite source CV score")
    summary = frame.groupby(["lambda_index", "lambda"], as_index=False)["balanced_accuracy"].agg(["mean", "std"])
    summary = summary.sort_values(["mean", "lambda"], ascending=[False, True])
    summary.to_csv(run_dir / "source_cv_summary.csv", index=False)
    selected = float(summary.iloc[0]["lambda"])
    result = {"selected_lambda": selected, "selection_metric": "mean_source_grouped_cv_balanced_accuracy",
              "selected_index": int(summary.iloc[0]["lambda_index"]), "n_folds": int(cfg["n_folds"]),
              "no_source_signal": selected == 0.0}
    (run_dir / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def final_task(cfg, run_dir, task: int, manifest):
    n_seeds = int(cfg["n_seeds"])
    if not 0 <= task < 2 * n_seeds:
        raise ValueError("Invalid final task index")
    condition = "ce" if task < n_seeds else "ce_pgc"
    seed_index = task % n_seeds
    seed = int(manifest["final_seeds"][seed_index])
    selection_path = run_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selected = float(json.loads(selection_path.read_text(encoding="utf-8"))["selected_lambda"])
    lam = 0.0 if condition == "ce" else selected
    out_dir = run_dir / "final" / condition / f"seed_{seed_index}"
    dataset = source_dataset(cfg)
    model, tokenizer = fit(cfg, dataset, np.arange(len(dataset)), lam=lam, seed=seed, out_dir=out_dir)
    rows = []
    target_hashes = {}
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for target in cfg["targets"]:
        target_hashes[target] = file_sha256(resolve_test_corpus(target, require_files=True, require_emb_csv=False).data_csv)
        target_data = target_dataset(cfg, target)
        frame = predict(model, tokenizer, target_data, cfg)
        frame.to_csv(pred_dir / f"predictions_{target}.csv", index=False)
        rows.append({"condition": condition, "seed_index": seed_index, "training_seed": seed,
                     "lambda": lam, "target": target, **metrics(frame)})
    pd.DataFrame(rows).to_csv(out_dir / "target_metrics.csv", index=False)
    (out_dir / "run_manifest.json").write_text(json.dumps({"condition": condition, "seed_index": seed_index,
        "training_seed": seed, "lambda": lam, "source_only_selection": True,
        "target_sha256": target_hashes}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def analyze(cfg, run_dir, manifest):
    n_seeds = int(cfg["n_seeds"])
    selection_path = run_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selected_lambda = float(json.loads(selection_path.read_text(encoding="utf-8"))["selected_lambda"])
    frames = {}
    target_hashes = {}
    metric_rows = []
    for target in cfg["targets"]:
        reference = None
        for condition in ("ce", "ce_pgc"):
            for seed in range(n_seeds):
                run_root = run_dir / "final" / condition / f"seed_{seed}"
                run_manifest_path = run_root / "run_manifest.json"
                if not run_manifest_path.is_file():
                    raise FileNotFoundError(run_manifest_path)
                run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                expected_lambda = 0.0 if condition == "ce" else selected_lambda
                if (run_manifest["condition"] != condition or int(run_manifest["seed_index"]) != seed
                        or int(run_manifest["training_seed"]) != int(manifest["final_seeds"][seed])
                        or float(run_manifest["lambda"]) != expected_lambda):
                    raise ValueError(f"Mismatched final-run manifest: {run_manifest_path}")
                recorded_hash = run_manifest["target_sha256"][target]
                if target in target_hashes and target_hashes[target] != recorded_hash:
                    raise ValueError(f"Target corpus changed between final runs: {target}")
                target_hashes[target] = recorded_hash
                path = run_dir / "final" / condition / f"seed_{seed}" / "predictions" / f"predictions_{target}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                validate_prediction_frame(frame, path)
                key = prediction_alignment_key(frame)
                if reference is None:
                    reference = key
                elif not key.equals(reference):
                    raise ValueError(f"Predictions not paired on identical units: {path}")
                frames[condition, seed, target] = frame
                metric_rows.append({"condition": condition, "seed_index": seed, "target": target, **metrics(frame)})
    results = pd.DataFrame(metric_rows)
    results.to_csv(run_dir / "target_metrics_all.csv", index=False)
    seed_rows = []
    for condition in ("ce", "ce_pgc"):
        for seed in range(n_seeds):
            sub = results[(results.condition == condition) & (results.seed_index == seed)]
            seed_rows.append({"condition": condition, "seed_index": seed,
                              "mean_target_ba": sub.balanced_accuracy.mean(),
                              "worst_target_ba": sub.balanced_accuracy.min()})
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(run_dir / "per_seed_ood.csv", index=False)
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    n_boot = int(cfg["n_bootstrap"])
    summary_rows = []
    bootstrap_draws = {}
    for target in cfg["targets"]:
        accidents = sorted(frames["ce", 0, target]["accident_id"].astype(str).unique())
        all_counts = np.stack([
            np.stack([confusion_by_accident(frames[condition, seed, target], accidents) for seed in range(n_seeds)])
            for condition in ("ce", "ce_pgc")
        ])
        n_accidents = len(accidents)
        draws = np.empty((n_boot, 2), dtype=float)
        for start in range(0, n_boot, 64):
            stop = min(start + 64, n_boot)
            weights = rng.multinomial(n_accidents, np.full(n_accidents, 1 / n_accidents), size=stop - start)
            totals = np.einsum("ba,msarc->bmsrc", weights, all_counts, optimize=True)
            draws[start:stop] = balanced_accuracy_from_confusions(totals).mean(axis=2)
        bootstrap_draws[target] = draws
        for method_index, condition in enumerate(("ce", "ce_pgc")):
            selected_metrics = results[(results.target == target) & (results.condition == condition)]
            scores = selected_metrics.balanced_accuracy
            summary_rows.append({"target": target, "condition": condition, "mean_ba": scores.mean(),
                                 "seed_sd": scores.std(ddof=1), "ci_low": np.quantile(draws[:, method_index], .025),
                                 "ci_high": np.quantile(draws[:, method_index], .975),
                                 "mean_a1_precision": selected_metrics.a1_precision.mean(),
                                 "seed_sd_a1_precision": selected_metrics.a1_precision.std(ddof=1),
                                 "mean_a1_recall": selected_metrics.a1_recall.mean(),
                                 "seed_sd_a1_recall": selected_metrics.a1_recall.std(ddof=1),
                                 "a1_support": int(selected_metrics.a1_support.iloc[0])})
    contrasts = []
    for target in cfg["targets"]:
        delta = bootstrap_draws[target][:, 1] - bootstrap_draws[target][:, 0]
        empirical = results[results.target == target].pivot(index="seed_index", columns="condition", values="balanced_accuracy")
        contrasts.append({"target": target, "mean_delta_ba": (empirical.ce_pgc - empirical.ce).mean(),
                          "seed_sd_delta": (empirical.ce_pgc - empirical.ce).std(ddof=1),
                          "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975)})
    combined = np.stack([bootstrap_draws[target] for target in cfg["targets"]]).mean(axis=0)
    per_seed = seed_frame.pivot(index="seed_index", columns="condition", values="mean_target_ba")
    for method_index, condition in enumerate(("ce", "ce_pgc")):
        values = seed_frame[seed_frame.condition == condition].mean_target_ba
        summary_rows.append({"target": "equal_weight_mean", "condition": condition,
                             "mean_ba": values.mean(), "seed_sd": values.std(ddof=1),
                             "ci_low": np.quantile(combined[:, method_index], .025),
                             "ci_high": np.quantile(combined[:, method_index], .975)})
    pd.DataFrame(summary_rows).to_csv(run_dir / "bootstrap_intervals.csv", index=False)
    delta = combined[:, 1] - combined[:, 0]
    contrasts.append({"target": "equal_weight_mean", "mean_delta_ba": (per_seed.ce_pgc - per_seed.ce).mean(),
                      "seed_sd_delta": (per_seed.ce_pgc - per_seed.ce).std(ddof=1),
                      "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975)})
    pd.DataFrame(contrasts).to_csv(run_dir / "paired_differences.csv", index=False)
    seed_frame.groupby("condition", as_index=False).agg(
        mean_target_ba=("mean_target_ba", "mean"), seed_sd_mean_target_ba=("mean_target_ba", "std"),
        mean_worst_target_ba=("worst_target_ba", "mean"), seed_sd_worst_target_ba=("worst_target_ba", "std"),
    ).to_csv(run_dir / "ood_summary.csv", index=False)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


def prediction_alignment_key(frame):
    if frame["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id in stored predictions")
    return frame[["row_id", "accident_id", "true_id"]].astype(str).reset_index(drop=True)


def validate_prediction_frame(frame, path):
    required = {"row_id", "accident_id", "true_id", "pred_id", "true_macro", "pred_macro"}
    required |= {f"prob_{role}" for role in ROLES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")
    for identifier, role in (("true_id", "true_macro"), ("pred_id", "pred_macro")):
        values = frame[identifier].to_numpy(dtype=int)
        if np.any((values < 0) | (values >= len(ROLES))):
            raise ValueError(f"Invalid role ID in {path}")
        if not np.array_equal(frame[role].astype(str).to_numpy(), np.asarray(ROLES)[values]):
            raise ValueError(f"Inconsistent role name in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "cv", "select", "final", "analyze"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pgc_pilot.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    cfg, run_dir = config_and_dir(args.config, args.run_dir)
    if args.phase == "preflight":
        preflight(cfg, run_dir, args.config)
        return
    manifest = require_manifest(cfg, run_dir, args.config)
    if args.phase == "cv":
        cv_task(cfg, run_dir, args.task_id)
    elif args.phase == "select":
        select_lambda(cfg, run_dir)
    elif args.phase == "final":
        final_task(cfg, run_dir, args.task_id, manifest)
    else:
        analyze(cfg, run_dir, manifest)


if __name__ == "__main__":
    main()
