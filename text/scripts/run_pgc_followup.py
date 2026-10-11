"""CE lambda diagnostics and source-selected, sampler-matched SupCon+PGC pilot.

Run one phase per SLURM task. Target labels are read only by final/diagnostic
evaluation and analysis, never by SupCon lambda selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
from contrastive_methods.losses.supcon_hobbit import SupConEmbeddingLoss
from pgc_pilot.core import accident_batches, balanced_accuracy_from_confusions, confusion_by_accident, pgc_loss
from safer_core.classification_eval import fit_logistic_on_embeddings
from scgm_text.batch_utils import batch_to_device
from scgm_text.collate import make_text_collate_fn
from scripts import run_pgc_pilot as pilot


def supcon_batches(indices, dataset, cfg, seed):
    options = cfg["supcon"]
    return accident_batches(indices, dataset.get_groups(), batch_size=int(options["batch_size"]),
                            chunk_size=int(options["accident_chunk_size"]), seed=seed)


def supcon_batch_support(dataset, batches):
    valid_anchors = 0
    total = 0
    active_graph = 0
    for batch in batches:
        labels = dataset.label_ids[batch]
        _, counts = np.unique(labels, return_counts=True)
        valid_anchors += int(counts[counts > 1].sum())
        total += len(batch)
        by_accident = {}
        for index in batch:
            by_accident.setdefault(str(dataset.groups[index]), set()).add(int(dataset.label_ids[index]))
        active_graph += int(any(sum(a in roles and b in roles for roles in by_accident.values()) >= 2
                                for a, b in pilot.EDGES))
    return {"batches": len(batches), "units": total, "valid_supcon_anchor_fraction": valid_anchors / total,
            "graph_active_batch_fraction": active_graph / len(batches)}


def preflight(cfg, run_dir, config_path):
    pilot.preflight(cfg, run_dir, config_path)
    dataset = pilot.source_dataset(cfg)
    batches = supcon_batches(np.arange(len(dataset)), dataset, cfg, int(cfg["cv_seed"]))
    support = supcon_batch_support(dataset, batches)
    if support["valid_supcon_anchor_fraction"] < 0.5 or support["graph_active_batch_fraction"] < 0.5:
        raise RuntimeError(f"Accident batches give inadequate SupCon/PGC support: {support}")
    (run_dir / "supcon_sampler_diagnostics.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
    print(json.dumps(support, indent=2), flush=True)


def write_target_outputs(cfg, run_dir, out_dir, *, family, lam_index, lam, seed_index, seed, predict_fn):
    rows = []
    hashes = {}
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for target in cfg["targets"]:
        spec = pilot.resolve_test_corpus(target, require_files=True, require_emb_csv=False)
        hashes[target] = pilot.file_sha256(spec.data_csv)
        frame = predict_fn(pilot.target_dataset(cfg, target))
        pilot.validate_prediction_frame(frame, pred_dir / f"predictions_{target}.csv")
        frame.to_csv(pred_dir / f"predictions_{target}.csv", index=False)
        rows.append({"family": family, "lambda_index": lam_index, "lambda": lam,
                     "seed_index": seed_index, "training_seed": seed, "target": target, **pilot.metrics(frame)})
    pd.DataFrame(rows).to_csv(out_dir / "target_metrics.csv", index=False)
    (out_dir / "done.json").write_text(json.dumps({"family": family, "lambda_index": lam_index,
        "lambda": lam, "seed_index": seed_index, "training_seed": seed,
        "target_sha256": hashes, "target_used_for_selection": False}, indent=2), encoding="utf-8")


def ce_task(cfg, run_dir, task, manifest):
    n_seeds = int(cfg["n_seeds"])
    lam_index, seed_index = divmod(task, n_seeds)
    grid = [float(v) for v in cfg["lambda_grid"]]
    if not 0 <= lam_index < len(grid):
        raise ValueError(f"Invalid CE task: {task}")
    lam = grid[lam_index]
    seed = int(manifest["final_seeds"][seed_index])
    out_dir = run_dir / "ce_diagnostic" / f"lambda_{lam_index}" / f"seed_{seed_index}"
    dataset = pilot.source_dataset(cfg)
    model, tokenizer = pilot.fit(cfg, dataset, np.arange(len(dataset)), lam=lam, seed=seed,
        out_dir=out_dir, diagnostics_every=int(cfg["diagnostics_every"]))
    write_target_outputs(cfg, run_dir, out_dir, family="ce_diagnostic", lam_index=lam_index,
        lam=lam, seed_index=seed_index, seed=seed,
        predict_fn=lambda target: pilot.predict(model, tokenizer, target, cfg))


def supcon_model(cfg):
    options = cfg["supcon"]
    model = ContrastiveEncoder(EncoderConfig(
        backbone_name=cfg["model"]["backbone_name"], max_seq_length=int(cfg["model"]["max_seq_length"]),
        backbone_trainable=True, train_last_n_layers=None, cache_backbone_embeddings=False,
        use_projector=True, projection=options["projection"], hiddim=int(options["hiddim"])))
    objective = SupConEmbeddingLoss(temperature=float(options["temperature"]),
        base_temperature=float(options["base_temperature"]), contrast_mode="all",
        normalize_embeddings=True)
    return model.to("cuda"), objective.to("cuda")


def fit_supcon(cfg, dataset, indices, *, lam, seed, out_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("SupCon full-encoder fine-tuning requires CUDA")
    pilot.set_seed(seed)
    model, objective = supcon_model(cfg)
    options = cfg["supcon"]
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=float(options["learning_rate"]),
                                  weight_decay=float(options["weight_decay"]))
    collate = make_text_collate_fn(model.tokenizer, int(cfg["model"]["max_seq_length"]))
    history, diagnostics = [], []
    groups = dataset.get_groups()
    for epoch in range(int(cfg["training"]["epochs"])):
        batches = supcon_batches(indices, dataset, cfg, seed + epoch)
        loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate)
        model.train()
        base_sum = graph_sum = 0.0
        valid_anchors = active_batches = 0
        for batch_number, batch in enumerate(loader, start=1):
            positions = batch["indices"].tolist()
            accidents = [str(groups[i]) for i in positions]
            batch = batch_to_device(batch, torch.device("cuda"))
            optimizer.zero_grad(set_to_none=True)
            z = model({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
            base = objective(z, batch["label_ids"])
            graph, counts = pgc_loss(z, batch["label_ids"], accidents)
            if not torch.isfinite(base) or not torch.isfinite(graph):
                raise FloatingPointError(f"Non-finite SupCon/PGC loss: epoch={epoch + 1}, batch={batch_number}")
            total = base + lam * graph
            if batch_number % int(cfg["diagnostics_every"]) == 0:
                base_grad = torch.autograd.grad(base, z, retain_graph=True)[0]
                graph_grad = torch.autograd.grad(graph, z, retain_graph=True)[0]
                base_value, graph_value = float(base.detach()), float(graph.detach())
                base_norm = float(torch.linalg.vector_norm(base_grad.detach()))
                graph_norm = float(torch.linalg.vector_norm(graph_grad.detach()))
                diagnostics.append({"epoch": epoch + 1, "batch": batch_number, "lambda": lam,
                    "n_units": len(positions), "supcon": base_value, "pgc": graph_value,
                    "weighted_pgc": lam * graph_value,
                    "weighted_loss_ratio": lam * graph_value / max(base_value, 1e-12),
                    "supcon_z_grad_norm": base_norm, "pgc_z_grad_norm": graph_norm,
                    "weighted_z_grad_ratio": lam * graph_norm / max(base_norm, 1e-12),
                    "active_edges": sum(count > 0 for count in counts.values()),
                    **{f"pairs_{edge}": count for edge, count in counts.items()}})
            total.backward()
            optimizer.step()
            base_sum += float(base.detach())
            graph_sum += float(graph.detach())
            labels = batch["label_ids"].cpu().numpy()
            _, frequencies = np.unique(labels, return_counts=True)
            valid_anchors += int(frequencies[frequencies > 1].sum())
            active_batches += int(any(counts.values()))
        row = {"epoch": epoch + 1, "lambda": lam, "seed": seed, "batches": len(batches),
            "mean_supcon": base_sum / len(batches), "mean_pgc": graph_sum / len(batches),
            "valid_anchor_fraction": valid_anchors / len(indices),
            "active_graph_batch_fraction": active_batches / len(batches)}
        history.append(row)
        print(json.dumps(row), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "batch_diagnostics.csv", index=False)
    return model


@torch.no_grad()
def encode_supcon(model, dataset, indices, cfg):
    model.eval()
    collate = make_text_collate_fn(model.tokenizer, int(cfg["model"]["max_seq_length"]))
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]),
        batch_size=int(cfg["supcon"]["eval_batch_size"]), shuffle=False, collate_fn=collate)
    parts = []
    for batch in loader:
        batch = batch_to_device(batch, torch.device("cuda"))
        z = model({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
        parts.append(F.normalize(z, p=2, dim=1).float().cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros((0, model.embedding_dim), dtype=np.float32)


def fit_readout(cfg, embeddings, labels, seed):
    return fit_logistic_on_embeddings(embeddings, labels, class_weight=cfg["supcon"]["classifier_class_weight"],
        seed=seed, classifier_overrides={"C": float(cfg["supcon"]["classifier_c"]),
                                     "penalty": "l2", "solver": "lbfgs"})


def classification_frame(readout, embeddings, dataset, indices):
    predictions = readout.predict(embeddings).astype(int)
    probs_raw = readout.predict_proba(embeddings)
    probs = np.zeros((len(indices), len(pilot.ROLES)), dtype=float)
    for column, role_id in enumerate(readout.classes_):
        probs[:, int(role_id)] = probs_raw[:, column]
    labels = dataset.label_ids[indices]
    frame = pd.DataFrame({"row_id": dataset.row_ids[indices], "accident_id": dataset.get_groups()[indices],
        "true_id": labels, "pred_id": predictions,
        "true_macro": [pilot.ROLES[i] for i in labels],
        "pred_macro": [pilot.ROLES[i] for i in predictions]})
    for role_id, role in enumerate(pilot.ROLES):
        frame[f"prob_{role}"] = probs[:, role_id]
    return frame


def supcon_cv_task(cfg, run_dir, task):
    grid = [float(v) for v in cfg["lambda_grid"]]
    lam_index, fold = divmod(task, int(cfg["n_folds"]))
    if not 0 <= lam_index < len(grid):
        raise ValueError(f"Invalid SupCon CV task: {task}")
    folds = np.load(run_dir / "source_folds.npz")
    train, validation = folds[f"train_{fold}"], folds[f"validation_{fold}"]
    dataset = pilot.source_dataset(cfg)
    if set(dataset.get_groups()[train]) & set(dataset.get_groups()[validation]):
        raise AssertionError("Accident leakage between training and validation")
    out_dir = run_dir / "supcon_cv" / f"lambda_{lam_index}_fold_{fold}"
    model = fit_supcon(cfg, dataset, train, lam=grid[lam_index], seed=int(cfg["cv_seed"]) + fold,
                       out_dir=out_dir)
    x_train = encode_supcon(model, dataset, train, cfg)
    readout = fit_readout(cfg, x_train, dataset.label_ids[train], int(cfg["cv_seed"]) + fold)
    x_val = encode_supcon(model, dataset, validation, cfg)
    result = {"lambda_index": lam_index, "lambda": grid[lam_index], "fold": fold,
              "n_train": len(train), "n_validation": len(validation),
              **pilot.metrics(classification_frame(readout, x_val, dataset, validation))}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def supcon_select(cfg, run_dir):
    rows = []
    for lam_index, lam in enumerate(cfg["lambda_grid"]):
        for fold in range(int(cfg["n_folds"])):
            path = run_dir / "supcon_cv" / f"lambda_{lam_index}_fold_{fold}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            if (int(row["lambda_index"]) != lam_index or int(row["fold"]) != fold
                    or float(row["lambda"]) != float(lam)):
                raise ValueError(f"Invalid SupCon CV result: {path}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["lambda_index", "lambda"], as_index=False).agg(
        mean_ba=("balanced_accuracy", "mean"), sd_ba=("balanced_accuracy", "std"))
    if not np.isfinite(summary["mean_ba"].to_numpy()).all():
        raise ValueError("Non-finite source CV score")
    summary = summary.sort_values(["mean_ba", "lambda"], ascending=[False, True])
    summary.to_csv(run_dir / "supcon_source_cv_summary.csv", index=False)
    selection = {"selected_lambda": float(summary.iloc[0]["lambda"]),
                 "selected_index": int(summary.iloc[0]["lambda_index"]),
                 "selection_metric": "mean_source_grouped_cv_balanced_accuracy",
                 "target_used_for_selection": False,
                 "no_source_signal": float(summary.iloc[0]["lambda"]) == 0.0}
    (run_dir / "supcon_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    summarize_loss_diagnostics(run_dir / "supcon_cv", run_dir / "supcon_cv_loss_diagnostics.csv")
    print(json.dumps(selection), flush=True)


def supcon_final_task(cfg, run_dir, task, manifest):
    n_seeds = int(cfg["n_seeds"])
    condition_index, seed_index = divmod(task, n_seeds)
    if condition_index not in (0, 1):
        raise ValueError(f"Invalid SupCon final task: {task}")
    selection = json.loads((run_dir / "supcon_selection.json").read_text(encoding="utf-8"))
    selected = float(selection["selected_lambda"])
    if condition_index == 1 and selected == 0:
        print("Source selected lambda=0: selected condition reuses the baseline predictions.", flush=True)
        return
    lam = 0.0 if condition_index == 0 else selected
    seed = int(manifest["final_seeds"][seed_index])
    out_dir = run_dir / "supcon_final" / f"condition_{condition_index}" / f"seed_{seed_index}"
    dataset = pilot.source_dataset(cfg)
    model = fit_supcon(cfg, dataset, np.arange(len(dataset)), lam=lam, seed=seed, out_dir=out_dir)
    x_train = encode_supcon(model, dataset, np.arange(len(dataset)), cfg)
    readout = fit_readout(cfg, x_train, dataset.label_ids, seed)
    def predict_target(target):
        indices = np.arange(len(target))
        embeddings = encode_supcon(model, target, indices, cfg)
        return classification_frame(readout, embeddings, target, indices)
    write_target_outputs(cfg, run_dir, out_dir, family="supcon_final", lam_index=condition_index,
        lam=lam, seed_index=seed_index, seed=seed, predict_fn=predict_target)


def analyze_family(cfg, run_dir, manifest, family):
    n_seeds = int(cfg["n_seeds"])
    if family == "ce_diagnostic":
        indices = list(range(len(cfg["lambda_grid"])))
        root = run_dir / "ce_diagnostic"
        directory = lambda i, s: root / f"lambda_{i}" / f"seed_{s}"
        selected_zero = False
    else:
        indices = [0, 1]
        root = run_dir / "supcon_final"
        directory = lambda i, s: root / f"condition_{i}" / f"seed_{s}"
        selected_zero = float(json.loads((run_dir / "supcon_selection.json").read_text(
            encoding="utf-8"))["selected_lambda"]) == 0.0
    frames = {}
    metric_rows = []
    for target in cfg["targets"]:
        reference = expected_hash = None
        for condition_index in indices:
            for seed_index in range(n_seeds):
                actual = 0 if selected_zero and condition_index == 1 else condition_index
                out_dir = directory(actual, seed_index)
                done_path = out_dir / "done.json"
                pred_path = out_dir / "predictions" / f"predictions_{target}.csv"
                if not done_path.is_file() or not pred_path.is_file():
                    raise FileNotFoundError(f"Missing completed {family} run: {out_dir}")
                done = json.loads(done_path.read_text(encoding="utf-8"))
                if (int(done["seed_index"]) != seed_index or int(done["lambda_index"]) != actual
                        or int(done["training_seed"]) != int(manifest["final_seeds"][seed_index])):
                    raise ValueError(f"Mismatched run manifest: {done_path}")
                expected_lam = (float(cfg["lambda_grid"][actual]) if family == "ce_diagnostic"
                                else 0.0 if actual == 0 else float(json.loads(
                                    (run_dir / "supcon_selection.json").read_text(encoding="utf-8"))["selected_lambda"]))
                if float(done["lambda"]) != expected_lam:
                    raise ValueError(f"Mismatched lambda: {done_path}")
                actual_hash = done["target_sha256"][target]
                if expected_hash is None:
                    expected_hash = actual_hash
                elif actual_hash != expected_hash:
                    raise ValueError(f"Target changed between runs: {target}")
                frame = pd.read_csv(pred_path)
                pilot.validate_prediction_frame(frame, pred_path)
                key = pilot.prediction_alignment_key(frame)
                if reference is None:
                    reference = key
                elif not key.equals(reference):
                    raise ValueError(f"Unpaired target predictions: {pred_path}")
                frames[target, condition_index, seed_index] = frame
                metric_rows.append({"family": family, "target": target, "condition_index": condition_index,
                                    "seed_index": seed_index, "lambda": expected_lam, **pilot.metrics(frame)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / f"{family}_target_metrics_all.csv", index=False)
    per_seed_ood = metrics.groupby(["condition_index", "seed_index"], as_index=False).agg(
        mean_target_ba=("balanced_accuracy", "mean"),
        worst_target_ba=("balanced_accuracy", "min"),
        mean_target_macro_f1=("macro_f1", "mean"))
    per_seed_ood.to_csv(run_dir / f"{family}_per_seed_ood.csv", index=False)
    per_seed_ood.groupby("condition_index", as_index=False).agg(
        mean_target_ba=("mean_target_ba", "mean"), seed_sd_mean_target_ba=("mean_target_ba", "std"),
        mean_worst_target_ba=("worst_target_ba", "mean"),
        seed_sd_worst_target_ba=("worst_target_ba", "std"),
        mean_target_macro_f1=("mean_target_macro_f1", "mean"),
    ).to_csv(run_dir / f"{family}_ood_summary.csv", index=False)
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    n_boot = int(cfg["n_bootstrap"])
    n_conditions = len(indices)
    summaries, contrasts, draws_by_target = [], [], {}
    for target in cfg["targets"]:
        accidents = sorted(frames[target, 0, 0]["accident_id"].astype(str).unique())
        counts = np.stack([np.stack([confusion_by_accident(frames[target, i, s], accidents)
                                      for s in range(n_seeds)]) for i in indices])
        draws = np.empty((n_boot, n_conditions), dtype=float)
        for start in range(0, n_boot, 64):
            end = min(start + 64, n_boot)
            weights = rng.multinomial(len(accidents), np.full(len(accidents), 1 / len(accidents)), size=end - start)
            totals = np.einsum("ba,msarc->bmsrc", weights, counts, optimize=True)
            draws[start:end] = balanced_accuracy_from_confusions(totals).mean(axis=2)
        draws_by_target[target] = draws
        for condition_index in indices:
            sub = metrics[(metrics.target == target) & (metrics.condition_index == condition_index)]
            scores = sub.balanced_accuracy.to_numpy()
            summaries.append({"target": target, "condition_index": condition_index,
                "lambda": float(sub["lambda"].iloc[0]), "mean_ba": scores.mean(), "seed_sd_ba": scores.std(ddof=1),
                "ci_low": np.quantile(draws[:, condition_index], .025),
                "ci_high": np.quantile(draws[:, condition_index], .975),
                "mean_macro_f1": sub.macro_f1.mean(), "seed_sd_macro_f1": sub.macro_f1.std(ddof=1),
                "mean_a1_precision": sub.a1_precision.mean(),
                "seed_sd_a1_precision": sub.a1_precision.std(ddof=1),
                "mean_a1_recall": sub.a1_recall.mean(),
                "seed_sd_a1_recall": sub.a1_recall.std(ddof=1),
                "a1_support": int(sub.a1_support.iloc[0])})
        pivot = metrics[metrics.target == target].pivot(index="seed_index", columns="condition_index", values="balanced_accuracy")
        for condition_index in indices[1:]:
            delta = draws[:, condition_index] - draws[:, 0]
            per_seed = pivot[condition_index] - pivot[0]
            contrasts.append({"target": target, "condition_index": condition_index,
                "lambda": float(metrics[metrics.condition_index == condition_index]["lambda"].iloc[0]),
                "mean_delta_ba": per_seed.mean(), "seed_sd_delta": per_seed.std(ddof=1),
                "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975)})
    combined = np.stack([draws_by_target[t] for t in cfg["targets"]]).mean(axis=0)
    per_seed_mean = metrics.groupby(["condition_index", "seed_index"], as_index=False).balanced_accuracy.mean()
    pivot = per_seed_mean.pivot(index="seed_index", columns="condition_index", values="balanced_accuracy")
    for condition_index in indices:
        scores = pivot[condition_index].to_numpy()
        summaries.append({"target": "equal_weight_mean", "condition_index": condition_index,
            "lambda": float(metrics[metrics.condition_index == condition_index]["lambda"].iloc[0]),
            "mean_ba": scores.mean(), "seed_sd_ba": scores.std(ddof=1),
            "ci_low": np.quantile(combined[:, condition_index], .025),
            "ci_high": np.quantile(combined[:, condition_index], .975)})
    for condition_index in indices[1:]:
        delta = combined[:, condition_index] - combined[:, 0]
        per_seed = pivot[condition_index] - pivot[0]
        contrasts.append({"target": "equal_weight_mean", "condition_index": condition_index,
            "lambda": float(metrics[metrics.condition_index == condition_index]["lambda"].iloc[0]),
            "mean_delta_ba": per_seed.mean(), "seed_sd_delta": per_seed.std(ddof=1),
            "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975)})
    pd.DataFrame(summaries).to_csv(run_dir / f"{family}_bootstrap_intervals.csv", index=False)
    pd.DataFrame(contrasts).to_csv(run_dir / f"{family}_paired_differences.csv", index=False)
    summarize_loss_diagnostics(root, run_dir / f"{family}_loss_diagnostics.csv")
    if family == "ce_diagnostic":
        (run_dir / "ce_diagnostic_NOT_FOR_SELECTION.txt").write_text(
            "These target evaluations are diagnostic only. Do not select lambda using target results.\n",
            encoding="utf-8")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


def summarize_loss_diagnostics(root, output):
    paths = sorted(root.glob("**/batch_diagnostics.csv"))
    if not paths:
        raise FileNotFoundError(f"No batch diagnostics in {root}")
    rows = []
    for path in paths:
        data = pd.read_csv(path)
        expected = {"epoch", "pgc", "weighted_pgc", "weighted_loss_ratio", "weighted_z_grad_ratio"}
        if not expected <= set(data.columns):
            raise ValueError(f"Incomplete batch diagnostics: {path}")
        if not np.isfinite(data[["pgc", "weighted_pgc", "weighted_loss_ratio",
                                 "weighted_z_grad_ratio"]].to_numpy()).all():
            raise ValueError(f"Non-finite batch diagnostic: {path}")
        for epoch, group in data.groupby("epoch"):
            rows.append({"run": str(path.parent.relative_to(root)), "epoch": int(epoch),
                "lambda": float(group["lambda"].iloc[0]), "sampled_batches": len(group),
                "pgc_mean": group.pgc.mean(), "pgc_p95": group.pgc.quantile(.95),
                "pgc_max": group.pgc.max(),
                "weighted_loss_ratio_mean": group.weighted_loss_ratio.mean(),
                "weighted_loss_ratio_p95": group.weighted_loss_ratio.quantile(.95),
                "weighted_z_grad_ratio_mean": group.weighted_z_grad_ratio.mean(),
                "weighted_z_grad_ratio_p95": group.weighted_z_grad_ratio.quantile(.95),
                "active_edge_fraction": (group.active_edges > 0).mean()})
    pd.DataFrame(rows).to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "ce", "ce_analyze", "supcon_cv",
                                           "supcon_select", "supcon_final", "supcon_analyze"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pgc_followup.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    cfg, run_dir = pilot.config_and_dir(args.config, args.run_dir)
    if args.phase == "preflight":
        preflight(cfg, run_dir, args.config)
        return
    manifest = pilot.require_manifest(cfg, run_dir, args.config)
    if args.phase in ("ce", "supcon_cv", "supcon_final") and args.task_id is None:
        parser.error("--task-id is required for GPU phases")
    if args.phase == "ce":
        ce_task(cfg, run_dir, args.task_id, manifest)
    elif args.phase == "ce_analyze":
        analyze_family(cfg, run_dir, manifest, "ce_diagnostic")
    elif args.phase == "supcon_cv":
        supcon_cv_task(cfg, run_dir, args.task_id)
    elif args.phase == "supcon_select":
        supcon_select(cfg, run_dir)
    elif args.phase == "supcon_final":
        supcon_final_task(cfg, run_dir, args.task_id, manifest)
    else:
        analyze_family(cfg, run_dir, manifest, "supcon_final")


if __name__ == "__main__":
    main()
