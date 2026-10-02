"""Aggregate the 2 x 2 x 4 backbone/source campaign after all jobs finish.

The script deliberately resamples accident identifiers, never factual units.
It reads only stored predictions and manifests, therefore does not refit a
model or expose target labels to source-only selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replication.runner import _model_config, load_replication_config, run_dir_for, training_seeds
from safer_core.embedding_paths import backbone_storage_id


ROLE_ORDER = ["A0", "A1", "B", "C"]
METHOD_ORDER = ["frozen", "cross_entropy", "supcon", "softtriple"]


def _task_seed(base_seed: int, stage: int, index: int) -> int:
    """Stable independent seed for one parallel bootstrap block."""
    sequence = np.random.SeedSequence([int(base_seed), int(stage), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def _parallel_map(tasks: list[tuple], worker, *, n_workers: int) -> list[Any]:
    if n_workers <= 1:
        return [worker(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(worker, tasks))


def _summary_task(task: tuple) -> dict[str, float]:
    seed_frames, seed, n_boot, confidence = task
    return _summary(seed_frames, rng=np.random.default_rng(seed), n_boot=n_boot, confidence=confidence)


def _multi_target_summary_task(task: tuple) -> dict[str, float]:
    by_target, seed, n_boot, confidence = task
    return _multi_target_summary(by_target, rng=np.random.default_rng(seed), n_boot=n_boot, confidence=confidence)


def _paired_bootstrap_task(task: tuple) -> np.ndarray:
    left_frames, right_frames, seed, n_boot = task
    return _paired_bootstrap_difference(left_frames, right_frames, rng=np.random.default_rng(seed), n_boot=n_boot)


def _accident_confusion_counts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Precompute one 4x4 confusion matrix for every accident.

    Bootstrap resampling then only sums these small matrices. This avoids
    rebuilding pandas DataFrames and calling sklearn for every draw.
    """
    accident_codes, _ = pd.factorize(frame["accident_id"].astype(str), sort=False)
    true_codes = pd.Categorical(frame["true_macro"].astype(str), categories=ROLE_ORDER).codes
    pred_codes = pd.Categorical(frame["pred_macro"].astype(str), categories=ROLE_ORDER).codes
    if np.any(true_codes < 0) or np.any(pred_codes < 0):
        raise ValueError("Unknown role in stored predictions")
    n_accidents = int(accident_codes.max()) + 1 if len(accident_codes) else 0
    counts = np.zeros((n_accidents, len(ROLE_ORDER), len(ROLE_ORDER)), dtype=np.int64)
    np.add.at(counts, (accident_codes, true_codes, pred_codes), 1)
    labels = pd.unique(frame["accident_id"].astype(str)).astype(str)
    return labels, counts


def _balanced_accuracy_from_confusion(counts: np.ndarray) -> float:
    row_totals = counts.sum(axis=1)
    valid = row_totals > 0
    if not np.any(valid):
        return float("nan")
    recalls = np.divide(
        np.diag(counts), row_totals,
        out=np.zeros(len(ROLE_ORDER), dtype=float), where=valid,
    )
    return float(recalls[valid].mean())


def _method(spec: dict[str, Any]) -> str:
    if spec["runner"] == "frozen":
        return "frozen"
    if spec["runner"] == "supervised_macro_ft":
        return "cross_entropy"
    return str(spec.get("method_name") or spec.get("method") or "").lower()


def _backbone(spec: dict[str, Any]) -> str:
    explicit = spec.get("backbone_id")
    if explicit:
        return str(explicit)
    return backbone_storage_id(str(spec.get("overrides", {}).get("model", {}).get("backbone_name", "unknown")))


def _bootstrap_mean_ba(seed_frames: list[pd.DataFrame], rng: np.random.Generator, n: int) -> np.ndarray:
    """Bootstrap accident IDs separately within each seed then average seeds."""
    values = np.empty(n, dtype=float)
    prepared = []
    for frame in seed_frames:
        labels, counts = _accident_confusion_counts(frame)
        prepared.append((np.arange(len(labels), dtype=np.int64), counts))
    for i in range(n):
        scores = []
        for ids, counts in prepared:
            sampled = rng.choice(ids, size=len(ids), replace=True)
            scores.append(_balanced_accuracy_from_confusion(counts[sampled].sum(axis=0)))
        values[i] = float(np.mean(scores))
    return values


def _summary(seed_frames: list[pd.DataFrame], *, rng: np.random.Generator, n_boot: int, confidence: float) -> dict[str, float]:
    seed_scores = [balanced_accuracy_score(f["true_macro"], f["pred_macro"]) for f in seed_frames]
    draws = _bootstrap_mean_ba(seed_frames, rng, n_boot)
    alpha = (1.0 - confidence) / 2.0
    return {"balanced_accuracy_mean": float(np.mean(seed_scores)), "balanced_accuracy_seed_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0, "ci_low": float(np.quantile(draws, alpha)), "ci_high": float(np.quantile(draws, 1 - alpha)), "n_seeds": len(seed_frames)}


def _multi_target_summary(by_target: dict[str, list[pd.DataFrame]], *, rng: np.random.Generator, n_boot: int, confidence: float) -> dict[str, float]:
    """Equal-weight target mean with accident-level bootstrap in every target."""
    target_summaries = [_summary(frames, rng=rng, n_boot=n_boot, confidence=confidence) for frames in by_target.values()]
    n_seeds = min(item["n_seeds"] for item in target_summaries)
    alpha = (1 - confidence) / 2
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = np.mean([_bootstrap_mean_ba(frames, rng, 1)[0] for frames in by_target.values()])
    return {"balanced_accuracy_mean": float(np.mean([x["balanced_accuracy_mean"] for x in target_summaries])), "balanced_accuracy_seed_sd": float(np.mean([x["balanced_accuracy_seed_sd"] for x in target_summaries])), "ci_low": float(np.quantile(draws, alpha)), "ci_high": float(np.quantile(draws, 1-alpha)), "n_seeds": n_seeds}


def _paired_bootstrap_difference(left_frames: list[pd.DataFrame], right_frames: list[pd.DataFrame], *, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Paired accident bootstrap; prediction rows have been alignment-checked."""
    draws = np.empty(n_boot)
    pairs = []
    for left, right in zip(left_frames, right_frames):
        left_labels, left_counts = _accident_confusion_counts(left)
        right_labels, right_counts = _accident_confusion_counts(right)
        right_index = {label: index for index, label in enumerate(right_labels)}
        if set(left_labels) != set(right_labels):
            raise ValueError("Paired predictions have different accident identifiers")
        right_counts = right_counts[[right_index[label] for label in left_labels]]
        if len(left_labels) != len(right_labels):
            raise ValueError("Paired predictions have different accident counts")
        pairs.append((np.arange(len(left_labels), dtype=np.int64), left_counts, right_counts))
    for draw in range(n_boot):
        diffs = []
        for ids, left_counts, right_counts in pairs:
            sampled = rng.choice(ids, len(ids), replace=True)
            left_ba = _balanced_accuracy_from_confusion(left_counts[sampled].sum(axis=0))
            right_ba = _balanced_accuracy_from_confusion(right_counts[sampled].sum(axis=0))
            diffs.append(left_ba - right_ba)
        draws[draw] = np.mean(diffs)
    return draws


def _load_runs(config_path: Path) -> tuple[dict[str, Any], dict[tuple[str, int, str], pd.DataFrame], list[dict[str, Any]]]:
    config = load_replication_config(config_path)
    frames: dict[tuple[str, int, str], pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    for model_id in config["models"]:
        spec = _model_config(config, model_id)
        source = str(spec.get("source_corpus", config["training"].get("source_corpus", "btp")))
        targets = [str(x) for x in spec.get("test_corpora", config["training"]["test_corpora"]) if str(x) != source]
        for seed in training_seeds(config):
            folder = run_dir_for(config, model_id, seed)
            manifest = folder / "run_manifest.json"
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "complete":
                continue
            for target in targets:
                path = folder / "predictions" / f"predictions_{target}.csv"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing completed prediction: {path}")
                frame = pd.read_csv(path)
                required = {"accident_id", "true_macro", "pred_macro"}
                if required - set(frame):
                    raise ValueError(f"Prediction columns missing in {path}: {sorted(required - set(frame))}")
                frames[(model_id, seed, target)] = frame
                records.append({"model_id": model_id, "source_corpus": source, "evaluation_corpus": target, "backbone": _backbone(spec), "method": _method(spec), "seed": seed, "path": str(path)})
    return config, frames, records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="output/replication_recipes/backbone_source_factorial.yaml")
    p.add_argument("--output", default="output/backbone_source_factorial_analysis")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-workers", type=int, default=1, help="Nombre de processus bootstrap parallèles.")
    args = p.parse_args()
    if args.n_workers < 1:
        raise ValueError("--n-workers doit être >= 1")
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config, frames, records = _load_runs(config_path)
    if not records:
        raise RuntimeError("No complete runs found.")
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "prediction_inventory.csv", index=False)
    confidence = float(config.get("bootstrap", {}).get("confidence_level", 0.95))
    bootstrap_seed = int(config.get("bootstrap", {}).get("seed", 2027))

    grouped: dict[tuple[str, str, str, str], list[pd.DataFrame]] = {}
    for r in records:
        grouped.setdefault((r["backbone"], r["source_corpus"], r["method"], r["evaluation_corpus"]), []).append(frames[(r["model_id"], r["seed"], r["evaluation_corpus"])])
    target_tasks = []
    target_keys = []
    for index, (key, seed_frames) in enumerate(grouped.items()):
        target_keys.append(key)
        target_tasks.append((seed_frames, _task_seed(bootstrap_seed, 1, index), args.n_bootstrap, confidence))
    target_summaries = _parallel_map(target_tasks, _summary_task, n_workers=args.n_workers)
    target_rows = []
    for (backbone, source, method, target), summary in zip(target_keys, target_summaries):
        target_rows.append({"backbone": backbone, "source_corpus": source, "method": method, "evaluation_corpus": target, **summary})
    target_summary = pd.DataFrame(target_rows)
    target_summary.to_csv(out / "ood_metrics_by_target.csv", index=False)

    ood_rows = []
    ood_tasks = []
    ood_keys = []
    ood_metadata = []
    ood_index = 0
    for (backbone, source, method), frame in target_summary.groupby(["backbone", "source_corpus", "method"], sort=False):
        model_id = pd.DataFrame(records).query("backbone == @backbone and source_corpus == @source and method == @method").model_id.iloc[0]
        for scope, targets in (("all_targets", list(frame.evaluation_corpus)), ("common_caou_nicollin", ["caou", "nicollin"])):
            selected = [t for t in targets if (model_id, training_seeds(config)[0], t) in frames]
            if len(selected) != len(targets):
                continue
            by_target = {t: [frames[(model_id, seed, t)] for seed in training_seeds(config) if (model_id, seed, t) in frames] for t in selected}
            ood_tasks.append((by_target, _task_seed(bootstrap_seed, 2, ood_index), args.n_bootstrap, confidence))
            ood_keys.append((backbone, source, method, scope, len(selected)))
            ood_index += 1
    ood_summaries = _parallel_map(ood_tasks, _multi_target_summary_task, n_workers=args.n_workers)
    for (backbone, source, method, scope, n_targets), summary in zip(ood_keys, ood_summaries):
        ood_rows.append({"backbone": backbone, "source_corpus": source, "method": method, "ood_scope": scope, "n_targets": n_targets, **summary})
    ood = pd.DataFrame(ood_rows)
    ood.to_csv(out / "ood_summary.csv", index=False)

    role_rows = []
    for record in records:
        frame = frames[(record["model_id"], record["seed"], record["evaluation_corpus"])]
        for role in ROLE_ORDER:
            actual = frame.true_macro.astype(str).eq(role)
            predicted = frame.pred_macro.astype(str).eq(role)
            tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum()); fn = int((actual & ~predicted).sum())
            role_rows.append({**{k: record[k] for k in ("model_id", "source_corpus", "evaluation_corpus", "backbone", "method", "seed")}, "role": role, "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan, "support": int(actual.sum())})
    roles = pd.DataFrame(role_rows)
    roles.to_csv(out / "per_role_metrics.csv", index=False)

    sns.set_theme(style="whitegrid")
    for target, table in roles.groupby("evaluation_corpus", sort=True):
        heat = table.groupby(["backbone", "source_corpus", "method", "role"], as_index=False).agg(recall=("recall", "mean"), support=("support", "mean"))
        heat["row"] = heat.backbone + " / " + heat.source_corpus + " / " + heat.method
        matrix = heat.pivot(index="row", columns="role", values="recall").reindex(columns=ROLE_ORDER) * 100
        labels = heat.pivot(index="row", columns="role", values="support").reindex(index=matrix.index, columns=ROLE_ORDER)
        annot = matrix.round(1).astype(str) + "\n(n=" + labels.fillna(0).astype(int).astype(str) + ")"
        fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(matrix))))
        sns.heatmap(matrix, annot=annot, fmt="", cmap="viridis", vmin=0, vmax=100, cbar_kws={"label": "Role recall (%)"}, ax=ax)
        ax.set(xlabel="Role", ylabel="Backbone / source / method", title=f"Per-role recall: {target}")
        fig.tight_layout(); fig.savefig(out / f"role_recall_{target}.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, scope in zip(axes, ["all_targets", "common_caou_nicollin"]):
        data = ood[ood.ood_scope.eq(scope)].copy(); data["series"] = data.backbone + " / " + data.source_corpus
        series_order = sorted(data["series"].unique())
        x = np.arange(len(METHOD_ORDER)); width = 0.78 / max(len(series_order), 1)
        for index, series in enumerate(series_order):
            part = data[data["series"].eq(series)].set_index("method").reindex(METHOD_ORDER)
            pos = x - 0.39 + width / 2 + index * width
            heights = part.balanced_accuracy_mean.to_numpy(dtype=float)
            lower = heights - part.ci_low.to_numpy(dtype=float); upper = part.ci_high.to_numpy(dtype=float) - heights
            bars = ax.bar(pos, heights, width, label=series, yerr=np.vstack([lower, upper]), capsize=3)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        ax.set_xticks(x, METHOD_ORDER)
        ax.set(title=scope.replace("_", " "), xlabel="Method", ylabel="OOD balanced accuracy")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ood_balanced_accuracy.png", dpi=220); plt.close(fig)

    paired_rows = []
    paired_tasks = []
    paired_metadata = []
    paired_index = 0
    for (backbone, source, target), data in pd.DataFrame(records).groupby(["backbone", "source_corpus", "evaluation_corpus"]):
        available = {row.method: row.model_id for row in data.drop_duplicates("method").itertuples()}
        for left, right in itertools.combinations(sorted(available), 2):
            left_frames = [frames[(available[left], seed, target)] for seed in training_seeds(config) if (available[left], seed, target) in frames]
            right_frames = [frames[(available[right], seed, target)] for seed in training_seeds(config) if (available[right], seed, target) in frames]
            if len(left_frames) != len(right_frames) or not left_frames:
                continue
            # Pair by seed, checking exact unit alignment before paired bootstrap.
            diffs = []
            for a, b in zip(left_frames, right_frames):
                keys = ["doc_id"] if "doc_id" in a and "doc_id" in b else ["accident_id", "true_macro"]
                aa = a.sort_values(keys).reset_index(drop=True); bb = b.sort_values(keys).reset_index(drop=True)
                if not aa[keys].equals(bb[keys]) or not aa.true_macro.equals(bb.true_macro):
                    raise ValueError(f"Predictions not aligned for paired comparison {left} vs {right} on {target}")
                diffs.append(balanced_accuracy_score(aa.true_macro, aa.pred_macro) - balanced_accuracy_score(bb.true_macro, bb.pred_macro))
            paired_tasks.append((left_frames, right_frames, _task_seed(bootstrap_seed, 3, paired_index), args.n_bootstrap))
            paired_metadata.append((backbone, source, target, left, right, float(np.mean(diffs)), len(diffs)))
            paired_index += 1
    paired_draws = _parallel_map(paired_tasks, _paired_bootstrap_task, n_workers=args.n_workers)
    alpha = (1 - confidence) / 2
    for (backbone, source, target, left, right, mean_difference, n_seeds), draws in zip(paired_metadata, paired_draws):
        paired_rows.append({"backbone": backbone, "source_corpus": source, "evaluation_corpus": target, "method_left": left, "method_right": right, "balanced_accuracy_difference": mean_difference, "ci_low": float(np.quantile(draws, alpha)), "ci_high": float(np.quantile(draws, 1-alpha)), "n_seeds": n_seeds})
    pd.DataFrame(paired_rows).to_csv(out / "paired_method_differences.csv", index=False)


if __name__ == "__main__":
    main()
