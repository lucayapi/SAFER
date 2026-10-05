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


def _all_pairwise_bootstrap_task(task: tuple) -> dict[tuple[str, str], np.ndarray]:
    method_frames, seed, n_boot = task
    methods = list(method_frames)
    frames = [frame for method in methods for frame in method_frames[method]]
    _, _, prepared = _aligned_accident_counts(frames)
    scores = _balanced_accuracy_from_totals(
        _bootstrap_confusion_totals(prepared, np.random.default_rng(seed), n_boot)
    ).reshape(n_boot, len(methods), -1).mean(axis=2)
    return {
        (left, right): scores[:, i] - scores[:, j]
        for i, left in enumerate(methods)
        for j, right in enumerate(methods)
        if i < j
    }


def _accident_confusion_counts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Precompute one 4x4 confusion matrix for every accident.

    Bootstrap resampling then only sums these small matrices. This avoids
    rebuilding pandas DataFrames and calling sklearn for every draw.
    """
    accident_codes, accident_ids = pd.factorize(frame["accident_id"].astype(str), sort=True)
    true_codes = pd.Categorical(frame["true_macro"].astype(str), categories=ROLE_ORDER).codes
    pred_codes = pd.Categorical(frame["pred_macro"].astype(str), categories=ROLE_ORDER).codes
    if np.any(true_codes < 0) or np.any(pred_codes < 0):
        raise ValueError("Unknown role in stored predictions")
    n_accidents = int(accident_codes.max()) + 1 if len(accident_codes) else 0
    counts = np.zeros((n_accidents, len(ROLE_ORDER), len(ROLE_ORDER)), dtype=np.int64)
    np.add.at(counts, (accident_codes, true_codes, pred_codes), 1)
    return accident_ids.astype(str), counts


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


def _method(spec: dict[str, Any], model_id: str | None = None) -> str:
    if spec["runner"] == "frozen":
        return "frozen"
    if spec["runner"] == "supervised_macro_ft":
        return "cross_entropy"
    method = str(spec.get("method_name") or spec.get("method") or "").lower()
    if not method:
        # Replication templates inherit the method from the base YAML in the
        # training runner, but the resolved recipe may not expose that field.
        # Recover it from the base config or, as a final fallback, the model id.
        base_name = Path(str(spec.get("base_config", ""))).stem.lower()
        if base_name in {"supcon", "softtriple"}:
            method = base_name
        elif model_id:
            suffix = str(model_id).lower().rsplit("_", 1)[-1]
            if suffix in {"supcon", "softtriple"}:
                method = suffix
    aliases = {"supervised_contrastive": "supcon", "soft_triple": "softtriple"}
    return aliases.get(method, method)


def _backbone(spec: dict[str, Any]) -> str:
    explicit = spec.get("backbone_id")
    if explicit:
        return str(explicit)
    return backbone_storage_id(str(spec.get("overrides", {}).get("model", {}).get("backbone_name", "unknown")))


def _aligned_accident_counts(frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return per-accident confusion matrices in a shared accident order."""
    if not frames:
        raise ValueError("Bootstrap requires at least one prediction frame")
    reference_labels, reference_counts = _accident_confusion_counts(frames[0])
    prepared = [reference_counts]
    for frame in frames[1:]:
        labels, counts = _accident_confusion_counts(frame)
        if len(labels) != len(reference_labels) or set(labels) != set(reference_labels):
            raise ValueError("Prediction frames have different accident identifiers")
        prepared.append(counts[[{label: index for index, label in enumerate(labels)}[label] for label in reference_labels]])
    return reference_labels, np.arange(len(reference_labels), dtype=np.int64), prepared


def _bootstrap_confusion_totals(
    prepared: list[np.ndarray], rng: np.random.Generator, n: int
) -> np.ndarray:
    """Compute cluster-bootstrap totals in batches using multinomial weights."""
    # A multinomial draw is equivalent to sampling n accident IDs with
    # replacement, but avoids repeatedly materialising full confusion matrices.
    n_accidents = prepared[0].shape[0]
    probability = np.full(n_accidents, 1.0 / n_accidents)
    statistics = np.stack([
        np.stack((counts.sum(axis=2), np.diagonal(counts, axis1=1, axis2=2)), axis=-1)
        for counts in prepared
    ])
    totals = np.empty((n, len(prepared), len(ROLE_ORDER), 2), dtype=np.float64)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        weights = rng.multinomial(n_accidents, probability, size=stop - start)
        totals[start:stop] = np.einsum("ba,sarc->bsrc", weights, statistics, optimize=True)
    return totals


def _balanced_accuracy_from_totals(totals: np.ndarray) -> np.ndarray:
    supports = totals[..., 0]
    true_positives = totals[..., 1]
    valid = supports > 0
    recalls = np.divide(true_positives, supports, out=np.zeros_like(true_positives), where=valid)
    return np.divide(
        (recalls * valid).sum(axis=-1), valid.sum(axis=-1),
        out=np.full(valid.shape[:-1], np.nan, dtype=float), where=valid.sum(axis=-1) > 0,
    )


def _bootstrap_mean_ba(seed_frames: list[pd.DataFrame], rng: np.random.Generator, n: int) -> np.ndarray:
    """Bootstrap target accidents once per draw, shared across training seeds."""
    _, ids, prepared = _aligned_accident_counts(seed_frames)
    del ids  # The number of sampled clusters is the number in the target corpus.
    return _balanced_accuracy_from_totals(_bootstrap_confusion_totals(prepared, rng, n)).mean(axis=1)


def _summary(seed_frames: list[pd.DataFrame], *, rng: np.random.Generator, n_boot: int, confidence: float) -> dict[str, float]:
    seed_scores = [balanced_accuracy_score(f["true_macro"], f["pred_macro"]) for f in seed_frames]
    draws = _bootstrap_mean_ba(seed_frames, rng, n_boot)
    alpha = (1.0 - confidence) / 2.0
    return {"balanced_accuracy_mean": float(np.mean(seed_scores)), "balanced_accuracy_seed_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0, "ci_low": float(np.quantile(draws, alpha)), "ci_high": float(np.quantile(draws, 1 - alpha)), "n_seeds": len(seed_frames)}


def _multi_target_summary(by_target: dict[str, list[pd.DataFrame]], *, rng: np.random.Generator, n_boot: int, confidence: float) -> dict[str, float]:
    """Equal-weight target mean with accident-level bootstrap in every target."""
    seed_counts = {len(frames) for frames in by_target.values()}
    if len(seed_counts) != 1 or not seed_counts or 0 in seed_counts:
        raise ValueError("Equal-target summaries require matching nonempty seed lists")
    n_seeds = seed_counts.pop()
    # Lists are built in the same training_seeds(config) order by the caller.
    # Average targets within each run before computing between-run dispersion.
    per_seed = np.mean([
        [balanced_accuracy_score(f["true_macro"], f["pred_macro"]) for f in frames]
        for frames in by_target.values()
    ], axis=0)
    alpha = (1 - confidence) / 2
    target_draws = [_bootstrap_mean_ba(frames, rng, n_boot) for frames in by_target.values()]
    draws = np.mean(target_draws, axis=0)
    return {"balanced_accuracy_mean": float(np.mean(per_seed)), "balanced_accuracy_seed_sd": float(np.std(per_seed, ddof=1)) if n_seeds > 1 else 0.0, "ci_low": float(np.quantile(draws, alpha)), "ci_high": float(np.quantile(draws, 1-alpha)), "n_seeds": n_seeds}


def _paired_bootstrap_difference(left_frames: list[pd.DataFrame], right_frames: list[pd.DataFrame], *, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Paired bootstrap with one shared accident sample across methods and seeds."""
    if len(left_frames) != len(right_frames) or not left_frames:
        raise ValueError("Paired bootstrap requires matching nonempty seed lists")
    left_labels, _, left_counts = _aligned_accident_counts(left_frames)
    right_labels, _, right_counts = _aligned_accident_counts(right_frames)
    if len(left_labels) != len(right_labels) or set(left_labels) != set(right_labels):
        raise ValueError("Paired methods have different accident identifiers")
    right_index = {label: index for index, label in enumerate(right_labels)}
    right_counts = [counts[[right_index[label] for label in left_labels]] for counts in right_counts]
    scores = _balanced_accuracy_from_totals(
        _bootstrap_confusion_totals([*left_counts, *right_counts], rng, n_boot)
    )
    n_seeds = len(left_counts)
    return (scores[:, :n_seeds] - scores[:, n_seeds:]).mean(axis=1)


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
                records.append({"model_id": model_id, "source_corpus": source, "evaluation_corpus": target, "backbone": _backbone(spec), "method": _method(spec, model_id), "seed": seed, "path": str(path)})
    return config, frames, records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="output/replication_recipes/backbone_source_factorial.yaml")
    p.add_argument("--output", default="output/backbone_source_factorial_analysis")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-workers", type=int, default=1, help="Nombre de processus bootstrap parallèles.")
    p.add_argument("--reuse-target-summary", action="store_true", help="Reuse existing target-level bootstrap results and recompute only multi-target and paired summaries.")
    p.add_argument("--reuse-ood-summary", action="store_true", help="Reuse the existing multi-target summary and recompute only paired contrasts and outputs.")
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
    target_summary_path = out / "ood_metrics_by_target.csv"
    if args.reuse_target_summary:
        if not target_summary_path.is_file():
            raise FileNotFoundError(f"Cannot reuse missing target-level summary: {target_summary_path}")
        target_summary = pd.read_csv(target_summary_path)
    else:
        target_tasks = []
        target_keys = []
        target_order = sorted({key[3] for key in grouped})
        for (key, seed_frames) in grouped.items():
            target_keys.append(key)
            # Reuse each target's accident resamples across all model conditions.
            target_tasks.append((seed_frames, _task_seed(bootstrap_seed, 1, target_order.index(key[3])), args.n_bootstrap, confidence))
        target_summaries = _parallel_map(target_tasks, _summary_task, n_workers=args.n_workers)
        target_rows = []
        for (backbone, source, method, target), summary in zip(target_keys, target_summaries):
            target_rows.append({"backbone": backbone, "source_corpus": source, "method": method, "evaluation_corpus": target, **summary})
        target_summary = pd.DataFrame(target_rows)
        target_summary.to_csv(target_summary_path, index=False)

    ood_path = out / "ood_summary.csv"
    if args.reuse_ood_summary:
        if not ood_path.is_file():
            raise FileNotFoundError(f"Cannot reuse missing multi-target summary: {ood_path}")
        ood = pd.read_csv(ood_path)
    else:
        ood_rows = []
        ood_tasks = []
        ood_keys = []
        for (backbone, source, method), frame in target_summary.groupby(["backbone", "source_corpus", "method"], sort=False):
            model_id = pd.DataFrame(records).query("backbone == @backbone and source_corpus == @source and method == @method").model_id.iloc[0]
            for scope, targets in (("all_targets", list(frame.evaluation_corpus)), ("common_caou_nicollin", ["caou", "nicollin"])):
                selected = [t for t in targets if (model_id, training_seeds(config)[0], t) in frames]
                if len(selected) != len(targets):
                    continue
                by_target = {t: [frames[(model_id, seed, t)] for seed in training_seeds(config) if (model_id, seed, t) in frames] for t in selected}
                # Identical target scopes share bootstrap streams across conditions.
                scope_index = 0 if scope == "all_targets" else 1
                ood_tasks.append((by_target, _task_seed(bootstrap_seed, 2, scope_index), args.n_bootstrap, confidence))
                ood_keys.append((backbone, source, method, scope, len(selected)))
        ood_summaries = _parallel_map(ood_tasks, _multi_target_summary_task, n_workers=args.n_workers)
        for (backbone, source, method, scope, n_targets), summary in zip(ood_keys, ood_summaries):
            ood_rows.append({"backbone": backbone, "source_corpus": source, "method": method, "ood_scope": scope, "n_targets": n_targets, **summary})
        ood = pd.DataFrame(ood_rows)
        ood.to_csv(ood_path, index=False)

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
    short_backbone = {"multilingual_e5_large": "E5", "qwen3": "Qwen3"}
    short_source = {"btp": "BTP", "metallurgie": "MET"}
    short_method = {"frozen": "F", "cross_entropy": "CE", "supcon": "SC", "softtriple": "ST"}
    for target, table in roles.groupby("evaluation_corpus", sort=True):
        heat = table.groupby(["backbone", "source_corpus", "method", "role"], as_index=False).agg(recall=("recall", "mean"), support=("support", "mean"))
        heat["row"] = (
            heat.backbone.map(short_backbone).fillna(heat.backbone)
            + " / " + heat.source_corpus.map(short_source).fillna(heat.source_corpus)
            + " / " + heat.method.map(short_method).fillna(heat.method)
        )
        matrix = heat.pivot(index="row", columns="role", values="recall").reindex(columns=ROLE_ORDER) * 100
        annot = matrix.apply(lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value:.1f}"))
        fig, ax = plt.subplots(figsize=(7.5, max(4, 0.36 * len(matrix))))
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
    paired_group_keys = []
    paired_groups = sorted(set((r["backbone"], r["source_corpus"], r["evaluation_corpus"]) for r in records))
    paired_group_index = {key: index for index, key in enumerate(paired_groups)}
    for (backbone, source, target), data in pd.DataFrame(records).groupby(["backbone", "source_corpus", "evaluation_corpus"]):
        available = {row.method: row.model_id for row in data.drop_duplicates("method").itertuples()}
        method_frames: dict[str, list[pd.DataFrame]] = {}
        method_scores: dict[str, list[float]] = {}
        for method, model_id in sorted(available.items()):
            run_frames = [frames[(model_id, seed, target)] for seed in training_seeds(config) if (model_id, seed, target) in frames]
            if run_frames:
                method_frames[method] = run_frames
                method_scores[method] = [balanced_accuracy_score(f.true_macro, f.pred_macro) for f in run_frames]
        if len(method_frames) < 2:
            continue
        # Check alignment once across every method and seed in this group.
        reference = method_frames[next(iter(method_frames))]
        for method, runs in method_frames.items():
            if len(runs) != len(reference):
                raise ValueError(f"Unequal seed count for {method} on {target}")
            for seed_index, (a, b) in enumerate(zip(reference, runs)):
                keys = ["doc_id"] if "doc_id" in a and "doc_id" in b else ["accident_id", "true_macro"]
                aa = a.sort_values(keys).reset_index(drop=True); bb = b.sort_values(keys).reset_index(drop=True)
                if not aa[keys].equals(bb[keys]) or not aa.true_macro.equals(bb.true_macro):
                    raise ValueError(f"Predictions not aligned for {method} at seed index {seed_index} on {target}")
        paired_tasks.append((method_frames, _task_seed(bootstrap_seed, 3, paired_group_index[(backbone, source, target)]), args.n_bootstrap))
        paired_group_keys.append((backbone, source, target))
        for left, right in itertools.combinations(method_frames, 2):
            left_scores, right_scores = method_scores[left], method_scores[right]
            if len(left_scores) != len(right_scores):
                continue
            mean_difference = float(np.mean(np.asarray(left_scores) - np.asarray(right_scores)))
            paired_metadata.append((backbone, source, target, left, right, mean_difference, len(left_scores)))
    paired_group_draws = _parallel_map(paired_tasks, _all_pairwise_bootstrap_task, n_workers=args.n_workers)
    draws_by_group = dict(zip(paired_group_keys, paired_group_draws))
    alpha = (1 - confidence) / 2
    for backbone, source, target, left, right, mean_difference, n_seeds in paired_metadata:
        draws = draws_by_group[(backbone, source, target)][(left, right)]
        paired_rows.append({"backbone": backbone, "source_corpus": source, "evaluation_corpus": target, "method_left": left, "method_right": right, "balanced_accuracy_difference": mean_difference, "ci_low": float(np.quantile(draws, alpha)), "ci_high": float(np.quantile(draws, 1-alpha)), "n_seeds": n_seeds})
    pd.DataFrame(paired_rows).to_csv(out / "paired_method_differences.csv", index=False)


if __name__ == "__main__":
    main()
