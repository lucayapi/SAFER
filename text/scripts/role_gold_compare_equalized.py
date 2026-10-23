#!/usr/bin/env python3
"""Compare volume-matched single-source and five-source GOLD campaigns.

Confidence intervals use paired cluster bootstrap resampling of test narratives
(`accident_id`). They are conditional on the trained checkpoints and seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CORPORA = ("btp", "metallurgie", "caou_chimie_plas", "bois_textille", "service_temporaire", "nicollin")
METHODS = ("frozen", "ce", "supcon", "softtriple")
ROLES = ("A0", "A1", "B", "C", "AUTRE")
ROLE_TO_ID = {role: i for i, role in enumerate(ROLES)}
ROOT = Path(__file__).resolve().parents[2]


def read_arm(root: Path, experiment: str) -> tuple[pd.DataFrame, dict]:
    result = root / experiment / "results"
    manifest_path = result / "manifest.json"
    scores_path = result / "scores_by_transfer.csv"
    if not manifest_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(f"Campaign results incomplete: {result}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = 300 if experiment.startswith("single") else 60
    if manifest.get("experiment") != experiment or manifest.get("rows") != expected_rows:
        raise ValueError(f"Unexpected/incomplete summary manifest: {manifest_path}")
    return pd.read_csv(scores_path), manifest


def _prediction_path(root: Path, experiment: str, fold: str, method: str,
                     seed: int | None, target: str) -> Path:
    if method == "frozen":
        base = root / experiment / "frozen" / fold
    else:
        if seed is None:
            raise ValueError("Trained prediction requires a seed")
        base = root / experiment / "trained" / fold / method / f"seed_{seed}"
    return base / f"predictions_{target}.csv"


def _load_predictions(path: Path) -> pd.DataFrame:
    metrics_path = path.with_name(path.name.replace("predictions_", "metrics_").replace(".csv", ".json"))
    if not path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Prediction or metric file is missing: {path}")
    expected_hash = json.loads(metrics_path.read_text(encoding="utf-8")).get("predictions_sha256")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"Prediction hash does not match its metric record: {path}")
    frame = pd.read_csv(path, dtype={"accident_id": str, "doc_id": str})
    required = {"accident_id", "doc_id", "pred_label", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction file lacks {sorted(required - set(frame.columns))}: {path}")
    if frame.doc_id.duplicated().any():
        raise ValueError(f"Duplicate doc_id in predictions: {path}")
    if not frame.pred_label.isin(ROLES).all() or not frame.prediction.isin(ROLES).all():
        raise ValueError(f"Unexpected role label in predictions: {path}")
    return frame


def _cluster_confusion(frame: pd.DataFrame, accident_order: list[str]) -> np.ndarray:
    """Return one 5x5 gold/prediction confusion matrix per accident."""
    group_to_index = {accident: i for i, accident in enumerate(accident_order)}
    if set(frame.accident_id.astype(str)) - set(group_to_index):
        raise ValueError("Prediction frame contains an unaligned accident_id")
    group = frame.accident_id.astype(str).map(group_to_index).to_numpy(dtype=int)
    gold = frame.pred_label.map(ROLE_TO_ID).to_numpy(dtype=int)
    pred = frame.prediction.map(ROLE_TO_ID).to_numpy(dtype=int)
    counts = np.zeros((len(accident_order), len(ROLES), len(ROLES)), dtype=np.int64)
    np.add.at(counts, (group, gold, pred), 1)
    return counts


def _metric_from_confusion(confusion: np.ndarray, metric: str) -> np.ndarray:
    """Compute BA or five-class macro-F1 over the final two confusion axes."""
    support = confusion.sum(axis=-1)
    recall = np.divide(np.diagonal(confusion, axis1=-2, axis2=-1), support,
                       out=np.zeros_like(support, dtype=float), where=support > 0)
    if metric == "ba":
        return recall.mean(axis=-1)
    predicted = confusion.sum(axis=-2)
    precision = np.divide(np.diagonal(confusion, axis1=-2, axis2=-1), predicted,
                          out=np.zeros_like(predicted, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return f1.mean(axis=-1)


def _bootstrap_target(single_by_source: dict[str, list[pd.DataFrame]],
                      lodo_frames: list[pd.DataFrame], *, n_boot: int,
                      seed: int) -> dict[str, np.ndarray | dict[str, float]]:
    """Paired accident-cluster bootstrap for both metrics and both arms."""
    all_frames = [f for source in sorted(single_by_source)
                  for f in single_by_source[source]] + lodo_frames
    if not all_frames or not single_by_source or not lodo_frames or n_boot < 1:
        raise ValueError("Bootstrap needs predictions for both arms and positive n_boot")
    reference = all_frames[0].sort_values("doc_id").reset_index(drop=True)
    key = reference[["doc_id", "accident_id", "pred_label"]].astype(str)
    accident_order = sorted(reference.accident_id.astype(str).unique())
    clusters: list[np.ndarray] = []
    for frame in all_frames:
        current = frame.sort_values("doc_id").reset_index(drop=True)
        check = current[["doc_id", "accident_id", "pred_label"]].astype(str)
        if not key.equals(check):
            raise ValueError("Predictions differ in tested documents, accident groups, or GOLD labels")
        clusters.append(_cluster_confusion(current, accident_order))

    rng = np.random.default_rng(seed)
    n_groups = len(accident_order)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    confusion = np.stack([np.einsum("bg,gij->bij", weights, count, optimize=True)
                          for count in clusters], axis=1)  # draw x model x role x role

    n_single_models = sum(map(len, single_by_source.values()))
    output: dict[str, np.ndarray | dict[str, float]] = {}
    for metric in ("ba", "macro_f1"):
        model_values = _metric_from_confusion(confusion, metric)
        idx = 0
        source_draws = []
        source_points = []
        for source in sorted(single_by_source):
            n_seed = len(single_by_source[source])
            values = model_values[:, idx:idx + n_seed]
            source_draws.append(values.mean(axis=1))
            source_points.append(float(np.mean([_metric_from_confusion(c.sum(axis=0), metric)
                                                for c in clusters[idx:idx + n_seed]])))
            idx += n_seed
        single_draw = np.stack(source_draws, axis=1).mean(axis=1)
        single_point = float(np.mean(source_points))
        lodo_slice = model_values[:, n_single_models:]
        lodo_draw = lodo_slice.mean(axis=1)
        lodo_point = float(np.mean([_metric_from_confusion(c.sum(axis=0), metric)
                                    for c in clusters[n_single_models:]]))
        output[metric] = {"single": single_draw, "lodo": lodo_draw,
                          "delta": lodo_draw - single_draw,
                          "single_point": single_point, "lodo_point": lodo_point,
                          "delta_point": lodo_point - single_point}
    return output


def _ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def _collect_target(root: Path, target: str, method: str) -> tuple[dict[str, list[pd.DataFrame]], list[pd.DataFrame]]:
    single: dict[str, list[pd.DataFrame]] = {}
    for source in CORPORA:
        if source == target:
            continue
        if method == "frozen":
            seeds: list[int | None] = [None]
        else:
            seeds = [42, 1729, 20261018]
        single[source] = [_load_predictions(_prediction_path(root, "single_source_equalized", source,
                                                               method, item_seed, target))
                         for item_seed in seeds]
    lodo_seeds: list[int | None] = [None] if method == "frozen" else [42, 1729, 20261018]
    lodo = [_load_predictions(_prediction_path(root, "lodo_equalized", target, method, item_seed, target))
            for item_seed in lodo_seeds]
    return single, lodo


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(os.getenv("ROLE_GOLD_RUN_ROOT", Path(os.getenv("SCRATCH", str(ROOT))) /
                                  "safer_role_gold_v3_equalized"))
    parser.add_argument("--run-root", type=Path, default=default_root)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be strictly between 0 and 1")
    root = args.run_root.expanduser().resolve()
    single, single_manifest = read_arm(root, "single_source_equalized")
    multi, multi_manifest = read_arm(root, "lodo_equalized")
    if single_manifest.get("gold_sha256") != multi_manifest.get("gold_sha256"):
        raise ValueError("Single-source and LODO summaries do not use the same GOLD snapshot")

    point_rows = []
    bootstrap_by_target: dict[tuple[str, str], dict] = {}
    bootstrap_rows = []
    for method_index, method in enumerate(METHODS):
        for target_index, target in enumerate(CORPORA):
            s = single[(single.method == method) & (single.target == target)]
            m = multi[(multi.method == method) & (multi.target == target)]
            expected_sources = set(CORPORA) - {target}
            if set(s.source) != expected_sources or len(m) != 1 or m.iloc[0].source != "five_sources":
                raise ValueError(f"Missing single/multisource transfer cells for {method}/{target}")
            point = {
                "method": method, "target": target,
                "single_source_ba": float(s.ba_mean.mean()),
                "single_source_macro_f1": float(s.macro_f1_mean.mean()),
                "lodo_ba": float(m.iloc[0].ba_mean),
                "lodo_macro_f1": float(m.iloc[0].macro_f1_mean),
                "delta_lodo_minus_single_ba": float(m.iloc[0].ba_mean - s.ba_mean.mean()),
                "delta_lodo_minus_single_macro_f1": float(m.iloc[0].macro_f1_mean - s.macro_f1_mean.mean()),
                "n_single_source_models": int(s.n_seeds.sum()),
                "n_lodo_seeds": int(m.iloc[0].n_seeds),
            }
            single_preds, lodo_preds = _collect_target(root, target, method)
            boot = _bootstrap_target(single_preds, lodo_preds,
                                     n_boot=args.n_bootstrap,
                                     seed=args.seed + target_index)
            bootstrap_by_target[(method, target)] = boot
            for metric, score_name, delta_name in (("ba", "balanced_accuracy", "delta_lodo_minus_single_ba"),
                                                    ("macro_f1", "macro_f1", "delta_lodo_minus_single_macro_f1")):
                values = boot[metric]
                assert isinstance(values, dict)
                if not (np.isclose(values["single_point"], point[f"single_source_{'ba' if metric == 'ba' else 'macro_f1'}"], atol=1e-8)
                        and np.isclose(values["lodo_point"], point[f"lodo_{'ba' if metric == 'ba' else 'macro_f1'}"], atol=1e-8)):
                    raise ValueError(f"Stored scores disagree with test predictions for {method}/{target}/{score_name}")
                single_ci = _ci(values["single"], args.confidence)
                lodo_ci = _ci(values["lodo"], args.confidence)
                delta_ci = _ci(values["delta"], args.confidence)
                point[f"single_source_{score_name}_ci_low"] = single_ci[0]
                point[f"single_source_{score_name}_ci_high"] = single_ci[1]
                point[f"lodo_{score_name}_ci_low"] = lodo_ci[0]
                point[f"lodo_{score_name}_ci_high"] = lodo_ci[1]
                point[f"{delta_name}_ci_low"] = delta_ci[0]
                point[f"{delta_name}_ci_high"] = delta_ci[1]
                bootstrap_rows.append({"method": method, "target": target, "metric": score_name,
                    "single_source_point": values["single_point"], "single_source_ci_low": single_ci[0],
                    "single_source_ci_high": single_ci[1], "lodo_point": values["lodo_point"],
                    "lodo_ci_low": lodo_ci[0], "lodo_ci_high": lodo_ci[1],
                    "delta_lodo_minus_single_point": values["delta_point"],
                    "delta_ci_low": delta_ci[0], "delta_ci_high": delta_ci[1],
                    "n_bootstrap": args.n_bootstrap, "cluster": "accident_id"})
            point_rows.append(point)

    frame = pd.DataFrame(point_rows)
    # Macro-domain intervals average the independently resampled domain scores,
    # using the same within-domain draw for both experimental arms.
    macro_rows = []
    for method in METHODS:
        for metric, metric_name in (("ba", "balanced_accuracy"), ("macro_f1", "macro_f1")):
            values = [bootstrap_by_target[(method, target)][metric] for target in CORPORA]
            singles = np.mean(np.stack([v["single"] for v in values]), axis=0)
            lodos = np.mean(np.stack([v["lodo"] for v in values]), axis=0)
            deltas = lodos - singles
            point_rows_for_method = frame[frame.method.eq(method)]
            col_single = "single_source_ba" if metric == "ba" else "single_source_macro_f1"
            col_lodo = "lodo_ba" if metric == "ba" else "lodo_macro_f1"
            point_single = float(point_rows_for_method[col_single].mean())
            point_lodo = float(point_rows_for_method[col_lodo].mean())
            ci_single, ci_lodo, ci_delta = (_ci(singles, args.confidence), _ci(lodos, args.confidence),
                                             _ci(deltas, args.confidence))
            macro_rows.append({"method": method, "metric": metric_name,
                "single_source_macro_domain": point_single, "single_ci_low": ci_single[0],
                "single_ci_high": ci_single[1], "lodo_macro_domain": point_lodo,
                "lodo_ci_low": ci_lodo[0], "lodo_ci_high": ci_lodo[1],
                "delta_lodo_minus_single": point_lodo - point_single,
                "delta_ci_low": ci_delta[0], "delta_ci_high": ci_delta[1],
                "n_bootstrap": args.n_bootstrap, "cluster": "accident_id within each fixed domain"})

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "equalized_single_vs_lodo_by_target.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(out / "equalized_single_vs_lodo_macro.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "equalized_bootstrap_by_target.csv", index=False)
    (out / "equalized_bootstrap_manifest.json").write_text(json.dumps({
        "method": "paired cluster bootstrap of test narratives",
        "cluster": "accident_id",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "confidence_level": args.confidence,
        "metrics": ["balanced_accuracy", "macro_f1"],
        "classes": list(ROLES),
        "interpretation": "Intervals quantify test-narrative sampling uncertainty conditional on the fitted checkpoints and fixed training seeds; they do not capture training-seed or domain-population uncertainty.",
    }, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Volume-matched GOLD role classification",
        "",
        "Each model used 2,400 training units (480 per role) and 375 validation units (75 per role). "
        "Single-source draws all units from one source; multisource uses 96 per role from each of five sources. "
        "Tests are the fixed RQ1 test narratives. Differences reflect the source-domain composition and number of domains; "
        "the experiment does not establish a causal effect of source count alone.",
        "",
        f"Intervals are paired {args.confidence:.0%} cluster-bootstrap CIs over `accident_id` ({args.n_bootstrap} resamples). "
        "They are conditional on the trained checkpoints and fixed seeds; seed variability is shown separately in the campaign summaries.",
        "",
        "## Macro-domain comparison",
        "",
        f"| Method | Metric | Single-source ({args.confidence:.0%} CI) | LODO ({args.confidence:.0%} CI) | LODO − single ({args.confidence:.0%} CI) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in macro_rows:
        lines.append(f"| {row['method']} | {row['metric']} | {row['single_source_macro_domain']:.4f} "
                     f"[{row['single_ci_low']:.4f}, {row['single_ci_high']:.4f}] | "
                     f"{row['lodo_macro_domain']:.4f} [{row['lodo_ci_low']:.4f}, {row['lodo_ci_high']:.4f}] | "
                     f"{row['delta_lodo_minus_single']:+.4f} [{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] |")
    lines += ["", "## Target-level results", "",
        "Detailed target CIs are in `equalized_bootstrap_by_target.csv`; point estimates and CIs are also in `equalized_single_vs_lodo_by_target.csv`.",
        "Macro-domain intervals resample accident narratives independently within each of the six fixed target domains, then average domains equally. They do not resample domains."]
    (out / "equalized_single_vs_lodo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out / 'equalized_single_vs_lodo.md'}", flush=True)


if __name__ == "__main__":
    main()
