"""Compute source-neighbour diagnostics from one refit and its own predictions.

Only the training-source embeddings are used as neighbours. Target true labels
enter the diagnostic role margin, never model selection or fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replication.runner import load_replication_config, task_matrix, output_root
from build_transfer_error_diagnostics import ROLES, canonical_metadata, read_predictions, recorded_ba, model_location


def aligned_embeddings(run: Path, corpus: str, expected: pd.DataFrame) -> np.ndarray:
    base = run / "embeddings" / f"projected_{corpus}"
    array_path = base.with_suffix(".npy")
    meta_path = base.with_name(base.name + "_metadata.csv")
    if not array_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Adapted embedding/metadata missing for {corpus}: {base}")
    array = np.load(array_path, mmap_mode="r")
    meta = pd.read_csv(meta_path, dtype={"accident_id": str, "doc_id": str, "fact_id": str})
    unit_key = "fact_id" if "fact_id" in meta else "doc_id"
    if unit_key not in meta or "accident_id" not in meta or "pred_label" not in meta:
        raise ValueError(f"Missing unit IDs/labels in {meta_path}")
    if array.ndim != 2 or len(array) != len(meta):
        raise ValueError(f"Embedding row count differs from metadata: {base}")
    keys = pd.MultiIndex.from_arrays([meta.accident_id.astype(str), meta[unit_key].astype(str)])
    if keys.has_duplicates or len(keys) != len(expected) or not keys.sort_values().equals(expected.index):
        raise ValueError(f"Embedding keys differ from prepared {corpus} corpus: {base}")
    indexed = meta.set_index(["accident_id", unit_key], verify_integrity=True)
    indexed.index.names = expected.index.names
    indexed = indexed.reindex(expected.index)
    if not np.array_equal(indexed.pred_label.astype(str), expected.pred_label.astype(str)):
        raise ValueError(f"Embedding labels differ from prepared {corpus} corpus: {base}")
    if "sentence" in indexed and not np.array_equal(indexed.sentence.astype(str), expected.sentence.astype(str)):
        raise ValueError(f"Embedding texts differ from prepared {corpus} corpus: {base}")
    row = pd.Series(np.arange(len(meta)), index=keys).reindex(expected.index).to_numpy(dtype=np.int64)
    values = np.asarray(array[row], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite adapted embeddings: {base}")
    return values


def geometry(source: np.ndarray, source_roles: np.ndarray, target: np.ndarray,
             target_roles: np.ndarray, *, k: int = 10, block: int = 256) -> pd.DataFrame:
    """Exact cosine neighbours and role-centroid margin; lower distance is closer."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source/target embedding dimensions differ")
    if k < 1 or len(source) < k:
        raise ValueError("Invalid source neighbourhood size")
    snorm = np.linalg.norm(source, axis=1)
    tnorm = np.linalg.norm(target, axis=1)
    if (snorm <= 0).any() or (tnorm <= 0).any():
        raise ValueError("Zero-norm adapted embedding")
    s = np.asarray(source / snorm[:, None], dtype=np.float32)
    t = np.asarray(target / tnorm[:, None], dtype=np.float32)
    role_codes = pd.Categorical(source_roles, categories=ROLES).codes
    true_codes = pd.Categorical(target_roles, categories=ROLES).codes
    if (role_codes < 0).any() or (true_codes < 0).any():
        raise ValueError("Unexpected role label")
    centroids = np.stack([s[role_codes == i].mean(axis=0) for i in range(4)])
    cnorm = np.linalg.norm(centroids, axis=1)
    if (cnorm <= 0).any():
        raise ValueError("Empty or zero-norm source role centroid")
    centroids = centroids / cnorm[:, None]
    distance = np.empty(len(t), dtype=np.float32)
    entropy = np.empty(len(t), dtype=np.float32)
    margin = np.empty(len(t), dtype=np.float32)
    nearest_true_role = np.empty(len(t), dtype=np.float32)
    nearest_wrong_role = np.empty(len(t), dtype=np.float32)
    for start in range(0, len(t), block):
        stop = min(start + block, len(t))
        cosine = t[start:stop] @ s.T
        top = np.argpartition(cosine, -k, axis=1)[:, -k:]
        top_cosine = np.take_along_axis(cosine, top, axis=1)
        distance[start:stop] = 1 - top_cosine.mean(axis=1)
        neighbours = role_codes[top]
        proportions = np.stack([(neighbours == role).mean(axis=1) for role in range(4)], axis=1)
        entropy[start:stop] = -(np.where(proportions > 0, proportions * np.log(np.maximum(proportions, 1e-12)), 0)).sum(axis=1)
        role_distance = 1 - t[start:stop] @ centroids.T
        rows = np.arange(stop - start)
        true_distance = role_distance[rows, true_codes[start:stop]]
        role_distance[rows, true_codes[start:stop]] = np.inf
        wrong_distance = role_distance.min(axis=1)
        nearest_true_role[start:stop] = true_distance
        nearest_wrong_role[start:stop] = wrong_distance
        margin[start:stop] = wrong_distance - true_distance
    return pd.DataFrame({"source_distance_k10": distance, "source_label_entropy_k10": entropy,
                         "role_margin": margin, "true_centroid_distance": nearest_true_role,
                         "wrong_centroid_distance": nearest_wrong_role})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="output/replication_recipes/adapted_geometry.yaml")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--block", type=int, default=256)
    args = parser.parse_args()
    config = load_replication_config(args.config)
    tasks = task_matrix(config)
    if not 0 <= args.task_index < len(tasks):
        raise ValueError("Invalid task index")
    task = tasks[args.task_index]
    model_id, seed = task["model_id"], int(task["training_seed"])
    backbone, source, method = next((b, s, m) for b in ("qwen3", "multilingual_e5_large", "camembert")
                                    for s in ("btp", "metallurgie")
                                    for m in ("cross_entropy", "supcon", "softtriple")
                                    if f"{b}_{s}_{m}" == model_id)
    run = output_root(config) / model_id / f"seed_{seed}"
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "complete":
        raise FileNotFoundError(f"Completed refit missing: {manifest_path}")
    out = output_root(config).parent / "features" / model_id / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    metadata = canonical_metadata()
    source_meta = metadata[source]
    source_z = aligned_embeddings(run, source, source_meta)
    comparison = []
    for target in (c for c in ("btp", "metallurgie", "caou", "nicollin") if c != source):
        expected = metadata[target]
        predictions = read_predictions(run / "predictions" / f"predictions_{target}.csv", expected)
        new_ba = float(balanced_accuracy_score(predictions.true_macro, predictions.pred_macro))
        if abs(new_ba - recorded_ba(run, target)) > 1e-8:
            raise ValueError(f"Refit BA does not match its saved predictions: {run}, {target}")
        target_z = aligned_embeddings(run, target, expected)
        geo = geometry(source_z, source_meta.pred_label.to_numpy(dtype=str), target_z,
                       expected.pred_label.to_numpy(dtype=str), block=args.block)
        keys = expected.index.to_frame(index=False)
        result = pd.concat([keys.reset_index(drop=True), predictions.reset_index(drop=True)[["true_macro", "pred_macro"]], geo], axis=1)
        result["error"] = (result.true_macro != result.pred_macro).astype(np.int8)
        result.insert(0, "target", target)
        result.insert(0, "method", method)
        result.insert(0, "source", source)
        result.insert(0, "backbone", backbone)
        result.insert(0, "seed", seed)
        result.to_csv(out / f"features_{target}.csv.gz", index=False, compression="gzip")
        historical_recipe, historical_root, _ = model_location(
            "camembert_base" if backbone == "camembert" else backbone, source, method)
        historical = historical_root / model_id / f"seed_{seed}"
        old_ba = recorded_ba(historical, target)
        comparison.append({"backbone": backbone, "source": source, "method": method,
                           "seed": seed, "target": target, "refit_ba": new_ba,
                           "historical_ba": old_ba, "delta_ba": new_ba - old_ba,
                           "abs_delta_over_1pp": abs(new_ba - old_ba) > .01,
                           "historical_recipe": str(historical_recipe)})
        print(f"{model_id} seed={seed} target={target}: BA refit={new_ba:.4f}, historical={old_ba:.4f}", flush=True)
    pd.DataFrame(comparison).to_csv(out / "run_comparison.csv", index=False)
    (out / "complete.json").write_text(json.dumps({"model_id": model_id, "seed": seed,
        "targets": [r["target"] for r in comparison], "k": 10, "space": "adapted final representation"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
