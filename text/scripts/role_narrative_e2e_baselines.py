#!/usr/bin/env python3
"""Genuine role-aware span evaluation for the two recoverable role baselines.

The saved CE/SupCon/SoftTriple/Triplet predictions are on reference units and
their selected full-encoder checkpoints were pruned. They cannot be substituted
for predictions on SaT segments. This script refits only the deterministic
TF-IDF and frozen-Qwen3 logistic heads at their saved source-selected C values.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import compare_narrative_only_saved_runs as spans
import role_qwen3_lodo_campaign as roles
from narrative_only_boundary_metric import trim_edges


ROOT = Path(__file__).resolve().parents[2]
SEGMENTERS = ("M0", "M1", "M2", "M3")
CLASSIFIERS = ("tfidf_lr", "qwen3_frozen_lr")
VERSION = "role_narrative_e2e_baselines_v1"
FIELDS = ("target", "segmenter", "classifier", "n_documents", "n_reference",
          "n_predicted", "span_matches", "correct_role_matches", "wrong_role_matches",
          "unmatched_reference", "unmatched_predictions", "unit_f1",
          "role_aware_precision", "role_aware_recall", "role_aware_f1",
          "matched_role_accuracy")
PREDICTION_FIELDS = ("target", "segmenter", "classifier", "document_id", "start",
                     "end", "normalized_start", "normalized_end", "predicted_role",
                     "matched_gold_index", "matched_gold_role", "match_iou")


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    storage = Path(os.environ.get("SCRATCH", ROOT))
    prepared = args.prepared or Path(os.environ.get("SAT_NARRATIVE_PREPARED",
        ROOT / "annotation/lodo_sat_transfer/narrative_only_lodo_v1"))
    baseline = args.sat_baselines or Path(os.environ.get("SAT_NARRATIVE_BASELINE_ROOT",
        storage / "safer_sat_lodo/runs/narrative_only_lodo_v3_baselines"))
    adapted = args.sat_m3 or Path(os.environ.get("SAT_NARRATIVE_RUN_ROOT",
        storage / "safer_sat_lodo/runs/narrative_only_lodo_checkpoint_v1"))
    output = args.output or Path(os.environ.get("ROLE_E2E_OUTPUT_ROOT",
        roles.root() / "e2e_baselines_v1"))
    os.environ["ROLE_QWEN3_BUNDLE"] = str(prepared.resolve())
    return tuple(p.resolve() for p in (prepared, baseline, adapted, output))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, records: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def binding(prepared: Path, baseline: Path, adapted: Path) -> dict:
    role_campaign = roles.campaign()
    provenance = spans.verify_provenance(prepared, baseline, adapted)
    campaign = read(role_campaign)
    if campaign["signatures"]["bundle_manifest_sha256"] != provenance["prepared_manifest_sha256"]:
        raise ValueError("Role and SaT campaigns use different LODO bundles")
    for target in roles.CORPORA:
        for method in CLASSIFIERS:
            dest = roles.root() / "baselines" / target / method
            if not roles.verified_done(dest, role_campaign):
                raise ValueError(f"Missing or changed reference-unit baseline: {dest}")
    return {"version": VERSION, "script_sha256": spans.digest(Path(__file__)),
            "role_campaign_sha256": spans.digest(role_campaign),
            "sat_provenance": provenance, "segmenters": list(SEGMENTERS),
            "classifiers": list(CLASSIFIERS),
            "evaluation": "target test spans classified from their own raw text; "
                          "one-to-one role-blind matching at normalized character IoU >= 0.5"}


def availability() -> dict:
    """Report why full-encoder methods cannot be evaluated from saved labels."""
    result = {}
    for method in ("ce", "supcon", "softtriple", "batch_triplet"):
        found = 0
        for target in roles.CORPORA:
            for seed in roles.SEEDS:
                folder = (roles.root() / "missing_ce_batch_triplet_v1" / "trained" / target /
                          method / f"seed_{seed}" if method in ("ce", "batch_triplet") else
                          roles.root() / "trained" / target / method / f"seed_{seed}")
                found += (folder / "best_model").is_dir()
        result[method] = {"selected_checkpoints_present": found, "expected": 18,
                          "reference_predictions_are_not_end_to_end": True}
    return result


def preflight(args: argparse.Namespace) -> None:
    prepared, baseline, adapted, output = paths(args)
    expected = binding(prepared, baseline, adapted)
    manifest = output / "manifest.json"
    if manifest.is_file() and read(manifest) != expected:
        if any(output.glob("target_*/complete.json")):
            raise ValueError(f"Existing completed output has a different provenance: {output}")
    write(manifest, expected)
    write(output / "adapted_checkpoint_availability.json", availability())
    print(json.dumps({"ready": True, "output": str(output),
                      "scope": "M0-M3 x TF-IDF/Frozen-Qwen3 only",
                      "adapted_checkpoints": availability()}, indent=2))


def require_manifest(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict]:
    prepared, baseline, adapted, output = paths(args)
    manifest = output / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Run the end-to-end preflight first")
    expected = binding(prepared, baseline, adapted)
    if read(manifest) != expected:
        raise ValueError("End-to-end provenance changed since preflight")
    return prepared, baseline, adapted, output, expected


def target_predictions(reference: list[dict], file: Path, method: str) -> list[list[tuple[int, int]]]:
    predictions = list(spans.load_jsonl(file))
    if len(reference) != len(predictions):
        raise ValueError(f"Incomplete {method} target predictions: {file}")
    return [spans.valid_spans(row, prediction, method)
            for row, prediction in zip(reference, predictions, strict=True)]


def prepare_segments(reference: list[dict], predictions: list[list[tuple[int, int]]]):
    texts, entries, by_document = [], [], []
    for row, predicted in zip(reference, predictions, strict=True):
        raw = row["raw_text"]
        indices = []
        for start, end in predicted:
            a, b = trim_edges(raw, start, end)
            if a == b:
                continue
            indices.append(len(entries))
            entries.append((row["document_id"], start, end, a, b))
            texts.append(raw[start:end])
        by_document.append(indices)
    return texts, entries, by_document


def align(row: dict, predicted: list[tuple[int, int]], labels: list[str]) -> tuple[dict, list[dict]]:
    """Use the same role-blind maximum-cardinality alignment as the M0-M3 audit."""
    raw = row["raw_text"]
    gold = [(trim_edges(raw, int(u["start"]), int(u["end"])), u["role"])
            for u in row["units"]]
    gold = [(interval, role) for interval, role in gold if interval[0] < interval[1]]
    clean = [(trim_edges(raw, a, b), a, b, label)
             for (a, b), label in zip(predicted, labels, strict=True)]
    clean = [item for item in clean if item[0][0] < item[0][1]]
    if len({item[0] for item in clean}) != len(clean):
        raise ValueError(f"Duplicate normalized prediction in {row['document_id']}")
    graph = []
    for interval, _, _, _ in clean:
        neighbors = []
        for index, (unit, _) in enumerate(gold):
            overlap = spans.interval_overlap(interval, unit)
            if overlap and overlap / ((interval[1] - interval[0]) +
                                    (unit[1] - unit[0]) - overlap) >= 0.5:
                neighbors.append(index)
        graph.append(neighbors)
    matched = [-1] * len(gold)

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in graph[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            prior = matched[gold_index]
            if prior < 0 or augment(prior, seen):
                matched[gold_index] = pred_index
                return True
        return False

    for index in range(len(clean)):
        augment(index, set())
    pred_to_gold = {pred_index: gold_index for gold_index, pred_index
                    in enumerate(matched) if pred_index >= 0}
    correct = sum(clean[pred_index][3] == gold[gold_index][1]
                  for pred_index, gold_index in pred_to_gold.items())
    counts = {"n_reference": len(gold), "n_predicted": len(clean),
              "span_matches": len(pred_to_gold), "correct_role_matches": correct,
              "wrong_role_matches": len(pred_to_gold) - correct,
              "unmatched_reference": len(gold) - len(pred_to_gold),
              "unmatched_predictions": len(clean) - len(pred_to_gold)}
    old = spans.partial_counts(row, predicted)
    if (old["iou_tp"], old["iou_pred"], old["iou_ref"]) != \
            (counts["span_matches"], counts["n_predicted"], counts["n_reference"]):
        raise ValueError(f"IoU matching differs from the M0-M3 audit: {row['document_id']}")
    records = []
    for index, (interval, start, end, label) in enumerate(clean):
        gold_index = pred_to_gold.get(index)
        gold_interval, gold_role = gold[gold_index] if gold_index is not None else (None, None)
        overlap = spans.interval_overlap(interval, gold_interval) if gold_interval else 0
        iou = (overlap / ((interval[1] - interval[0]) +
                          (gold_interval[1] - gold_interval[0]) - overlap)
               if gold_interval else None)
        records.append({"document_id": row["document_id"], "start": start, "end": end,
                        "normalized_start": interval[0], "normalized_end": interval[1],
                        "predicted_role": label, "matched_gold_index": gold_index,
                        "matched_gold_role": gold_role, "match_iou": iou})
    return counts, records


def score_target(target: str, method: str, classifier: str, reference: list[dict],
                 predictions: list[list[tuple[int, int]]], labels: list[str]) -> tuple[dict, list[dict]]:
    totals = collections.Counter()
    output = []
    offset = 0
    for row, predicted in zip(reference, predictions, strict=True):
        valid = sum(a < b for start, end in predicted
                    for a, b in [trim_edges(row["raw_text"], start, end)])
        # Empty normalized spans have no classifier input and cannot match a fact.
        full_labels = []
        for a, b in predicted:
            trimmed = trim_edges(row["raw_text"], a, b)
            if trimmed[0] < trimmed[1]:
                full_labels.append(labels[offset])
                offset += 1
            else:
                full_labels.append("")
        assert sum(bool(x) for x in full_labels) == valid
        counts, records = align(row, predicted, full_labels)
        totals.update(counts)
        output.extend({"target": target, "segmenter": method,
                       "classifier": classifier, **record} for record in records)
    if offset != len(labels):
        raise ValueError("Not all classified segments were consumed")
    unit_f1 = 2 * totals["span_matches"] / (totals["n_predicted"] + totals["n_reference"])
    role_f1 = 2 * totals["correct_role_matches"] / (totals["n_predicted"] + totals["n_reference"])
    return {"target": target, "segmenter": method, "classifier": classifier,
            "n_documents": len(reference), **{name: totals[name] for name in
            ("n_reference", "n_predicted", "span_matches", "correct_role_matches",
             "wrong_role_matches", "unmatched_reference", "unmatched_predictions")},
            "unit_f1": unit_f1,
            "role_aware_precision": totals["correct_role_matches"] / totals["n_predicted"]
                                    if totals["n_predicted"] else 0.0,
            "role_aware_recall": totals["correct_role_matches"] / totals["n_reference"]
                                 if totals["n_reference"] else 0.0,
            "role_aware_f1": role_f1,
            "matched_role_accuracy": totals["correct_role_matches"] / totals["span_matches"]
                                     if totals["span_matches"] else 0.0}, output


def baseline_predict(model, features) -> np.ndarray:
    """Match the original float32 probability export and tie behaviour."""
    learner = model[-1] if hasattr(model, "steps") else model
    probs = np.zeros((features.shape[0], len(roles.ROLES)), dtype=np.float32)
    probs[:, learner.classes_.astype(int)] = model.predict_proba(features)
    return probs.argmax(axis=1)


def fit_baseline(target: str, method: str, frames: dict, splits: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from macro_transfer.tfidf_baseline import fit_vectorizer, normalize_stopwords

    train, _, test = roles.selection(target, frames, splits)
    folder = roles.root() / "baselines" / target / method
    saved = read(folder / "metrics.json")
    c = saved["selection"]["C"]
    if c not in roles.CS:
        raise ValueError(f"Unexpected saved C for {target}/{method}")
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=1000,
                                    random_state=roles.SEEDS[0])
    if method == "tfidf_lr":
        if saved["selection"]["stopwords_sha256"] != roles.digest(roles.old_qwen.STOPWORDS):
            raise ValueError("TF-IDF stopword file changed")
        stops = normalize_stopwords(roles.old_qwen.STOPWORDS, roles.TFIDF_PARAMS)
        vectorizer, features = fit_vectorizer(train.sentence.tolist(),
                                              roles.TFIDF_PARAMS, stops, "source train")
        if len(vectorizer.vocabulary_) != saved["selection"]["vocabulary_size"]:
            raise ValueError("TF-IDF vocabulary changed")
        classifier.fit(features, train.label_id.to_numpy())
        verify = baseline_predict(classifier, vectorizer.transform(test.sentence.tolist()))
        transform = vectorizer.transform
    else:
        exported = {}
        for corpus in roles.CORPORA:
            data = pd.read_csv(roles.old_qwen.qwen_path(corpus), dtype={"doc_id": str})
            exported[corpus] = data.set_index("doc_id")
        features = roles.vectors(train, exported)
        classifier = make_pipeline(StandardScaler(), classifier)
        classifier.fit(features, train.label_id.to_numpy())
        verify = baseline_predict(classifier, roles.vectors(test, exported))
        del exported, features
        gc.collect()
        transform = None
    saved_predictions = pd.read_csv(folder / "predictions.csv", dtype={"doc_id": str})
    if (saved_predictions.doc_id.tolist() != test.doc_id.tolist() or
            saved_predictions.pred_label.tolist() != test.pred_label.tolist()):
        raise ValueError(f"Refitted {method} has different target facts: {target}")
    old_labels = saved_predictions.prediction.tolist()
    disagreements = sum(a != b for a, b in
                        zip(old_labels, (roles.ROLES[i] for i in verify), strict=True))
    agreement = 1 - disagreements / len(test)
    # A changed sklearn/SciPy build may change a small number of borderline
    # LR decisions. Record the difference instead of claiming identical weights.
    if agreement < 0.99:
        raise ValueError(f"Large baseline refit drift: {target}/{method}: {agreement:.4%}")
    gold_metrics = roles.old_role.scores(test.label_id.to_numpy(), verify)
    audit = {"target": target, "classifier": method, "saved_C": c,
             "n_reference_units": len(test), "saved_prediction_disagreements": disagreements,
             "saved_prediction_agreement": agreement,
             "refit_gold_balanced_accuracy": gold_metrics["balanced_accuracy"],
             "saved_gold_balanced_accuracy": saved["balanced_accuracy"],
             "refit_gold_macro_f1": gold_metrics["macro_f1"],
             "saved_gold_macro_f1": saved["macro_f1"]}
    return classifier, transform, audit


def qwen_encoder(frames: dict):
    import torch
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
    from contrastive_methods.hf_training_common import encode_texts

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen Qwen3 span encoding requires CUDA")
    model = ContrastiveEncoder(EncoderConfig(
        backbone_name=roles.old_qwen.backbone_path(),
        max_seq_length=roles.MAX_LENGTH, backbone_trainable=False,
        cache_backbone_embeddings=False, use_projector=False)).to("cuda")
    config = ContrastiveConfig(method_name="frozen_qwen3_e2e", dataset_path=Path("unused"),
        backbone_name=roles.BACKBONE, input_prefix="", max_seq_length=roles.MAX_LENGTH,
        backbone_trainable=False, cache_backbone_embeddings=False,
        use_projector=False, projection="none", hiddim=1024, use_amp=True,
        encode_batch_size=8)
    samples = [str(frames[c].iloc[0].sentence) for c in roles.CORPORA]
    expected = []
    for c in roles.CORPORA:
        row = pd.read_csv(roles.old_qwen.qwen_path(c), nrows=1)
        expected.append(row.filter(regex="^dim_").iloc[0].to_numpy(dtype=np.float32))
    actual = encode_texts(model, samples, config, "cuda", batch_size=8,
                          normalize_embeddings=True, show_progress=False)
    similarities = np.sum(actual * np.asarray(expected), axis=1)
    if not np.isfinite(similarities).all() or min(similarities) < 0.999:
        raise ValueError(f"Pinned Qwen3 does not reproduce the frozen exports: {similarities}")
    return model, config, encode_texts


def smoke(args: argparse.Namespace) -> None:
    """Exercise the pinned frozen encoder on actual M3 spans before the array."""
    import torch

    prepared, _, adapted, output, _ = require_manifest(args)
    frames, _, _ = roles.load_data(check_qwen=True)
    target = roles.CORPORA[0]
    reference = list(spans.load_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl"))
    predictions = target_predictions(
        reference, adapted / f"target_{target}" / "predictions.jsonl", "M3")
    texts, _, _ = prepare_segments(reference[:16], predictions[:16])
    if not texts:
        raise ValueError("No actual M3 segments available for GPU smoke")
    encoder, config, encode_texts = qwen_encoder(frames)
    vectors = encode_texts(encoder, texts[:8], config, "cuda", batch_size=8,
                           normalize_embeddings=True, show_progress=False)
    if vectors.shape != (min(8, len(texts)), 1024) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid frozen-Qwen3 smoke embeddings: {vectors.shape}")
    write(output / "gpu_smoke.json", {
        "binding_sha256": spans.digest(output / "manifest.json"),
        "target": target, "n_segments": len(vectors), "embedding_dim": vectors.shape[1]})
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[smoke] frozen Qwen3 encoded {len(vectors)} actual M3 segments", flush=True)


def run_target(args: argparse.Namespace) -> None:
    import torch

    prepared, baseline, adapted, output, _ = require_manifest(args)
    marker = output / "gpu_smoke.json"
    if not marker.is_file() or read(marker).get("binding_sha256") != spans.digest(output / "manifest.json"):
        raise RuntimeError("Run the end-to-end GPU smoke first")
    target = roles.CORPORA[args.task_index]
    dest = output / f"target_{target}"
    complete = dest / "complete.json"
    if complete.is_file():
        marker = read(complete)
        if (marker.get("binding_sha256") == spans.digest(output / "manifest.json") and
                all(spans.digest(dest / name) == sha for name, sha in
                    marker["files_sha256"].items())):
            print(f"[skip] verified {dest}")
            return
        raise ValueError(f"Existing incomplete/changed end-to-end result: {dest}")
    frames, splits, _ = roles.load_data(check_qwen=True)
    reference = list(spans.load_jsonl(prepared / "folds" / f"target_{target}" / "test.jsonl"))
    if len(reference) != len(splits[target]["test"]):
        raise ValueError(f"Target document count changed: {target}")
    segmentations = {}
    for method in SEGMENTERS:
        file = (adapted / f"target_{target}" / "predictions.jsonl" if method == "M3" else
                baseline / "predictions" / target / f"{method}.jsonl")
        segmentations[method] = target_predictions(reference, file, method)
    all_scores, all_predictions, refit_audits = [], [], []
    for classifier_name in CLASSIFIERS:
        classifier, vectorizer_transform, audit = fit_baseline(target, classifier_name,
                                                                frames, splits)
        refit_audits.append(audit)
        encoder = config = encode_texts = None
        if classifier_name == "qwen3_frozen_lr":
            encoder, config, encode_texts = qwen_encoder(frames)
        for method, predicted in segmentations.items():
            texts, _, _ = prepare_segments(reference, predicted)
            if not texts:
                raise ValueError(f"No {target}/{method} segments to classify")
            if vectorizer_transform is not None:
                features = vectorizer_transform(texts)
            else:
                features = encode_texts(encoder, texts, config, "cuda", batch_size=8,
                                        normalize_embeddings=True, show_progress=False)
            labels = [roles.ROLES[index] for index in baseline_predict(classifier, features)]
            counts, records = score_target(target, method, classifier_name,
                                           reference, predicted, labels)
            all_scores.append(counts)
            all_predictions.extend(records)
            print(f"[done] {target} {method} {classifier_name}: role F1={counts['role_aware_f1']:.4f}",
                  flush=True)
            del features, texts, labels
            gc.collect()
        del classifier, encoder
        gc.collect()
        torch.cuda.empty_cache()
    if len(all_scores) != len(SEGMENTERS) * len(CLASSIFIERS):
        raise ValueError("Incomplete target matrix")
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "scores.csv", all_scores, FIELDS)
    write_csv(dest / "predictions.csv", all_predictions, PREDICTION_FIELDS)
    write(dest / "refit_audit.json", refit_audits)
    write(complete, {"binding_sha256": spans.digest(output / "manifest.json"),
         "files_sha256": {name: spans.digest(dest / name) for name in
                          ("scores.csv", "predictions.csv", "refit_audit.json")}})


def summarize(args: argparse.Namespace) -> None:
    _, _, _, output, _ = require_manifest(args)
    rows, audits = [], []
    for target in roles.CORPORA:
        dest = output / f"target_{target}"
        marker = read(dest / "complete.json")
        if marker["binding_sha256"] != spans.digest(output / "manifest.json"):
            raise ValueError(f"Target binding mismatch: {target}")
        for name, sha in marker["files_sha256"].items():
            if spans.digest(dest / name) != sha:
                raise ValueError(f"Changed end-to-end result: {dest / name}")
        table = pd.read_csv(dest / "scores.csv")
        if len(table) != 8 or set(table.target) != {target}:
            raise ValueError(f"Incomplete {target} scores")
        rows.extend(table.to_dict("records"))
        audits.extend(read(dest / "refit_audit.json"))
    table = pd.DataFrame(rows)
    grouped = table.groupby(["segmenter", "classifier"], as_index=False).agg(
        n_targets=("target", "nunique"), unit_f1=("unit_f1", "mean"),
        role_aware_f1=("role_aware_f1", "mean"),
        role_aware_precision=("role_aware_precision", "mean"),
        role_aware_recall=("role_aware_recall", "mean"),
        matched_role_accuracy=("matched_role_accuracy", "mean"))
    if len(grouped) != 8 or not grouped.n_targets.eq(6).all():
        raise ValueError("Expected 8 configurations on all six targets")
    out = output / "results"
    write_csv(out / "scores_by_target.csv", rows, FIELDS)
    write_csv(out / "macro_six_targets.csv", grouped.to_dict("records"), tuple(grouped.columns))
    write_csv(out / "refit_audit_by_target.csv", audits, tuple(audits[0]))
    write(out / "summary.json", {"scope": "genuine end-to-end for recoverable baselines only",
          "binding_sha256": spans.digest(output / "manifest.json"),
          "scores_by_target_sha256": spans.digest(out / "scores_by_target.csv"),
          "macro_sha256": spans.digest(out / "macro_six_targets.csv"),
          "refit_audit_sha256": spans.digest(out / "refit_audit_by_target.csv")})
    print(grouped.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "smoke", "target", "summary"))
    parser.add_argument("--task-index", type=int, choices=range(6))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--sat-baselines", type=Path)
    parser.add_argument("--sat-m3", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "target":
        if args.task_index is None:
            parser.error("target requires --task-index 0..5")
        run_target(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
