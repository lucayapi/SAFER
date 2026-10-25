#!/usr/bin/env python3
"""Source-only LODO classification experiments on annotated spans.

This campaign compares two annotation-level decisions independently:
  * facthood: keep a GOLD factual span (including OTHER) or reject a GOLD
    explicitly excluded span;
  * role: assign one of the five roles to a GOLD factual span.

It deliberately does not consume segmenter predictions or claim end-to-end
performance. The downstream candidate-span gate is evaluated after the
segmentation study is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEXT = Path(__file__).resolve().parents[1]
ROOT = TEXT.parent
sys.path.insert(0, str(TEXT))

import role_gold_campaign as gold
import rq1_e2_crossfit as bundle_tools
import rq1_e2_excluded as exclusion_tools
import sat_lodo_gold_v3 as sat

CORPORA = gold.CORPORA
CSV_NAME = gold.CSV_NAME
ROLES = tuple(gold.ROLES)
ROLE_ID = {name: i for i, name in enumerate(ROLES)}
# One fixed seed for the requested single-run campaign. Results are descriptive
# for this fit and do not estimate between-seed training variability.
SEEDS = (42,)
METHODS = ("tfidf_lr", "qwen3_frozen_lr", "qwen3_finetuned")
BACKBONE = "Qwen/Qwen3-Embedding-0.6B"
# Snapshot currently used to create the local frozen Qwen3 exports. The smoke
# stage checks this assumption against one saved vector from every corpus.
BACKBONE_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_LENGTH = 256
EPOCHS = 15
PATIENCE = 3
CS = (0.01, 0.1, 1.0, 10.0)
GATE_THRESHOLDS = tuple(float(x) for x in np.arange(0.1, 1.0, 0.1))
QWEN_DIR_CANDIDATES = (
    TEXT / "embeddings" / "gold_v3" / "qwen3",
    TEXT / "embeddings" / "qwen3",  # Existing local layout.
)
STOPWORDS = TEXT / "macro_transfer" / "stop_metier.txt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(decision: str) -> Path:
    if decision not in ("facthood", "roles"):
        raise ValueError(decision)
    base = Path(os.getenv("DECISION_GOLD_RUN_ROOT",
                         str(Path(os.getenv("SCRATCH", str(ROOT))) / "safer_decision_gold_single_v1")))
    return base.expanduser().resolve() / decision


def bundle() -> Path:
    return gold.bundle_path()


def qwen_path(corpus: str) -> Path:
    for directory in QWEN_DIR_CANDIDATES:
        if not directory.exists():
            continue
        missing = [CSV_NAME[c] + ".csv" for c in CORPORA
                   if not (directory / (CSV_NAME[c] + ".csv")).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Qwen3 directory is present but incomplete: {directory}; missing {missing}"
            )
        return directory / f"{CSV_NAME[corpus]}.csv"
    expected = ", ".join(str(path) for path in QWEN_DIR_CANDIDATES)
    raise FileNotFoundError(
        f"Missing the six Qwen3 GOLD exports. Expected a complete set in one of: {expected}"
    )


def backbone_path() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=BACKBONE, revision=BACKBONE_REVISION)


def validate_qwen_exports(frames: dict[str, pd.DataFrame]) -> dict:
    """Bind the manually transferred exports to exact GOLD doc_id order."""
    record = {}
    for corpus in CORPORA:
        path = qwen_path(corpus)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Qwen3 GOLD vectors: {path}. Transfer the complete folder "
                "text/embeddings/gold_v3/qwen3/ to the cluster."
            )
        header = pd.read_csv(path, nrows=0).columns.tolist()
        dims = [c for c in header if c.startswith("dim_")]
        if header != ["doc_id", *dims] or len(dims) != 1024 or len(set(dims)) != 1024:
            raise ValueError(f"Qwen3 export must contain doc_id plus 1024 dimensions: {path}")
        actual = pd.read_csv(path, usecols=["doc_id"], dtype={"doc_id": str})["doc_id"].tolist()
        expected = frames[corpus].doc_id.astype(str).tolist()
        if actual != expected or len(actual) != len(set(actual)):
            raise ValueError(f"Qwen3 IDs/order do not match frozen GOLD: {corpus}")
        record[corpus] = {"path": str(path), "sha256": digest(path),
                          "rows": len(actual), "dimensions": len(dims),
                          "doc_id_order_sha256": hashlib.sha256(
                              "\n".join(actual).encode("utf-8")).hexdigest()}
    return record


def validate_source_partitions(bundle_path: Path, frames: dict, split: dict) -> None:
    """Cross-check facts, offsets, roles and narrative IDs against the RQ1 bundle."""
    for target in CORPORA:
        groups = bundle_tools.bundle_groups(bundle_path, target)
        for part, rows in groups.items():
            allowed = {corpus: set(split[corpus][part]) for corpus in CORPORA
                       if (part != "test" or corpus == target)}
            observed = set()
            for row in rows:
                corpus, document_id = row["corpus"], str(row["document_id"])
                if corpus not in allowed or document_id not in allowed[corpus]:
                    raise ValueError(f"Bundle row outside frozen split: {target}/{part}/{corpus}/{document_id}")
                key = (corpus, document_id)
                if key in observed:
                    raise ValueError(f"Repeated narrative in split: {key}")
                observed.add(key)
                sat.validate_gold(row)
                exclusion_tools.check_annotations(row)
                part_df = frames[corpus]
                part_df = part_df[part_df.accident_id.astype(str).eq(document_id)]
                expected_facts = sorted(
                    (int(r.start), int(r.end), str(r.pred_label), str(r.sentence))
                    for r in part_df.itertuples(index=False)
                )
                observed_facts = sorted(
                    (int(u["start"]), int(u["end"]), str(u["role"]),
                     row["raw_text"][int(u["start"]):int(u["end"])])
                    for u in row["units"]
                )
                if expected_facts != observed_facts:
                    raise ValueError(f"GOLD facts differ between CSV and span bundle: {key}")


def collect_exclusions(bundle_path: Path) -> dict[str, list[dict]]:
    """Collect all explicit exclusions across folds, deduplicated by source span."""
    result = {c: [] for c in CORPORA}
    seen: dict[tuple[str, str, int], tuple[int, int, str]] = {}
    for target in CORPORA:
        groups = bundle_tools.bundle_groups(bundle_path, target)
        for part in ("train", "val", "test"):
            for row in groups[part]:
                corpus, document_id = row["corpus"], str(row["document_id"])
                spans = exclusion_tools.reference_spans(row, "excluded_spans")
                for ix, (a, b) in enumerate(spans):
                    key = (corpus, document_id, ix)
                    value = (a, b, row["raw_text"][a:b])
                    if key in seen and seen[key] != value:
                        raise ValueError(f"Conflicting repeated exclusion annotation: {key}")
                    if key not in seen:
                        seen[key] = value
                        result[corpus].append({"corpus": corpus, "accident_id": document_id,
                                               "excluded_index": ix, "start": a, "end": b,
                                               "text": value[2]})
    return result


def exclusion_file(root: Path, corpus: str) -> Path:
    return root / "qwen3_excluded" / f"{corpus}.npz"


def exclusion_manifest_path(root: Path, corpus: str) -> Path:
    return root / "qwen3_excluded" / f"{corpus}.manifest.json"


def campaign_binding(decision: str, bundle_path: Path, signatures: dict,
                     qwen_manifest: dict, split: dict) -> dict:
    return {"version": "decision_gold_lodo_v1", "decision": decision,
            "gold_sha256": gold.GOLD_V3_SNAPSHOT_SHA256,
            "bundle_manifest_sha256": digest(bundle_path / "manifest.json"),
            "gold_signatures": signatures, "qwen_exports": qwen_manifest,
            "stopwords_sha256": digest(STOPWORDS), "methods": list(METHODS),
            "roles": list(ROLES), "seeds": list(SEEDS),
            "backbone": BACKBONE, "backbone_revision": BACKBONE_REVISION,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS, "patience": PATIENCE, "C_grid": list(CS),
            "split_counts": {c: {p: len(ids) for p, ids in parts.items()}
                             for c, parts in split.items()},
            "script_sha256": digest(Path(__file__))}


def load_campaign(decision: str):
    bundle_path = bundle()
    frames, split, signatures = gold.load_gold(bundle_path)
    if not STOPWORDS.is_file():
        raise FileNotFoundError(
            f"Transfer the métier stop-word file manually to the cluster at {STOPWORDS}"
        )
    qwen_manifest = validate_qwen_exports(frames)
    validate_source_partitions(bundle_path, frames, split)
    root = output_root(decision)
    binding = campaign_binding(decision, bundle_path, signatures, qwen_manifest, split)
    return root, bundle_path, frames, split, signatures, qwen_manifest, binding


def prepare(args) -> None:
    root, bundle_path, frames, split, signatures, qwen_manifest, binding = load_campaign(args.decision)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "binding.json"
    if marker.is_file():
        if read_json(marker) != binding:
            raise ValueError(f"Existing campaign binding differs from data/code: {root}")
    elif any(root.iterdir()):
        raise ValueError(f"Refusing to bind a non-empty output directory: {root}")
    else:
        write_json(marker, binding)
    counts = {}
    for target in CORPORA:
        groups = bundle_tools.bundle_groups(bundle_path, target)
        counts[target] = {part: {"narratives": len(rows),
            "facts": sum(len(r["units"]) for r in rows),
            "excluded": sum(len(r["excluded_spans"]) for r in rows)}
            for part, rows in groups.items()}
    class_names = ("KEEP", "REJECT") if args.decision == "facthood" else ROLES
    distributions = {}
    for target in CORPORA:
        train, val, test = examples_for_target(args.decision, target, bundle_path, frames, split)
        distributions[target] = {}
        for name, frame in (("train", train), ("validation", val), ("test", test)):
            distributions[target][name] = {
                "examples": int(len(frame)),
                "narratives": int(frame[["corpus", "accident_id"]].drop_duplicates().shape[0]),
                "classes": {label: int((frame.label_id == i).sum())
                            for i, label in enumerate(class_names)}}
    write_json(root / "decision_class_distributions.json", distributions)
    write_json(root / "data_counts.json", counts)
    if args.decision == "facthood":
        exclusions = collect_exclusions(bundle_path)
        missing = [c for c in CORPORA if not exclusions[c]]
        if missing:
            raise ValueError(f"No explicit exclusion spans for corpora: {missing}")
        write_json(root / "excluded_text_counts.json",
                   {c: len(rows) for c, rows in exclusions.items()})
    free_gib = shutil.disk_usage(root).free / 2**30
    print(json.dumps({"ready": True, "decision": args.decision,
                      "output": str(root), "free_gib": round(free_gib, 2),
                      "methods": list(METHODS), "seeds": list(SEEDS),
                      "qwen_exports": qwen_manifest}, indent=2), flush=True)


def verify_binding(decision: str, root: Path) -> tuple:
    data = load_campaign(decision)
    expected = data[-1]
    path = root / "binding.json"
    if not path.is_file() or read_json(path) != expected:
        raise ValueError(f"Missing or changed source/data binding: {root}")
    return data


def encode_exclusions(args) -> None:
    import torch
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
    from contrastive_methods.hf_training_common import encode_texts

    root = output_root("facthood")
    data = verify_binding("facthood", root)
    bundle_path, qwen_manifest = data[1], data[5]
    corpus = CORPORA[args.corpus_index]
    output, manifest_path = exclusion_file(root, corpus), exclusion_manifest_path(root, corpus)
    expected = {"corpus": corpus, "qwen_backbone": BACKBONE,
                "max_length": MAX_LENGTH, "input_prefix": "", "normalized": True,
                "bundle_manifest_sha256": digest(bundle_path / "manifest.json"),
                "qwen_csv_sha256": qwen_manifest[corpus]["sha256"],
                "script_sha256": digest(Path(__file__))}
    if output.is_file() and manifest_path.is_file():
        saved = read_json(manifest_path)
        if saved.get("expected") != expected or digest(output) != saved.get("npz_sha256"):
            raise ValueError(f"Existing exclusion embeddings have different provenance: {output}")
        with np.load(output, allow_pickle=False) as arr:
            if len(arr["keys"]) != saved.get("rows") or arr["embeddings"].shape != (saved.get("rows"), 1024):
                raise ValueError(f"Corrupt exclusion embeddings: {output}")
        print(f"[skip] verified exclusion vectors {corpus}", flush=True)
        return
    exclusions = collect_exclusions(bundle_path)[corpus]
    if not exclusions:
        raise ValueError(f"No explicit exclusions for {corpus}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 frozen embedding export requires a GPU")
    cfg = ContrastiveConfig(method_name="frozen_qwen3_export", dataset_path=Path("unused"),
        backbone_name=BACKBONE, input_prefix="", max_seq_length=MAX_LENGTH,
        backbone_trainable=False, train_last_n_layers=None, cache_backbone_embeddings=False,
        use_projector=False, projection="none", hiddim=1024, use_amp=True,
        encode_batch_size=8)
    model = ContrastiveEncoder(EncoderConfig(backbone_name=backbone_path(),
        max_seq_length=MAX_LENGTH, backbone_trainable=False,
        cache_backbone_embeddings=False, use_projector=False)).to("cuda")
    texts = [r["text"] for r in exclusions]
    vectors = encode_texts(model, texts, cfg, "cuda", batch_size=8,
                           normalize_embeddings=True, show_progress=True)
    if vectors.shape != (len(exclusions), 1024) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid frozen Qwen3 exclusion embeddings: {corpus}")
    keys = np.asarray([f"{r['corpus']}|{r['accident_id']}|{r['excluded_index']}"
                       for r in exclusions], dtype=str)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, keys=keys, embeddings=vectors.astype(np.float32))
    os.replace(temp, output)
    write_json(manifest_path, {"expected": expected, "rows": len(keys),
                               "npz_sha256": digest(output)})
    print(json.dumps({"encoded_exclusions": len(keys), "corpus": corpus,
                      "path": str(output)}, indent=2), flush=True)


def examples_for_target(decision: str, target: str, bundle_path: Path,
                        frames: dict, split: dict):
    """Return fixed train/validation/test examples, all grouped by narrative."""
    train_facts, val_facts, tests = gold.selection("lodo", target, frames, split)
    if decision == "roles":
        def cv(frame):
            return pd.DataFrame({"corpus": frame.corpus.astype(str),
                "accident_id": frame.accident_id.astype(str), "doc_id": frame.doc_id.astype(str),
                "text": frame.sentence.astype(str), "label_id": frame.label_id.astype(int),
                "label": frame.pred_label.astype(str), "source_type": "fact"})
        return cv(train_facts), cv(val_facts), cv(tests[target])

    groups = bundle_tools.bundle_groups(bundle_path, target)
    parts = {}
    for part, rows in groups.items():
        records = []
        for row in rows:
            corpus, document_id = row["corpus"], str(row["document_id"])
            for ix, (a, b) in enumerate(exclusion_tools.reference_spans(row, "excluded_spans")):
                text = row["raw_text"][a:b]
                if not text.strip():
                    raise ValueError(f"Empty exclusion after trimming: {corpus}/{document_id}/{ix}")
                records.append({"corpus": corpus, "accident_id": document_id,
                    "doc_id": f"EXCL:{document_id}:{ix}", "excluded_index": ix,
                    "text": text, "label_id": 1, "label": "REJECT", "source_type": "excluded"})
        parts[part] = pd.DataFrame(records)

    def combine(facts, exclusions):
        fact_rows = pd.DataFrame({"corpus": facts.corpus.astype(str),
            "accident_id": facts.accident_id.astype(str), "doc_id": facts.doc_id.astype(str),
            "text": facts.sentence.astype(str), "label_id": 0,
            "label": "KEEP", "source_type": "fact"})
        result = pd.concat([fact_rows, exclusions], ignore_index=True)
        result["label_id"] = result.label_id.astype(int)
        if result.empty or result.text.str.strip().eq("").any():
            raise ValueError("Empty gate examples or text")
        if result.groupby(["corpus", "accident_id", "source_type", "doc_id"]).size().max() > 1:
            raise ValueError("Duplicate gate example within a narrative")
        if set(result.label_id.unique()) != {0, 1}:
            raise ValueError("A gate split must include both factual and excluded spans")
        return result
    return combine(train_facts, parts["train"]), combine(val_facts, parts["val"]), combine(tests[target], parts["test"])


def class_metrics(y: np.ndarray, pred: np.ndarray, decision: str) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    n_classes = 2 if decision == "facthood" else 5
    labels = list(range(n_classes))
    if set(np.unique(y).tolist()) != set(labels):
        raise ValueError(f"Evaluation split lacks one or more {decision} classes")
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    names = ("KEEP", "REJECT") if decision == "facthood" else ROLES
    result = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "recall_by_label": {names[i]: float(recall[i]) for i in labels},
              "class_counts": {names[i]: int((y == i).sum()) for i in labels},
              "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
              "n_examples": int(len(y))}
    if decision == "facthood":
        result["fact_recall"] = float(recall[0])
        result["exclusion_recall"] = float(recall[1])
    return result


def gate_threshold(y: np.ndarray, p_reject: np.ndarray) -> tuple[float, float]:
    best = None
    for threshold in GATE_THRESHOLDS:
        pred = (p_reject >= threshold).astype(int)
        score = class_metrics(y, pred, "facthood")["balanced_accuracy"]
        if best is None or (score, -abs(threshold - .5)) > (best[1], -abs(best[0] - .5)):
            best = (threshold, score)
    assert best is not None
    return best


def load_exclusion_vectors(root: Path, corpus: str) -> dict[str, np.ndarray]:
    path, sidecar = exclusion_file(root, corpus), exclusion_manifest_path(root, corpus)
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Run the exclusion embedding stage first: {path}")
    saved = read_json(sidecar)
    binding = read_json(root / "binding.json")
    expected = {"corpus": corpus, "qwen_backbone": BACKBONE,
                "max_length": MAX_LENGTH, "input_prefix": "", "normalized": True,
                "bundle_manifest_sha256": binding["bundle_manifest_sha256"],
                "qwen_csv_sha256": binding["qwen_exports"][corpus]["sha256"],
                "script_sha256": digest(Path(__file__))}
    if saved.get("expected") != expected or digest(path) != saved.get("npz_sha256"):
        raise ValueError(f"Changed exclusion embedding file: {path}")
    with np.load(path, allow_pickle=False) as arr:
        keys = arr["keys"].astype(str)
        values = arr["embeddings"].astype(np.float32)
    if values.shape != (len(keys), 1024) or len(set(keys.tolist())) != len(keys):
        raise ValueError(f"Invalid exclusion embedding matrix: {path}")
    return dict(zip(keys.tolist(), values, strict=True))


def load_fact_vectors(corpus: str, expected_ids: list[str]) -> dict[str, np.ndarray]:
    path = qwen_path(corpus)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dims = [c for c in header if c.startswith("dim_")]
    if header != ["doc_id", *dims] or len(dims) != 1024:
        raise ValueError(f"Qwen vector schema does not match expected 1024D export: {corpus}")
    wanted = set(map(str, expected_ids))
    found: dict[str, np.ndarray] = {}
    for chunk in pd.read_csv(path, usecols=["doc_id", *dims], dtype={"doc_id": str}, chunksize=8192):
        selected = chunk[chunk.doc_id.isin(wanted)]
        for row in selected.itertuples(index=False, name=None):
            key = str(row[0])
            if key in found:
                raise ValueError(f"Repeated Qwen vector ID {key} in {corpus}")
            vector = np.asarray(row[1:], dtype=np.float32)
            if vector.shape != (1024,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid Qwen vector {key} in {corpus}")
            found[key] = vector
    if set(found) != wanted:
        missing = sorted(wanted - set(found))[:10]
        raise ValueError(f"Missing Qwen doc_ids for {corpus}: {missing}")
    return found


def vectors_for(frame: pd.DataFrame, fact_maps: dict, excluded_maps: dict) -> np.ndarray:
    rows = []
    for r in frame.itertuples(index=False):
        if r.source_type == "fact":
            rows.append(fact_maps[r.corpus][str(r.doc_id)])
        else:
            key = f"{r.corpus}|{r.accident_id}|{int(r.excluded_index)}"
            rows.append(excluded_maps[r.corpus][key])
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 1024 or not np.isfinite(matrix).all():
        raise ValueError("Invalid Qwen3 feature matrix")
    return matrix


def predictions(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        classes = model.named_steps["logisticregression"].classes_.astype(int)
    else:
        classes = model.classes_.astype(int)
    raw = model.predict_proba(x)
    k = 2 if raw.shape[1] <= 2 and (len(classes) == 2) else 5
    probs = np.zeros((len(x), k), dtype=np.float32)
    probs[:, classes] = raw
    return probs.argmax(axis=1), probs


def select_lr(decision: str, x_train, y_train, x_val, y_val, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = None
    for c in CS:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, random_state=seed))
        model.fit(x_train, y_train)
        raw = model.predict_proba(x_val)
        classes = model.named_steps["logisticregression"].classes_.astype(int)
        probs = np.zeros((len(y_val), 2 if decision == "facthood" else 5), dtype=np.float32)
        probs[:, classes] = raw
        if decision == "facthood":
            threshold, _ = gate_threshold(y_val, probs[:, 1])
            pred = (probs[:, 1] >= threshold).astype(int)
        else:
            threshold, pred = None, probs.argmax(1)
        score = class_metrics(y_val, pred, decision)["balanced_accuracy"]
        candidate = (score, -c, model, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3], best[0]


def tfidf_method(decision: str, train: pd.DataFrame, val: pd.DataFrame,
                 test: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict]:
    from sklearn.linear_model import LogisticRegression
    from macro_transfer.tfidf_baseline import fit_vectorizer, normalize_stopwords
    params = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2,
              "max_df": 1.0, "max_features": 50000, "sublinear_tf": True,
              "norm": "l2", "lowercase": True, "strip_accents": None}
    stopwords = normalize_stopwords(STOPWORDS, params)
    vectorizer, x_train = fit_vectorizer(train.text.tolist(), params, stopwords, "source train")
    x_val = vectorizer.transform(val.text.tolist()).tocsr()
    x_test = vectorizer.transform(test.text.tolist()).tocsr()
    model, threshold, score = select_lr(decision, x_train, train.label_id.to_numpy(),
                                       x_val, val.label_id.to_numpy(), seed)
    _, probs = predictions(model, x_test)
    pred = ((probs[:, 1] >= threshold).astype(int) if decision == "facthood"
            else probs.argmax(1))
    return pred, {"validation_ba": score, "selected_C": float(model.named_steps["logisticregression"].C),
                  "selected_reject_threshold": threshold,
                  "vocabulary_size": len(vectorizer.vocabulary_)}


def qwen_frozen_method(decision: str, train: pd.DataFrame, val: pd.DataFrame,
                       test: pd.DataFrame, seed: int, root: Path) -> tuple[np.ndarray, dict]:
    combined = pd.concat([train, val, test], ignore_index=True)
    fact_maps = {}
    for corpus in CORPORA:
        needed = combined.loc[(combined.corpus == corpus) & (combined.source_type == "fact"), "doc_id"]
        if not needed.empty:
            fact_maps[corpus] = load_fact_vectors(corpus, needed.astype(str).unique().tolist())
    excluded_maps = ({c: load_exclusion_vectors(root, c) for c in CORPORA}
                     if decision == "facthood" else {})
    x_train = vectors_for(train, fact_maps, excluded_maps)
    x_val = vectors_for(val, fact_maps, excluded_maps)
    x_test = vectors_for(test, fact_maps, excluded_maps)
    model, threshold, score = select_lr(decision, x_train, train.label_id.to_numpy(),
                                       x_val, val.label_id.to_numpy(), seed)
    _, probs = predictions(model, x_test)
    pred = ((probs[:, 1] >= threshold).astype(int) if decision == "facthood"
            else probs.argmax(1))
    return pred, {"validation_ba": score, "selected_C": float(model.named_steps["logisticregression"].C),
                  "selected_reject_threshold": threshold}


def verify_qwen_export_revision(frames: dict[str, pd.DataFrame], device: str) -> dict:
    """Check cached frozen vectors against the pinned backbone on one fact per corpus."""
    import torch
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
    from contrastive_methods.hf_training_common import encode_texts

    samples = []
    texts = []
    for corpus in CORPORA:
        row = frames[corpus].iloc[0]
        exported = pd.read_csv(qwen_path(corpus), nrows=1, dtype={"doc_id": str})
        if exported.empty or str(row.doc_id) != str(exported.iloc[0].doc_id):
            raise ValueError(f"Cannot find the first GOLD row in Qwen export: {corpus}")
        dims = [c for c in exported.columns if c.startswith("dim_")]
        samples.append(exported.iloc[0][dims].to_numpy(dtype=np.float32))
        texts.append(str(row.sentence))

    cfg = ContrastiveConfig(method_name="decision_gold_revision_check", dataset_path=Path("unused"),
        backbone_name=BACKBONE, input_prefix="", max_seq_length=MAX_LENGTH,
        backbone_trainable=False, cache_backbone_embeddings=False,
        use_projector=False, projection="none", hiddim=1024, use_amp=True,
        encode_batch_size=8)
    model = ContrastiveEncoder(EncoderConfig(backbone_name=backbone_path(),
        max_seq_length=MAX_LENGTH, backbone_trainable=False,
        cache_backbone_embeddings=False, use_projector=False)).to(device)
    actual = encode_texts(model, texts, cfg, device, batch_size=8,
                         normalize_embeddings=True, show_progress=False)
    del model
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    similarities = np.sum(actual * np.asarray(samples), axis=1)
    if actual.shape != (len(CORPORA), 1024) or not np.isfinite(similarities).all():
        raise ValueError("Invalid Qwen3 revision-check output")
    result = {corpus: float(score) for corpus, score in zip(CORPORA, similarities, strict=True)}
    if min(result.values()) < 0.999:
        raise ValueError(
            "Frozen Qwen3 CSVs do not match the pinned model revision/configuration; "
            f"cosine checks were {result}. Regenerate exports or confirm their provenance."
        )
    return result


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def text_loader(frame: pd.DataFrame, tokenizer, *, shuffle: bool, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class SpanDataset(Dataset):
        def __init__(self, df):
            self.texts = df.text.astype(str).tolist()
            self.labels = df.label_id.astype(int).tolist()
        def __len__(self): return len(self.texts)
        def __getitem__(self, i): return self.texts[i], self.labels[i]

    def collate(rows):
        texts, labels = zip(*rows)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.float32 if False else torch.long)}

    return DataLoader(SpanDataset(frame), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=2, pin_memory=True)


def probabilities_from_model(model, loader, device, decision: str):
    import torch
    model.eval(); parts = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model.forward_logits(batch["input_ids"].to(device),
                                              batch["attention_mask"].to(device)).float()
            probs = torch.sigmoid(logits.squeeze(-1)) if decision == "facthood" else torch.softmax(logits, -1)
            parts.append(probs.cpu().numpy())
    return np.concatenate(parts, axis=0)


def finetune_method(decision: str, seed: int, train: pd.DataFrame, val: pd.DataFrame,
                    test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoTokenizer
    from supervised_macro_ft.model import SupervisedMacroModel

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 fine-tuning requires a CUDA GPU")
    seed_everything(seed)
    device = torch.device("cuda")
    model_ref = backbone_path()
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    n_classes = 1 if decision == "facthood" else 5
    model = SupervisedMacroModel(backbone_name=model_ref, num_classes=n_classes,
        pooling="mean", backbone_trainable=True, train_last_n_layers=None,
        gradient_checkpointing=True, projection=None, dropout=0.1).to(device)
    if decision == "facthood":
        negatives = int((train.label_id == 0).sum())
        positives = int((train.label_id == 1).sum())
        if min(negatives, positives) == 0:
            raise ValueError("Gate training needs both factual and exclusion examples")
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device))
    else:
        counts = np.bincount(train.label_id.to_numpy(), minlength=5).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError("Role training split lacks at least one of the five roles")
        weights = len(train) / (5.0 * counts)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    # Small microbatches keep full Qwen3 fine-tuning feasible on 40-80 GB GPUs.
    train_loader = text_loader(train, tokenizer, shuffle=True, batch_size=4)
    val_loader = text_loader(val, tokenizer, shuffle=False, batch_size=32)
    best_ba, best_state, best_threshold, best_epoch, stale = -1.0, None, None, 0, 0
    history = []
    for epoch in range(EPOCHS):
        model.train(); total = 0.0; count = 0
        optimizer.zero_grad(set_to_none=True)
        accumulation = 4
        for batch_index, batch in enumerate(train_loader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(batch["input_ids"].to(device),
                                              batch["attention_mask"].to(device))
                y = batch["labels"].to(device)
                loss = (loss_fn(logits.squeeze(-1), y.float()) if decision == "facthood"
                        else loss_fn(logits, y))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite Qwen3 {decision} loss at epoch {epoch + 1}")
            group_size = min(accumulation, len(train_loader) -
                             (batch_index // accumulation) * accumulation)
            (loss / group_size).backward()
            total += float(loss.detach()); count += 1
            if (batch_index + 1) % accumulation == 0 or (batch_index + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        val_probs = probabilities_from_model(model, val_loader, device, decision)
        if decision == "facthood":
            threshold, _ = gate_threshold(val.label_id.to_numpy(), val_probs)
            val_pred = (val_probs >= threshold).astype(int)
        else:
            threshold, val_pred = None, val_probs.argmax(1)
        metric = class_metrics(val.label_id.to_numpy(), val_pred, decision)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, count),
                        "validation_ba": metric["balanced_accuracy"],
                        "reject_threshold": threshold})
        print(json.dumps({"decision": decision, "seed": seed, **history[-1]}), flush=True)
        if metric["balanced_accuracy"] > best_ba + 1e-12:
            best_ba, best_threshold, best_epoch, stale = metric["balanced_accuracy"], threshold, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Qwen3 fine-tuning selected no epoch")
    model.load_state_dict(best_state); model.to(device)
    test_loader = text_loader(test, tokenizer, shuffle=False, batch_size=32)
    test_probs = probabilities_from_model(model, test_loader, device, decision)
    pred = ((test_probs >= best_threshold).astype(int) if decision == "facthood"
            else test_probs.argmax(1))
    del model
    torch.cuda.empty_cache()
    return pred, {"validation_ba": float(best_ba), "selected_epoch": int(best_epoch),
                  "selected_reject_threshold": best_threshold, "history": history}


def save_result(folder: Path, decision: str, target: str, method: str, seed: int,
                test: pd.DataFrame, pred: np.ndarray, selection: dict, binding_hash: str) -> None:
    if len(pred) != len(test):
        raise ValueError("Prediction count differs from test rows")
    y = test.label_id.to_numpy(dtype=int)
    metrics = class_metrics(y, pred, decision)
    metrics["n_narratives"] = int(test[["corpus", "accident_id"]].drop_duplicates().shape[0])
    metrics.update({"decision": decision, "target": target, "method": method,
                    "seed": seed, "selection": selection,
                    "target_used_for_selection": False,
                    "annotation_level_only": True,
                    "test_rows_sha256": hashlib.sha256(
                        "\n".join(test.corpus.astype(str) + "|" + test.accident_id.astype(str) +
                                 "|" + test.doc_id.astype(str)).encode("utf-8")).hexdigest(),
                    "binding_sha256": binding_hash})
    pred_file = folder / "predictions.csv"
    folder.mkdir(parents=True, exist_ok=True)
    out = test[["corpus", "accident_id", "doc_id", "source_type", "label"]].copy()
    out["prediction"] = [str(int(p)) for p in pred]
    tmp = pred_file.with_name(pred_file.name + f".{os.getpid()}.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, pred_file)
    metrics["predictions_sha256"] = digest(pred_file)
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "complete.json", {"binding_sha256": binding_hash,
        "predictions_sha256": digest(pred_file), "metrics_sha256": digest(folder / "metrics.json"),
        "test_rows": len(test)})


def task_spec(index: int) -> tuple[str, int]:
    if not 0 <= index < len(CORPORA) * len(SEEDS):
        raise ValueError(f"Task index outside 0..{len(CORPORA) * len(SEEDS)-1}")
    return CORPORA[index % len(CORPORA)], SEEDS[index // len(CORPORA)]


def run(args) -> None:
    root = output_root(args.decision)
    data = verify_binding(args.decision, root)
    bundle_path, frames, split, binding = data[1], data[2], data[3], data[-1]
    target, seed = task_spec(args.task_index)
    train, val, test = examples_for_target(args.decision, target, bundle_path, frames, split)
    if args.decision == "facthood":
        for part, frame in (("train", train), ("validation", val), ("test", test)):
            if set(frame.label_id.unique()) != {0, 1}:
                raise ValueError(f"{target}/{part} lacks factual or exclusion examples")
    binding_hash = digest(root / "binding.json")
    folder = root / f"target_{target}" / f"seed_{seed}"
    if seed == SEEDS[0]:
        for method in ("tfidf_lr", "qwen3_frozen_lr"):
            dest = folder / method
            if (dest / "complete.json").is_file():
                marker = read_json(dest / "complete.json")
                if marker.get("binding_sha256") != binding_hash or digest(dest / "predictions.csv") != marker.get("predictions_sha256"):
                    raise ValueError(f"Completed baseline changed: {dest}")
                print(f"[skip] {dest}", flush=True)
                continue
            predictor = tfidf_method if method == "tfidf_lr" else qwen_frozen_method
            if method == "tfidf_lr":
                pred, selected = predictor(args.decision, train, val, test, seed)
            else:
                pred, selected = predictor(args.decision, train, val, test, seed, root)
            save_result(dest, args.decision, target, method, seed, test, pred, selected, binding_hash)
    finetune_dest = folder / "qwen3_finetuned"
    if (finetune_dest / "complete.json").is_file():
        marker = read_json(finetune_dest / "complete.json")
        if marker.get("binding_sha256") != binding_hash or digest(finetune_dest / "predictions.csv") != marker.get("predictions_sha256"):
            raise ValueError(f"Completed fine-tuning changed: {finetune_dest}")
        print(f"[skip] {finetune_dest}", flush=True)
        return
    pred, selected = finetune_method(args.decision, seed, train, val, test)
    save_result(finetune_dest, args.decision, target, "qwen3_finetuned", seed,
                test, pred, selected, binding_hash)


def smoke(args) -> None:
    import torch
    root = output_root(args.decision)
    data = verify_binding(args.decision, root)
    bundle_path, frames, split = data[1], data[2], data[3]
    target = CORPORA[0]
    train, val, test = examples_for_target(args.decision, target, bundle_path, frames, split)
    train = train.groupby("label_id", group_keys=False).head(30).reset_index(drop=True)
    val = val.groupby("label_id", group_keys=False).head(10).reset_index(drop=True)
    test = test.groupby("label_id", group_keys=False).head(10).reset_index(drop=True)
    if any(set(frame.label_id.unique()) != set(range(2 if args.decision == "facthood" else 5))
           for frame in (train, val, test)):
        raise ValueError("Smoke subset does not contain every decision class")
    qwen_similarity = verify_qwen_export_revision(frames, "cuda")
    tfidf_pred, tfidf_selection = tfidf_method(args.decision, train, val, test, SEEDS[0])
    qwen_pred, qwen_selection = qwen_frozen_method(args.decision, train, val, test, SEEDS[0], root)
    ft_pred, ft_selection = finetune_method(args.decision, SEEDS[0], train, val, test)
    passed = all(len(pred) == len(test) for pred in (tfidf_pred, qwen_pred, ft_pred))
    record = {"decision": args.decision, "n_train": len(train), "n_val": len(val),
              "n_test": len(test), "gpu": torch.cuda.get_device_name(0),
              "passed": passed,
              "pinned_qwen_revision": BACKBONE_REVISION,
              "qwen_export_cosine_checks": qwen_similarity,
              "methods": {"tfidf_lr": tfidf_selection,
                          "qwen3_frozen_lr": qwen_selection,
                          "qwen3_finetuned": ft_selection}}
    if not record["passed"]:
        raise RuntimeError("Qwen3 GPU smoke failed")
    write_json(root / "smoke_complete.json", record)
    print(json.dumps(record, indent=2), flush=True)


def summary(args) -> None:
    root = output_root(args.decision)
    data = verify_binding(args.decision, root)
    binding_hash = digest(root / "binding.json")
    rows = []
    expected_methods = METHODS if args.decision == "roles" else METHODS
    for target in CORPORA:
        train, val, test = examples_for_target(args.decision, target, data[1], data[2], data[3])
        for seed in SEEDS:
            methods = ("tfidf_lr", "qwen3_frozen_lr", "qwen3_finetuned") if seed == SEEDS[0] else ("qwen3_finetuned",)
            for method in methods:
                folder = root / f"target_{target}" / f"seed_{seed}" / method
                complete = read_json(folder / "complete.json")
                metrics = read_json(folder / "metrics.json")
                pred_path = folder / "predictions.csv"
                if (complete.get("binding_sha256") != binding_hash or
                    complete.get("predictions_sha256") != digest(pred_path) or
                    complete.get("metrics_sha256") != digest(folder / "metrics.json") or
                    metrics.get("target_used_for_selection") is not False):
                    raise ValueError(f"Missing or changed model outputs: {folder}")
                pred_frame = pd.read_csv(pred_path, dtype=str)
                expected_ids = (test[["corpus", "accident_id", "doc_id", "source_type", "label"]]
                                .astype(str).reset_index(drop=True))
                observed_ids = pred_frame[["corpus", "accident_id", "doc_id", "source_type", "label"]]
                if not observed_ids.reset_index(drop=True).equals(expected_ids):
                    raise ValueError(f"Predictions are not aligned to fixed test examples: {folder}")
                rows.append({"target": target, "method": method, "seed": seed,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"], "n_examples": metrics["n_examples"],
                    "n_narratives": metrics["n_narratives"],
                    "fact_recall": metrics.get("fact_recall"),
                    "exclusion_recall": metrics.get("exclusion_recall")})
    out = root / "results"
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "scores_by_target_seed.csv", index=False)
    grouped = frame.groupby(["target", "method"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "first"),
        macro_f1=("macro_f1", "first"), n_examples=("n_examples", "first"))
    if args.decision == "facthood":
        gate_diag = frame.groupby(["target", "method"], as_index=False).agg(
            fact_recall=("fact_recall", "first"),
            exclusion_recall=("exclusion_recall", "first"))
        grouped = grouped.merge(gate_diag, on=["target", "method"], validate="one_to_one")
    grouped.to_csv(out / "scores_by_target.csv", index=False)
    macro = grouped.groupby("method", as_index=False).agg(
        macro_domain_balanced_accuracy=("balanced_accuracy", "mean"),
        sd_across_domains=("balanced_accuracy", "std"),
        macro_domain_macro_f1=("macro_f1", "mean"),
        n_domains=("target", "nunique"))
    if args.decision == "facthood":
        macro_diag = grouped.groupby("method", as_index=False).agg(
            macro_domain_fact_recall=("fact_recall", "mean"),
            macro_domain_exclusion_recall=("exclusion_recall", "mean"))
        macro = macro.merge(macro_diag, on="method", validate="one_to_one")
    macro.to_csv(out / "macro_domain_summary.csv", index=False)
    write_json(out / "manifest.json", {"decision": args.decision,
        "binding_sha256": binding_hash, "targets": list(CORPORA),
        "methods": list(METHODS), "seeds": list(SEEDS),
        "annotation_level_only": True, "rows": len(frame),
        "scores_sha256": digest(out / "scores_by_target_seed.csv"),
        "macro_sha256": digest(out / "macro_domain_summary.csv")})
    print(json.dumps({"decision": args.decision, "summary": str(out),
                      "macro": macro.to_dict(orient="records")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("prepare", prepare), ("embed-exclusions", encode_exclusions),
                     ("smoke", smoke), ("run", run), ("summary", summary)):
        p = sub.add_parser(name)
        p.add_argument("--decision", choices=("facthood", "roles"), required=True)
        if name == "embed-exclusions":
            p.add_argument("--corpus-index", type=int, choices=range(6), required=True)
        if name == "run":
            p.add_argument("--task-index", type=int,
                           choices=range(len(CORPORA) * len(SEEDS)), required=True)
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
