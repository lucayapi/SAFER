#!/usr/bin/env python3
"""Exploratory SaT-guided factual span refinement on the frozen RQ1 splits.

M3 is frozen. Its source-train features are in-sample and MUST NOT be reported
as an independent estimate of generalisation. Target tests are never used to
choose a checkpoint, candidate threshold, or decoding penalty.
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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import rq1_final as campaign
import rq1_final_report as report
import sat_lodo_gold_v3 as sat

THRESHOLDS = (0.02, 0.05, 0.10)
SEEDS = (13, 23, 37)
PENALTIES = (-2.0, -1.0, 0.0, 1.0, 2.0)
PROJECTION_DIM = 128
EPOCHS = 8
PATIENCE = 2
MAX_NEG_PER_POS = 6
VERSION = "rq1_refiner_exploratory_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict) -> tuple[str, str]:
    return row["corpus"], row["document_id"]


def read_split(bundle: Path, target: str, name: str) -> list[dict]:
    return sat.read_jsonl(bundle / "folds" / f"target_{target}" / f"{name}.jsonl")


def validate_inputs(bundle: Path, m3_root: Path, target: str) -> tuple[dict, dict, dict]:
    manifest = campaign.check_bundle(bundle)
    if target not in manifest["fold_order"]:
        raise ValueError(f"Unknown target: {target}")
    run = m3_root / f"target_{target}"
    selection = load_json(run / "selection.json")
    metrics = load_json(run / "metrics.json")
    if (selection["target"] != target or metrics["target"] != target or
            selection["bundle"]["bundle_manifest_sha256"] != digest(bundle / "manifest.json") or
            metrics["selection_sha256"] != digest(run / "selection.json") or
            metrics["selection"] != selection or metrics["bundle"] != selection["bundle"]):
        raise ValueError(f"M3 selection or GOLD binding mismatch: {target}")
    for name, expected in selection["M3_weights"].items():
        if digest(run / "selected_adapter" / name) != expected:
            raise ValueError(f"M3 adapter changed: {target}/{name}")
    groups = {name: read_split(bundle, target, name) for name in ("train", "val", "test")}
    identities = [{key(row) for row in rows} for rows in groups.values()]
    if (sum(len(ids) for ids in identities) != sum(len(rows) for rows in groups.values()) or
            any(identities[i] & identities[j] for i, j in ((0, 1), (0, 2), (1, 2))) or
            any(row["corpus"] == target for name in ("train", "val") for row in groups[name]) or
            any(row["corpus"] != target for row in groups["test"])):
        raise ValueError(f"Partition leakage: {target}")
    if len(groups["test"]) != manifest["report"][target]["test"]:
        raise ValueError(f"Target test count changed: {target}")
    for rows in groups.values():
        for row in rows:
            sat.validate_gold(row)
    for name, cache in (("val", run / "validation_cache" /
                         f"{selection['M3_candidate']}.sqlite3"),
                        ("test", run / "test_probabilities.sqlite3")):
        marker = cache.with_suffix(".binding.json")
        expected_binding = ({"bundle": selection["bundle"],
                             "candidate": selection["M3_candidate"],
                             "weights": selection["M3_weights"]} if name == "val" else
                            {"bundle": selection["bundle"],
                             "selected_candidate": selection["M3_candidate"],
                             "weights": selection["M3_weights"]})
        if load_json(marker) != expected_binding:
            raise ValueError(f"M3 {name} cache has different weights: {target}")
        with closing(sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            found = set(db.execute("SELECT corpus,document_id FROM probabilities"))
        if found != {key(row) for row in groups[name]}:
            raise ValueError(f"Incomplete M3 {name} probability cache: {target}")
    if digest(run / "predictions" / "M3.jsonl") != metrics["prediction_sha256"]["M3"]:
        raise ValueError(f"Selected M3 predictions changed: {target}")
    return selection, metrics, groups


def binding(bundle: Path, m3_root: Path, target: str, selection: dict) -> dict:
    return {"version": VERSION, "code_sha256": digest(Path(__file__)),
            "gold_manifest_sha256": digest(bundle / "manifest.json"),
            "m3_selection_sha256": digest(m3_root / f"target_{target}" / "selection.json"),
            "m3_weights": selection["M3_weights"], "target": target,
            "projection_dim": PROJECTION_DIM, "thresholds": THRESHOLDS}


def bind_folder(folder: Path, expected: dict) -> None:
    file = folder / "binding.json"
    if file.is_file():
        if load_json(file) != expected:
            raise ValueError(f"Existing output belongs to different inputs/code: {folder}")
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Unbound output directory: {folder}")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        json_write(file, expected)


def _token_windows(length: int, block_size: int, stride: int):
    if length <= 0:
        return
    for j in range(0, length, stride):
        start, end = j, j + block_size
        done = end >= length
        if done:
            end = length
            start = max(end - block_size, 0)
        yield start, end
        if done:
            break


class HiddenCapture:
    """Keep the official extract() windowing/logit aggregation unchanged."""

    def __init__(self, wrapper):
        self.model = wrapper.model
        self.config = wrapper.config
        self.hidden = []

    def __call__(self, *, attention_mask, input_ids=None, hashed_ids=None, language_ids=None):
        import torch
        device = next(self.model.parameters()).device
        kwargs = {"attention_mask": torch.as_tensor(attention_mask, device=device)}
        if input_ids is not None:
            kwargs["input_ids"] = torch.as_tensor(input_ids, device=device)
        if hashed_ids is not None:
            kwargs["hashed_ids"] = torch.as_tensor(hashed_ids, device=device)
        if language_ids is not None:
            kwargs["language_ids"] = torch.as_tensor(language_ids, device=device)
        with torch.inference_mode():
            output = self.model(**kwargs, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("M3 did not expose contextual hidden states")
        self.hidden.extend(output.hidden_states[-1].detach().float().cpu().numpy())
        return {"logits": output.logits.detach().float().cpu().numpy()}


@lru_cache(maxsize=2)
def projection_matrix(hidden_dim: int) -> np.ndarray:
    rng = np.random.default_rng(20261018)
    matrix = rng.standard_normal((hidden_dim, PROJECTION_DIM), dtype=np.float32)
    matrix, _ = np.linalg.qr(matrix)
    return matrix.astype(np.float32)


def extract_document(model, raw: str, *, expected: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return character probabilities, projected token states and token offsets.

    The same tokenizer, 512-token windows, 64-token stride, float16 logit
    aggregation and character interpolation as SaT.predict_proba are used.
    """
    from wtpsplit.extract import extract
    from wtpsplit.utils import Constants, sigmoid, token_to_char_probs

    if not raw.strip():
        raise ValueError("Empty narrative cannot provide hidden states")
    capture = HiddenCapture(model.model)
    logits, mappings, _, tokens = extract([raw], capture, stride=64,
        max_block_size=512, batch_size=8, weighting="uniform", tokenizer=model.tokenizer)
    offsets = np.asarray(mappings[0], dtype=np.int32)
    n = len(offsets)
    if len(capture.hidden) == 0 or n != len(logits[0]):
        raise ValueError("Missing token alignment")
    # extract() clips XLM subword windows to 510 to make room for special tokens.
    block = min(510, n)
    downsampling = getattr(model.model.config, "downsampling_rate", 1)
    block = math.ceil(block / downsampling) * downsampling
    windows = list(_token_windows(n, block, 64))
    if len(windows) != len(capture.hidden):
        raise ValueError("Hidden-state windows differ from official extract()")
    states = np.zeros((n, capture.hidden[0].shape[-1]), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    for (a, b), hidden in zip(windows, capture.hidden, strict=True):
        states[a:b] += hidden[1:1 + b - a]
        counts[a:b] += 1
    if not np.all(counts):
        raise ValueError("Uncovered tokenizer positions")
    states /= counts[:, None]
    projected = states @ projection_matrix(states.shape[-1])
    char_logits = token_to_char_probs(raw, tokens["input_ids"][0], logits[0],
                                      model.special_tokens, mappings[0])
    probabilities = np.asarray(sigmoid(char_logits[:, Constants.NEWLINE_INDEX]), dtype=np.float32)
    if (len(probabilities) != len(raw) or not np.isfinite(probabilities).all() or
            len(projected) != n or np.any(offsets[:, 1] > len(raw))):
        raise ValueError("Character probabilities or token offsets are invalid")
    if expected is not None and (len(expected) != len(probabilities) or
                                 np.max(np.abs(expected - probabilities)) > 2e-3):
        raise ValueError(f"Extracted probabilities disagree with pinned M3 cache: "
                         f"max_abs={np.max(np.abs(expected - probabilities)):.6f}")
    return probabilities, projected.astype(np.float16), offsets


def _read_feature(db: sqlite3.Connection, row: dict):
    found = db.execute("SELECT text_sha256,prob,states,offsets,n_tokens FROM features "
                       "WHERE corpus=? AND document_id=?", key(row)).fetchone()
    if found is None:
        return None
    text_hash, prob, states, offsets, n = found
    if text_hash != hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest():
        raise ValueError(f"Feature cache text changed: {key(row)}")
    probabilities = np.frombuffer(prob, dtype=np.float32)
    hidden = np.frombuffer(states, dtype=np.float16).reshape(n, PROJECTION_DIM)
    mapping = np.frombuffer(offsets, dtype=np.int32).reshape(n, 2)
    if len(probabilities) != len(row["raw_text"]) or n != len(mapping):
        raise ValueError(f"Feature cache corrupt: {key(row)}")
    return probabilities, hidden, mapping


def _open_feature_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS features (corpus TEXT, document_id TEXT, "
               "text_sha256 TEXT, prob BLOB, states BLOB, offsets BLOB, n_tokens INTEGER, "
               "PRIMARY KEY (corpus,document_id))")
    return db


def feature_cache(bundle: Path, m3_root: Path, target: str, groups: dict,
                  out: Path, *, limit: int | None = None) -> None:
    from tqdm import tqdm
    selection = load_json(m3_root / f"target_{target}" / "selection.json")
    model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
    model = sat.load_pinned_sat(model_dir, tokenizer_dir,
                                m3_root / f"target_{target}" / "selected_adapter")
    model.model.model.eval()
    cache = out / "features.sqlite3"
    external = {}
    for name, path in (("val", m3_root / f"target_{target}" / "validation_cache" /
                        f"{selection['M3_candidate']}.sqlite3"),
                       ("test", m3_root / f"target_{target}" / "test_probabilities.sqlite3")):
        external[name] = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        with closing(_open_feature_db(cache)) as db:
            for name in ("train", "val", "test"):
                rows = groups[name][:limit] if limit else groups[name]
                for row in tqdm(rows, desc=f"M3 features {target}/{name}", unit="récit"):
                    if _read_feature(db, row) is not None:
                        continue
                    expected = sat.get_probs(external[name], row) if name != "train" else None
                    prob, states, offsets = extract_document(model, row["raw_text"], expected=expected)
                    db.execute("INSERT INTO features VALUES (?,?,?,?,?,?,?)",
                               (*key(row), hashlib.sha256(row["raw_text"].encode("utf-8")).hexdigest(),
                                prob.tobytes(), states.tobytes(), offsets.tobytes(), len(offsets)))
                    db.commit()
            if not limit:
                expected_ids = {key(row) for rows in groups.values() for row in rows}
                actual = set(db.execute("SELECT corpus,document_id FROM features"))
                if actual != expected_ids:
                    raise ValueError(f"Incomplete or foreign feature cache: {target}")
    finally:
        for connection in external.values():
            connection.close()
        del model


def nonspace_geometry(raw: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rank k denotes a cut after k non-space characters; start/end offsets
    # trim adjacent whitespace without changing the original narrative.
    prefix = np.zeros(len(raw) + 1, dtype=np.int32)
    prefix[1:] = np.cumsum(np.fromiter((not c.isspace() for c in raw),
                                            dtype=np.int8, count=len(raw)))
    positions = np.flatnonzero(np.fromiter((not c.isspace() for c in raw),
                                            dtype=np.bool_, count=len(raw)))
    starts = np.empty(len(positions) + 1, dtype=np.int32)
    ends = np.empty(len(positions) + 1, dtype=np.int32)
    starts[:-1], starts[-1] = positions, len(raw)
    ends[0], ends[1:] = 0, positions + 1
    return prefix, starts, ends


def candidate_ranks(raw: str, probs: np.ndarray, prefix: np.ndarray,
                    threshold: float) -> list[int]:
    total = int(prefix[-1])
    rank = {0, total}
    rank.update(int(prefix[i + 1]) for i in np.flatnonzero(probs > threshold))
    rank.update(int(prefix[i + 1]) for i, char in enumerate(raw) if char == "\n")
    return sorted(value for value in rank if 0 <= value <= total)


def rank_probabilities(probs: np.ndarray, prefix: np.ndarray) -> np.ndarray:
    result = np.zeros(int(prefix[-1]) + 1, dtype=np.float32)
    np.maximum.at(result, prefix[1:], probs)
    result[0] = result[-1] = 1.0
    return result


def gold_rank_spans(row: dict, prefix: np.ndarray) -> list[tuple[int, int]]:
    raw = row["raw_text"]
    return [(int(prefix[a]), int(prefix[b])) for unit in row["units"]
            for a, b in [sat.trim_span(raw, unit["start"], unit["end"])]]


def token_vector(states: np.ndarray, offsets: np.ndarray, char: int) -> np.ndarray:
    # Tokenizer offsets are in Python-character coordinates, like GOLD offsets.
    indexes = np.flatnonzero((offsets[:, 0] <= char) & (char < offsets[:, 1]))
    if len(indexes) == 0:
        distances = np.minimum(np.abs(offsets[:, 0] - char), np.abs(offsets[:, 1] - char))
        return states[int(np.argmin(distances))].astype(np.float32)
    return states[int(indexes[0])].astype(np.float32)


def _span_features(raw: str, probs: np.ndarray, states: np.ndarray,
                   offsets: np.ndarray, spans: list[tuple[int, int]],
                   prefix: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if not spans:
        return np.empty((0, 5 * PROJECTION_DIM + 3), dtype=np.float16)
    rp = rank_probabilities(probs, prefix)
    result = np.empty((len(spans), 5 * PROJECTION_DIM + 3), dtype=np.float16)
    zero = np.zeros(PROJECTION_DIM, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        start, end = int(starts[a]), int(ends[b])
        if not 0 <= start < end <= len(raw):
            raise ValueError("Invalid candidate offsets")
        indexes = np.flatnonzero((offsets[:, 1] > start) & (offsets[:, 0] < end))
        if len(indexes) == 0:
            raise ValueError("Candidate has no aligned tokens")
        left = states[indexes[0] - 1].astype(np.float32) if indexes[0] else zero
        right = states[indexes[-1] + 1].astype(np.float32) if indexes[-1] + 1 < len(states) else zero
        first = states[indexes[0]].astype(np.float32)
        last = states[indexes[-1]].astype(np.float32)
        pool = np.mean(states[indexes].astype(np.float32), axis=0)
        pa = float(np.clip(rp[a], 1e-5, 1 - 1e-5))
        pb = float(np.clip(rp[b], 1e-5, 1 - 1e-5))
        result[i] = np.concatenate((first, last, pool, left, right,
                                    [math.log(pa / (1 - pa)), math.log(pb / (1 - pb)),
                                     math.log1p(end - start)]))
    return result


def boundary_features(raw: str, probs: np.ndarray, states: np.ndarray,
                      offsets: np.ndarray, ranks: list[int],
                      starts: np.ndarray, ends: np.ndarray,
                      prefix: np.ndarray) -> np.ndarray:
    rp = rank_probabilities(probs, prefix)
    result = np.empty((len(ranks), 2 * PROJECTION_DIM + 2), dtype=np.float16)
    for i, rank in enumerate(ranks):
        left = token_vector(states, offsets, max(0, int(ends[rank]) - 1))
        right = token_vector(states, offsets, min(len(raw) - 1, int(starts[rank])))
        p = float(np.clip(rp[rank], 1e-5, 1 - 1e-5))
        result[i] = np.concatenate((left, right, [math.log(p / (1 - p)),
                                                 rank / max(1, int(prefix[-1]))]))
    return result


@dataclass
class Example:
    row: dict
    threshold: float
    ranks: list[int]
    spans: list[tuple[int, int]]
    features: np.ndarray
    labels: np.ndarray
    merges: list[tuple[int, int, int]]
    fragments: list[tuple[int, int, int]]
    kind: str


def make_example(row: dict, feature: tuple[np.ndarray, np.ndarray, np.ndarray],
                 threshold: float, max_length: int, kind: str,
                 *, training: bool = False) -> Example:
    raw = row["raw_text"]
    probs, states, offsets = feature
    prefix, starts, ends = nonspace_geometry(raw)
    ranks = candidate_ranks(raw, probs, prefix, threshold)
    gold = gold_rank_spans(row, prefix)
    gold_set = set(gold)
    if kind == "E2":
        internals = ranks[1:-1]
        gold_boundaries = {v for pair in gold for v in pair if 0 < v < prefix[-1]}
        features = boundary_features(raw, probs, states, offsets, internals,
                                     starts, ends, prefix)
        labels = np.array([r in gold_boundaries for r in internals], dtype=np.float32)
        return Example(row, threshold, internals, [], features, labels, [], [], kind)
    spans = [(a, b) for i, a in enumerate(ranks) for b in ranks[i + 1:]
             if b > a and int(ends[b]) - int(starts[a]) <= max_length]
    # A candidate wholly in an unannotated gap is not a known negative unless
    # the gap is explicitly excluded in the reference.
    excluded = [(int(prefix[x["start"]]), int(prefix[x["end"]]))
                for x in row.get("excluded_spans", [])]
    labels = np.array([1 if pair in gold_set else
                       0 if any(a < pair[1] and pair[0] < b for a, b in gold) or
                       any(a <= pair[0] and pair[1] <= b for a, b in excluded) else -1
                       for pair in spans], dtype=np.int8)
    merges, fragments = [], []
    index = {span: i for i, span in enumerate(spans)}
    for g1, g2 in zip(gold, gold[1:]):
        union = (g1[0], g2[1])
        if g1 in index and g2 in index and union in index:
            merges.append((index[g1], index[g2], index[union]))
    for g in gold:
        if g not in index:
            continue
        for rank in ranks:
            if g[0] < rank < g[1] and (g[0], rank) in index and (rank, g[1]) in index:
                fragments.append((index[g], index[(g[0], rank)], index[(rank, g[1])]))
                break
    if training:
        rng = np.random.default_rng(int(hashlib.sha256(
            f"{row['document_id']}:sampling".encode()).hexdigest()[:16], 16))
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels == 0)
        n_negative = min(len(negatives), max(8, MAX_NEG_PER_POS * len(positives)))
        chosen = set(positives.tolist())
        if n_negative:
            chosen.update(rng.choice(negatives, n_negative, replace=False).tolist())
        chosen.update(i for triple in merges + fragments for i in triple)
        keep = sorted(chosen)
        new_index = {old: i for i, old in enumerate(keep)}
        merges = [tuple(new_index[i] for i in triple) for triple in merges]
        fragments = [tuple(new_index[i] for i in triple) for triple in fragments]
        spans = [spans[i] for i in keep]
        labels = labels[keep]
    features = _span_features(raw, probs, states, offsets, spans, prefix, starts, ends)
    return Example(row, threshold, ranks, spans, features, labels, merges, fragments, kind)


def spans_from_ranks(raw: str, chosen: list[tuple[int, int]]) -> list[tuple[int, int]]:
    _, starts, ends = nonspace_geometry(raw)
    spans = [(int(starts[a]), int(ends[b])) for a, b in chosen]
    if spans != sorted(set(spans)) or any(x[1] > y[0] for x, y in zip(spans, spans[1:])):
        raise ValueError("Decoder returned overlapping spans")
    return spans


def decode_spans(spans: list[tuple[int, int]], scores: np.ndarray,
                 penalty: float) -> list[tuple[int, int]]:
    """Weighted interval scheduling with an explicit skip option."""
    if len(spans) != len(scores):
        raise ValueError("Span-score count mismatch")
    ranks = sorted({v for pair in spans for v in pair})
    if not ranks:
        return []
    incoming = collections.defaultdict(list)
    for pair, score in zip(spans, scores, strict=True):
        incoming[pair[1]].append((pair[0], float(score)))
    best = {ranks[0]: 0.0}
    back = {}
    for prev, end in zip(ranks, ranks[1:]):
        value, choice = best[prev], None
        for start, score in incoming[end]:
            candidate = best[start] + score - penalty
            if candidate > value:
                value, choice = candidate, (start, end)
        best[end] = value
        back[end] = (prev if choice is None else choice[0], choice)
    result = []
    current = ranks[-1]
    while current != ranks[0]:
        current, choice = back[current]
        if choice is not None:
            result.append(choice)
    return result[::-1]


def new_network(kind: str):
    import torch
    size = 2 * PROJECTION_DIM + 2 if kind == "E2" else 5 * PROJECTION_DIM + 3
    return torch.nn.Sequential(torch.nn.Linear(size, 256), torch.nn.ReLU(),
                               torch.nn.Dropout(0.2), torch.nn.Linear(256, 64),
                               torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(64, 1))


def predict_scores(model, example: Example, device: str) -> np.ndarray:
    import torch
    if not len(example.features):
        return np.empty(0, dtype=np.float32)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(example.features), 512):
            x = torch.from_numpy(example.features[start:start + 512].astype(np.float32)).to(device)
            outputs.append(model(x).squeeze(-1).cpu().numpy())
    return np.concatenate(outputs)


def predict_example(model, example: Example, device: str,
                    penalty: float) -> list[tuple[int, int]]:
    scores = predict_scores(model, example, device)
    raw = example.row["raw_text"]
    if example.kind == "E2":
        chosen = [rank for rank, score in zip(example.ranks, scores, strict=True)
                  if score > penalty]
        total = int(nonspace_geometry(raw)[0][-1])
        bounds = [0, *chosen, total]
        return spans_from_ranks(raw, list(zip(bounds, bounds[1:])))
    return spans_from_ranks(raw, decode_spans(example.spans, scores, penalty))


def macro_f1(examples: list[Example], predictions: list[list[tuple[int, int]]]) -> float:
    by_corpus = collections.defaultdict(lambda: collections.Counter())
    for example, spans in zip(examples, predictions, strict=True):
        counts = sat.score(example.row, spans)
        by_corpus[example.row["corpus"]].update(counts)
    if len(by_corpus) != 5:
        raise ValueError("Validation must contain exactly five source domains")
    return float(np.mean([sat.prf(c["span_tp"], c["span_pred"], c["span_ref"])[2]
                          for c in by_corpus.values()]))


def evaluate_grid(model, examples_by_threshold: dict[float, list[Example]],
                  kind: str, device: str) -> tuple[dict, list[dict]]:
    details = []
    for threshold, examples in examples_by_threshold.items():
        scores = [predict_scores(model, example, device) for example in examples]
        for penalty in PENALTIES:
            predictions = []
            for example, value in zip(examples, scores, strict=True):
                if kind == "E2":
                    chosen = [rank for rank, score in zip(example.ranks, value, strict=True)
                              if score > penalty]
                    total = int(nonspace_geometry(example.row["raw_text"])[0][-1])
                    bounds = [0, *chosen, total]
                    spans = spans_from_ranks(example.row["raw_text"],
                                             list(zip(bounds, bounds[1:])))
                else:
                    spans = spans_from_ranks(example.row["raw_text"],
                                             decode_spans(example.spans, value, penalty))
                predictions.append(spans)
            details.append({"threshold": threshold, "penalty": penalty,
                            "macro_exact_f1": macro_f1(examples, predictions)})
    chosen = max(details, key=lambda r: (r["macro_exact_f1"],
                                         -abs(r["threshold"] - 0.05),
                                         -abs(r["penalty"]), -r["threshold"], -r["penalty"]))
    return chosen, details


def train_network(kind: str, seed: int, train: list[Example],
                  validation: dict[float, list[Example]], out: Path,
                  device: str) -> dict:
    import torch
    from tqdm import tqdm
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = new_network(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best = None
    stale = 0
    history = []
    by_domain = collections.defaultdict(list)
    for index, item in enumerate(train):
        by_domain[item.row["corpus"]].append(index)
    if len(by_domain) != 5:
        raise ValueError("Training must contain exactly five source domains")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        per_domain = math.ceil(len(train) / len(by_domain))
        order = np.concatenate([rng.choice(indexes, per_domain,
                    replace=per_domain > len(indexes)) for indexes in by_domain.values()])
        rng.shuffle(order)
        for first in tqdm(range(0, len(order), 16),
                          desc=f"{kind} seed={seed} epoch={epoch}/{EPOCHS}",
                          unit="batch", leave=False):
            batch = [train[int(index)] for index in order[first:first + 16]
                     if len(train[int(index)].features)]
            if not batch:
                continue
            x = torch.from_numpy(np.concatenate([item.features for item in batch]).astype(np.float32)).to(device)
            y = torch.from_numpy(np.concatenate([item.labels for item in batch]).astype(np.float32)).to(device)
            known = y >= 0
            if not known.any():
                continue
            raw_scores = model(x).squeeze(-1)
            n_pos = int((y[known] == 1).sum().item())
            n_neg = int((y[known] == 0).sum().item())
            pos_weight = min(10.0, n_neg / max(1, n_pos))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                raw_scores[known], y[known],
                pos_weight=torch.tensor(pos_weight, device=device))
            if kind == "E4":
                margin = 1.0
                merges, fragments = [], []
                offset = 0
                for item in batch:
                    merges.extend(tuple(offset + i for i in triple) for triple in item.merges)
                    fragments.extend(tuple(offset + i for i in triple) for triple in item.fragments)
                    offset += len(item.features)
                if merges:
                    triples = torch.tensor(merges, device=device)
                    loss = loss + 0.5 * torch.relu(margin + raw_scores[triples[:, 2]] -
                        raw_scores[triples[:, 0]] - raw_scores[triples[:, 1]]).mean()
                if fragments:
                    triples = torch.tensor(fragments, device=device)
                    loss = loss + 0.5 * torch.relu(margin + raw_scores[triples[:, 1]] +
                        raw_scores[triples[:, 2]] - raw_scores[triples[:, 0]]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        choice, grid = evaluate_grid(model, validation, kind, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                  **choice}
        history.append(record)
        print(json.dumps({"kind": kind, "seed": seed, **record}), flush=True)
        if best is None or choice["macro_exact_f1"] > best["macro_exact_f1"] + 1e-12:
            best = record
            stale = 0
            out.mkdir(parents=True, exist_ok=True)
            temp = out / "best.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "kind": kind, "seed": seed,
                        "epoch": epoch, "source_validation": choice}, temp)
            temp.replace(out / "best.pt")
            json_write(out / "validation_grid.json", grid)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best is None:
        raise ValueError("No network checkpoint selected")
    json_write(out / "history.json", history)
    json_write(out / "selection.json", best)
    return best


def _max_source_gold_length(groups: dict) -> int:
    values = [u["end"] - u["start"] for name in ("train", "val")
              for row in groups[name] for u in row["units"]]
    if not values:
        raise ValueError("No source GOLD units")
    return max(values)


def make_examples(db: sqlite3.Connection, rows: list[dict], threshold: float,
                  max_length: int, kind: str, *, training: bool = False) -> list[Example]:
    from tqdm import tqdm
    return [make_example(row, _read_feature(db, row), threshold, max_length,
                         kind, training=training)
            for row in tqdm(rows, desc=f"Candidates {kind}/{threshold}/{len(rows)}", unit="récit")]


def save_predictions(path: Path, rows: list[dict], spans: list[list[tuple[int, int]]],
                     *, model: str, threshold: float | None, seed: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row, doc_spans in zip(rows, spans, strict=True):
            raw = row["raw_text"]
            report.checked_spans(row, [{"start": a, "end": b, "text": raw[a:b]}
                                       for a, b in doc_spans])
            stream.write(json.dumps({"corpus": row["corpus"], "document_id": row["document_id"],
                "model": model, "seed": seed, "threshold": threshold,
                "segments": [{"start": a, "end": b, "text": raw[a:b]}
                             for a, b in doc_spans]}, ensure_ascii=False) + "\n")
    temp.replace(path)


def test_metrics(rows: list[dict], predictions: list[list[tuple[int, int]]],
                 model: str, target: str) -> dict:
    records = [{"document_id": row["document_id"], **report.score(row, spans)}
               for row, spans in zip(rows, predictions, strict=True)]
    result = report.aggregate(records, experiment=VERSION, source="", target=target, model=model)
    return result


def baseline_threshold(db: sqlite3.Connection, rows: list[dict]) -> tuple[dict, list[dict]]:
    values = []
    for threshold in THRESHOLDS:
        predictions = [sat.legacy.spans_from_probabilities(row["raw_text"],
                       sat.get_probs(db, row), threshold) for row in rows]
        counts = collections.defaultdict(collections.Counter)
        for row, pred in zip(rows, predictions, strict=True):
            counts[row["corpus"]].update(sat.score(row, pred))
        if len(counts) != 5:
            raise ValueError("M3 calibration must use five sources")
        f1 = float(np.mean([sat.prf(c["span_tp"], c["span_pred"], c["span_ref"])[2]
                            for c in counts.values()]))
        values.append({"threshold": threshold, "macro_exact_f1": f1})
    return max(values, key=lambda x: (x["macro_exact_f1"],
                                      -abs(x["threshold"] - 0.05), -x["threshold"])), values


def fit(args) -> None:
    import torch
    bundle, m3_root, root = args.bundle.resolve(), args.m3_root.resolve(), args.out.resolve()
    manifest = campaign.check_bundle(bundle)
    target = manifest["fold_order"][args.task_index]
    selection, old_metrics, groups = validate_inputs(bundle, m3_root, target)
    out = root / f"target_{target}"
    bind_folder(out, binding(bundle, m3_root, target, selection))
    if (out / "complete.json").is_file():
        complete = load_json(out / "complete.json")
        if (complete["binding_sha256"] != digest(out / "binding.json") or
                complete["metrics_sha256"] != digest(out / "metrics.json") or
                any(digest(out / rel) != sha for rel, sha in
                    complete["prediction_sha256"].items())):
            raise ValueError("Completed rotation changed")
        print(f"Already complete: {target}")
        return
    feature_cache(bundle, m3_root, target, groups, out, limit=args.limit)
    if args.limit:
        print(f"Smoke features complete: {target}", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Full refiner fit requires one GPU")
    max_length = _max_source_gold_length(groups)
    import transformers
    import wtpsplit
    json_write(out / "parameters.json", {"candidate_thresholds": THRESHOLDS,
        "max_gold_source_char_length": max_length, "seeds": SEEDS, "penalties": PENALTIES,
        "max_epochs": EPOCHS, "patience": PATIENCE, "projection_dim": PROJECTION_DIM,
        "projection_seed": 20261018, "max_sampled_negatives_per_positive": MAX_NEG_PER_POS,
        "mlp_hidden_widths": [256, 64], "dropout": 0.2, "optimizer": "AdamW",
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "deterministic_torch_algorithms": True,
        "merge_loss_weight": 0.5, "fragment_loss_weight": 0.5, "ranking_margin": 1.0,
        "source_domain_sampling": "equal expected documents per source corpus per epoch",
        "candidate_training_threshold": 0.02,
        "unannotated_gap_label": "unknown unless explicitly excluded",
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "transformers_version": transformers.__version__, "wtpsplit_version": wtpsplit.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "training_features_in_sample_for_m3": True,
        "target_test_used_for_selection": False})
    cache = out / "features.sqlite3"
    with closing(sqlite3.connect(cache)) as db, closing(sqlite3.connect(
            (m3_root / f"target_{target}" / "validation_cache" /
             f"{selection['M3_candidate']}.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as val_db:
        m3_choice, m3_grid = baseline_threshold(val_db, groups["val"])
        json_write(out / "E1_validation.json", {"selected": m3_choice, "grid": m3_grid})
        tests = {}
        m3_db = sqlite3.connect(m3_root / f"target_{target}" / "test_probabilities.sqlite3")
        original_m3 = report.prediction_file(
            m3_root / f"target_{target}" / "predictions" / "M3.jsonl", groups["test"])
        try:
            for label, threshold in (("E0", selection["M3_threshold"]),
                                     ("E1", m3_choice["threshold"])):
                spans = [sat.legacy.spans_from_probabilities(row["raw_text"],
                        sat.get_probs(m3_db, row), threshold) for row in groups["test"]]
                if label == "E0" and any(spans[i] != original_m3[row["document_id"]]
                                          for i, row in enumerate(groups["test"])):
                    raise ValueError(f"E0 does not reproduce selected M3 predictions: {target}")
                save_predictions(out / "predictions" / f"{label}.jsonl", groups["test"],
                                 spans, model=label, threshold=threshold, seed=None)
                tests[label] = test_metrics(groups["test"], spans, label, target)
        finally:
            m3_db.close()
        for kind in ("E2", "E3", "E4"):
            # Candidate arrays stay in RAM; the per-document hidden-state cache
            # and selected small networks provide resumability without large
            # duplicated checkpoint files.
            train = make_examples(db, groups["train"], 0.02, max_length, kind, training=True)
            validation = {threshold: make_examples(db, groups["val"], threshold,
                           max_length, kind) for threshold in THRESHOLDS}
            statistics = collections.defaultdict(lambda: collections.Counter())
            for item in train:
                bucket = statistics[item.row["corpus"]]
                bucket["documents"] += 1
                bucket["known_positive"] += int(np.sum(item.labels == 1))
                bucket["known_negative"] += int(np.sum(item.labels == 0))
                bucket["unlabeled"] += int(np.sum(item.labels < 0))
                bucket["merge_constraints"] += len(item.merges)
                bucket["fragment_constraints"] += len(item.fragments)
            json_write(out / f"{kind}_training_counts.json",
                       {corpus: dict(counts) for corpus, counts in statistics.items()})
            test = None
            for seed in SEEDS:
                fit_out = out / kind / f"seed_{seed}"
                if all((fit_out / name).is_file() for name in
                       ("selection.json", "best.pt", "history.json", "validation_grid.json")):
                    chosen = load_json(fit_out / "selection.json")
                else:
                    chosen = train_network(kind, seed, train, validation, fit_out, device)
                state = torch.load(fit_out / "best.pt", map_location="cpu", weights_only=False)
                if (state["kind"] != kind or state["seed"] != seed or
                        state["source_validation"] !=
                        {k: chosen[k] for k in ("threshold", "penalty", "macro_exact_f1")}):
                    raise ValueError("Saved refiner checkpoint does not match source selection")
                model = new_network(kind).to(device)
                model.load_state_dict(state["state_dict"])
                if test is None or test[0].threshold != chosen["threshold"]:
                    test = make_examples(db, groups["test"], chosen["threshold"], max_length, kind)
                predictions = [predict_example(model, example, device, chosen["penalty"])
                               for example in test]
                path = out / "predictions" / f"{kind}_seed_{seed}.jsonl"
                save_predictions(path, groups["test"], predictions, model=kind,
                                 threshold=chosen["threshold"], seed=seed)
                tests[f"{kind}_seed_{seed}"] = test_metrics(groups["test"], predictions, kind, target)
                del model
            del train, validation, test
        json_write(out / "metrics.json", {"target": target, "reference": manifest["reference"],
            "selection_m3_sha256": digest(m3_root / f"target_{target}" / "selection.json"),
            "test": tests, "n_test": len(groups["test"]),
            "exploratory_m3_source_train_in_sample": True})
        json_write(out / "complete.json", {"binding_sha256": digest(out / "binding.json"),
            "metrics_sha256": digest(out / "metrics.json"),
            "prediction_sha256": {str(p.relative_to(out)): digest(p) for p in
                                  sorted((out / "predictions").glob("*.jsonl"))}})
    print(json.dumps({"target": target, "n_test": len(groups["test"]),
                      "E0": tests["E0"]["exact_f1"],
                      "E1": tests["E1"]["exact_f1"]}), flush=True)


def preflight(args) -> None:
    bundle, m3_root, out = args.bundle.resolve(), args.m3_root.resolve(), args.out.resolve()
    manifest = campaign.check_bundle(bundle)
    estimates = []
    for target in manifest["fold_order"]:
        selection, _, groups = validate_inputs(bundle, m3_root, target)
        # Projected states + arrays + SQLite overhead, inflated for prepared examples.
        estimated = sum(len(row["raw_text"]) for rows in groups.values() for row in rows)
        estimates.append(estimated * PROJECTION_DIM * 2 // 3 * 5)
        print(json.dumps({"target": target, "documents": {k: len(v) for k, v in groups.items()},
                          "M3_checkpoint": selection["M3_candidate"]}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    required = max(12 * 1024 ** 3,
                   sum(sorted(estimates, reverse=True)[:2]) + 2 * 1024 ** 3)
    free = shutil.disk_usage(out).free
    if free < required:
        raise RuntimeError(f"Insufficient free space for two concurrent rotations: "
                           f"free={free / 1024**3:.1f} GiB, required={required / 1024**3:.1f} GiB")
    json_write(out / "preflight.json", {"version": VERSION,
        "gold_manifest_sha256": digest(bundle / "manifest.json"),
        "two_task_required_bytes": required, "free_bytes_at_check": free,
        "target_test_documents": sum(manifest["report"][t]["test"] for t in manifest["fold_order"]),
        "exploratory_m3_source_train_in_sample": True})


def smoke(args) -> None:
    import torch
    bundle, m3_root, root = args.bundle.resolve(), args.m3_root.resolve(), args.out.resolve()
    manifest = campaign.check_bundle(bundle)
    target = manifest["fold_order"][0]
    selection, _, full = validate_inputs(bundle, m3_root, target)
    sources = [corpus for corpus in manifest["fold_order"] if corpus != target]
    groups = {}
    for name in ("train", "val"):
        groups[name] = [next(row for row in full[name] if row["corpus"] == corpus
                             and row["raw_text"].strip() and row["units"])
                        for corpus in sources]
    groups["test"] = [next(row for row in full["test"] if row["raw_text"].strip())]
    folder = root / "smoke"
    bind_folder(folder, binding(bundle, m3_root, target, selection))
    if (folder / "smoke_ok.json").is_file():
        if (load_json(folder / "smoke_ok.json")["binding_sha256"] !=
                digest(folder / "binding.json") or
                any(not (folder / f"{kind}.jsonl").is_file()
                    for kind in ("E2", "E3", "E4"))):
            raise ValueError("Existing smoke marker is incomplete or changed")
        print(f"Smoke already passed: {folder}")
        return
    feature_cache(bundle, m3_root, target, groups, folder)
    max_length = _max_source_gold_length(groups)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Smoke test requires a GPU")
    with closing(sqlite3.connect(folder / "features.sqlite3")) as db:
        model_dir, tokenizer_dir = sat.pinned_backbone(bundle)
        model = sat.load_pinned_sat(model_dir, tokenizer_dir,
                                   m3_root / f"target_{target}" / "selected_adapter")
        source_example = groups["train"][0]
        cached = _read_feature(db, source_example)[0]
        native = np.asarray(model.predict_proba(source_example["raw_text"], stride=64),
                            dtype=np.float32)
        if len(cached) != len(native) or np.max(np.abs(cached - native)) > 2e-3:
            raise ValueError("Smoke: hidden-state extraction does not reproduce SaT.predict_proba")
        del model
        for kind in ("E2", "E3", "E4"):
            train = make_examples(db, groups["train"], 0.02, max_length, kind, training=True)
            val = {threshold: make_examples(db, groups["val"], threshold,
                    max_length, kind) for threshold in THRESHOLDS}
            selected = train_network(kind, SEEDS[0], train, val, folder / kind, device)
            state = torch.load(folder / kind / "best.pt", map_location="cpu", weights_only=False)
            model = new_network(kind).to(device)
            model.load_state_dict(state["state_dict"])
            one = make_examples(db, groups["test"], selected["threshold"], max_length, kind)
            spans = predict_example(model, one[0], device, selected["penalty"])
            save_predictions(folder / f"{kind}.jsonl", groups["test"], [spans],
                             model=kind, threshold=selected["threshold"], seed=SEEDS[0])
    json_write(folder / "smoke_ok.json", {"binding_sha256": digest(folder / "binding.json"),
               "tested": ["hidden/probability equivalence", "GPU training", "checkpoint reload",
                          "original offsets", "E2", "E3", "E4"]})


def summary(args) -> None:
    bundle, m3_root, root = args.bundle.resolve(), args.m3_root.resolve(), args.out.resolve()
    manifest = campaign.check_bundle(bundle)
    rows = []
    for target in manifest["fold_order"]:
        selection, _, groups = validate_inputs(bundle, m3_root, target)
        path = root / f"target_{target}"
        done = load_json(path / "complete.json")
        if (done["binding_sha256"] != digest(path / "binding.json") or
                done["metrics_sha256"] != digest(path / "metrics.json") or
                load_json(path / "binding.json") != binding(bundle, m3_root, target, selection)):
            raise ValueError(f"Incomplete or changed refiner rotation: {target}")
        metrics = load_json(path / "metrics.json")
        if metrics["n_test"] != len(groups["test"]):
            raise ValueError(f"Wrong test size: {target}")
        expected = {"E0", "E1"} | {f"{kind}_seed_{seed}"
                                    for kind in ("E2", "E3", "E4") for seed in SEEDS}
        if set(metrics["test"]) != expected or len(done["prediction_sha256"]) != len(expected):
            raise ValueError(f"Missing methods or seeds: {target}")
        for relative, expected_sha in done["prediction_sha256"].items():
            prediction = path / relative
            if digest(prediction) != expected_sha:
                raise ValueError(f"Changed prediction file: {prediction}")
            report.prediction_file(prediction, groups["test"])
        for label, values in metrics["test"].items():
            rows.append({"target": target, "model": label, "n_test": values["n_test"],
                **{k: values[k] for k in ("exact_precision", "exact_recall", "exact_f1",
                    "boundary_precision", "boundary_recall", "boundary_f1", "soft_f1",
                    "soft75_f1", "fusion", "fragmented", "missed", "spurious")}})
    import csv
    path = root / "summary.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    macro = {}
    for label in {row["model"] for row in rows}:
        selected = [row for row in rows if row["model"] == label]
        macro[label] = {field: float(np.mean([r[field] for r in selected]))
                        for field in ("exact_f1", "boundary_f1", "soft_f1", "soft75_f1")}
    by_method = []
    for target in manifest["fold_order"]:
        for kind in ("E2", "E3", "E4"):
            selected = [r for r in rows if r["target"] == target and
                        r["model"].startswith(f"{kind}_seed_")]
            if len(selected) != len(SEEDS):
                raise ValueError(f"Missing seed for {target}/{kind}")
            item = {"target": target, "model": kind, "n_test": selected[0]["n_test"],
                    "n_seeds": len(SEEDS)}
            for field in ("exact_f1", "boundary_f1", "soft_f1", "soft75_f1"):
                values = [r[field] for r in selected]
                item[f"{field}_mean"] = float(np.mean(values))
                item[f"{field}_std"] = float(np.std(values, ddof=1))
            by_method.append(item)
    path = root / "summary_by_method.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_method[0]))
        writer.writeheader()
        writer.writerows(by_method)
    temp.replace(path)
    macro_method = {}
    for kind in ("E2", "E3", "E4"):
        seed_scores = [macro[f"{kind}_seed_{seed}"]["exact_f1"] for seed in SEEDS]
        macro_method[kind] = {
            "exact_f1_mean": float(np.mean(seed_scores)),
            "exact_f1_std": float(np.std(seed_scores, ddof=1)),
            "gain_vs_E0_mean": float(np.mean(seed_scores) - macro["E0"]["exact_f1"]),
        }
    json_write(root / "summary.json", {"version": VERSION, "n_test": 2690,
        "macro_domain_by_seed": macro, "per_domain_seed_summary": by_method,
        "macro_method": macro_method,
        "exploratory_m3_source_train_in_sample": True,
        "interpretation": "Target tests are unseen by M3 and refiner; refiner training features are in-sample for M3."})
    lines = ["# Raffinement factuel guidé par M3 — résultats exploratoires", "",
             "Les 2 690 récits de test sont identiques entre méthodes. Chaque cible",
             "est absente des sources utilisées pour l'entraînement et la validation.",
             "Les caractéristiques M3 des récits sources d'entraînement restent",
             "**en échantillon pour M3** ; ces scores ne suffisent pas à établir une",
             "performance définitive pour l'article.", "",
             "## Moyenne macro des six cibles", "",
             "| Méthode | Exact-span F1 | Boundary F1 | Soft Span F1 |",
             "| --- | ---: | ---: | ---: |"]
    for label in ("E0", "E1"):
        value = macro[label]
        lines.append(f"| {label} | {value['exact_f1']:.4f} | "
                     f"{value['boundary_f1']:.4f} | {value['soft_f1']:.4f} |")
    for kind in ("E2", "E3", "E4"):
        values = [macro[f"{kind}_seed_{seed}"] for seed in SEEDS]
        cell = lambda field: (f"{np.mean([v[field] for v in values]):.4f} ± "
                              f"{np.std([v[field] for v in values], ddof=1):.4f}")
        lines.append(f"| {kind} | {cell('exact_f1')} | {cell('boundary_f1')} | "
                     f"{cell('soft_f1')} |")
    lines.extend(["", "## Exact-span F1 par domaine cible", "",
                  "| Cible | E0 | E1 | E2 (3 graines) | E3 (3 graines) | E4 (3 graines) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for target in manifest["fold_order"]:
        base = {r["model"]: r for r in rows if r["target"] == target}
        learned = {r["model"]: r for r in by_method if r["target"] == target}
        lines.append(f"| {target} | {base['E0']['exact_f1']:.4f} | "
                     f"{base['E1']['exact_f1']:.4f} | " + " | ".join(
                         f"{learned[k]['exact_f1_mean']:.4f} ± "
                         f"{learned[k]['exact_f1_std']:.4f}"
                         for k in ("E2", "E3", "E4")) + " |")
    lines.extend(["", "`summary.csv` contient précision, rappel, F1, Soft Span,",
                  "fusions et fragmentations pour chaque rotation et graine.",
                  "`summary_by_method.csv` donne les moyennes et écarts-types entre graines.", ""])
    temporary = root / "RESULTS.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(root / "RESULTS.md")
    print(json.dumps(macro, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "smoke", "fit", "summary"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--m3-root", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "fit":
            command.add_argument("--task-index", type=int, required=True)
            command.add_argument("--limit", type=int)
    args = parser.parse_args()
    {"preflight": preflight, "smoke": smoke, "fit": fit,
     "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
