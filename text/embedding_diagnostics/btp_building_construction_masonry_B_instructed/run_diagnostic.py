#!/usr/bin/env python3
"""Cosine-neighbour diagnostic for instructed Qwen B-mechanism embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
UNITS_PATH = ROOT / "text/dataset/families/btp/data_btp_building_construction_and_masonry.csv"
EMBEDDINGS_PATH = ROOT / "text/embeddings/btp_building_construction_and_masonry/Qwen3-Embedding-0.6B_btp_B_mechanism_instruct.csv"
MANIFEST_PATH = EMBEDDINGS_PATH.with_suffix(".manifest.json")
OUT_DIR = Path(__file__).resolve().parent / "results"
N_ANCHORS = 50
N_NEIGHBOURS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def choose_diverse_anchors(vectors: np.ndarray, accident_ids: np.ndarray, n: int) -> np.ndarray:
    """Deterministic farthest-first coverage, with at most one anchor per accident."""
    # The exported vectors are L2-normalised. Renormalise to guard against rounding.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarity = vectors @ vectors.T
    # Start from the semantic medoid, then greedily cover the least represented point.
    first = int(np.argmax(similarity.mean(axis=1)))
    selected = [first]
    selected_accidents = {str(accident_ids[first])}
    nearest_selected = similarity[first].copy()
    while len(selected) < n:
        eligible = np.array([str(a) not in selected_accidents for a in accident_ids])
        scores = nearest_selected.copy()
        scores[~eligible] = np.inf
        nxt = int(np.argmin(scores))
        selected.append(nxt)
        selected_accidents.add(str(accident_ids[nxt]))
        nearest_selected = np.maximum(nearest_selected, similarity[nxt])
    return np.asarray(selected, dtype=int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(UNITS_PATH, dtype={"fact_id": "Int64", "accident_id": "string"})
    emb = pd.read_csv(EMBEDDINGS_PATH, dtype={"doc_id": "Int64"})
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    required_unit = {"accident_id", "fact_id", "sentence", "pred_label"}
    if not required_unit.issubset(units.columns):
        raise ValueError(f"Missing unit columns: {sorted(required_unit - set(units.columns))}")
    if "doc_id" not in emb:
        raise ValueError("Embedding export has no doc_id column")
    vector_cols = [c for c in emb.columns if c.startswith("dim_")]
    if not vector_cols:
        raise ValueError("No dim_* embedding columns found")
    if units.fact_id.duplicated().any() or emb.doc_id.duplicated().any():
        raise ValueError("Duplicate fact_id or doc_id; refusing a non-unique join")

    b_units = units.loc[units.pred_label.eq("B"), ["accident_id", "fact_id", "sentence"]].copy()
    b_units["fact_id"] = b_units.fact_id.astype("Int64")
    if set(b_units.fact_id.dropna()) != set(emb.doc_id.dropna()):
        missing = sorted(set(b_units.fact_id.dropna()) - set(emb.doc_id.dropna()))
        extra = sorted(set(emb.doc_id.dropna()) - set(b_units.fact_id.dropna()))
        raise ValueError(f"B-unit/embedding IDs do not align: missing={len(missing)}, extra={len(extra)}")

    data = b_units.merge(emb, left_on="fact_id", right_on="doc_id", how="inner", validate="one_to_one")
    X = data[vector_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite embedding values found")
    norms = np.linalg.norm(X, axis=1)
    if (norms < 1e-12).any():
        raise ValueError("Zero-norm embedding found")
    X /= norms[:, None]

    anchor_idx = choose_diverse_anchors(X, data.accident_id.astype(str).to_numpy(), N_ANCHORS)
    # Cosine similarities between all B-only units. We exclude every unit from
    # the anchor's accident to avoid retrieving another segment of the same report.
    similarities = X[anchor_idx] @ X.T
    rows = []
    anchor_rows = []
    for ai, idx in enumerate(anchor_idx, start=1):
        anchor = data.iloc[idx]
        eligible = data.accident_id.astype(str).to_numpy() != str(anchor.accident_id)
        eligible[idx] = False
        scores = similarities[ai - 1].copy()
        scores[~eligible] = -np.inf
        top = np.argpartition(-scores, N_NEIGHBOURS - 1)[:N_NEIGHBOURS]
        top = top[np.argsort(-scores[top])]
        anchor_rows.append({
            "anchor_no": ai,
            "anchor_fact_id": int(anchor.fact_id),
            "anchor_accident_id": str(anchor.accident_id),
            "anchor_text": anchor.sentence,
            "anchor_selection": "deterministic farthest-first coverage in instructed B embedding space",
        })
        for rank, ni in enumerate(top, start=1):
            neighbour = data.iloc[int(ni)]
            rows.append({
                "anchor_no": ai,
                "anchor_fact_id": int(anchor.fact_id),
                "anchor_accident_id": str(anchor.accident_id),
                "anchor_text": anchor.sentence,
                "rank": rank,
                "neighbour_fact_id": int(neighbour.fact_id),
                "neighbour_accident_id": str(neighbour.accident_id),
                "cosine_similarity": float(scores[ni]),
                "neighbour_text": neighbour.sentence,
                "mechanism_judgement": "",
                "review_note": "",
            })

    neighbours = pd.DataFrame(rows)
    anchors = pd.DataFrame(anchor_rows)
    neighbours.to_csv(OUT_DIR / "B_top10_cosine_neighbours_for_review.csv", index=False, encoding="utf-8-sig")
    anchors.to_csv(OUT_DIR / "B_reference_units.csv", index=False, encoding="utf-8-sig")

    per_anchor = neighbours.groupby("anchor_no").cosine_similarity.agg(["mean", "min", "max"]).reset_index()
    per_anchor.to_csv(OUT_DIR / "B_top10_similarity_summary.csv", index=False)
    audit = {
        "corpus": "BTP / Building construction and masonry",
        "role": "B",
        "embedding_file": str(EMBEDDINGS_PATH.relative_to(ROOT)),
        "embedding_model": manifest.get("model"),
        "embedding_instruction": manifest.get("instruction"),
        "rows_in_source_units": int(len(units)),
        "role_B_units": int(len(b_units)),
        "embedding_rows": int(len(emb)),
        "embedding_dimensions": int(len(vector_cols)),
        "id_join": "fact_id = doc_id, one-to-one",
        "alignment_complete": True,
        "anchor_selection": "deterministic farthest-first coverage in cosine space; at most one reference per accident",
        "anchor_count": int(N_ANCHORS),
        "neighbours_per_anchor": int(N_NEIGHBOURS),
        "neighbour_pool": "role B only; exclude all units from the anchor accident",
        "pairs_exported": int(len(neighbours)),
        "similarity_summary": {
            "top1_mean": float(neighbours.loc[neighbours["rank"].eq(1), "cosine_similarity"].mean()),
            "top10_mean": float(neighbours.cosine_similarity.mean()),
            "top10_minimum": float(neighbours.cosine_similarity.min()),
            "top10_maximum": float(neighbours.cosine_similarity.max()),
        },
        "annotation_status": "pairwise same/related/different mechanism judgement intentionally left blank for substantive review",
        "source_sha256": sha256(UNITS_PATH),
        "embedding_sha256": sha256(EMBEDDINGS_PATH),
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
