#!/usr/bin/env python3
"""Cosine-neighbour diagnostic using ordinary (non-instructed) Qwen embeddings.

Uses the exact same 50 reference fact IDs as the instructed-embedding audit,
then retrieves the ten closest B-only units while excluding the reference
accident. This keeps the comparison paired at the reference level.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
UNITS_PATH = ROOT / "text/dataset/families/btp/data_btp_building_construction_and_masonry.csv"
EMBEDDINGS_PATH = ROOT / "text/embeddings/families/btp/Qwen3-Embedding-0.6B_btp_building_construction_and_masonry.csv"
REFERENCE_PATH = ROOT / "text/embedding_diagnostics/btp_building_construction_masonry_B_instructed/results/B_reference_units.csv"
INSTRUCTED_REVIEW_PATH = ROOT / "text/embedding_diagnostics/btp_building_construction_masonry_B_instructed/results/B_top10_cosine_neighbours_reviewed.csv"
OUT_DIR = HERE / "results"
N_NEIGHBOURS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(UNITS_PATH, dtype={"fact_id": "Int64", "accident_id": "string"})
    emb = pd.read_csv(EMBEDDINGS_PATH, dtype={"doc_id": "Int64"})
    refs = pd.read_csv(REFERENCE_PATH, dtype={"anchor_fact_id": "Int64"})
    required = {"accident_id", "fact_id", "sentence", "pred_label"}
    if not required.issubset(units.columns):
        raise ValueError(f"Missing unit columns: {sorted(required - set(units.columns))}")
    vector_cols = [c for c in emb.columns if c.startswith("dim_")]
    if "doc_id" not in emb or not vector_cols:
        raise ValueError("Embedding export must contain doc_id and dim_* columns")
    if units.fact_id.duplicated().any() or emb.doc_id.duplicated().any():
        raise ValueError("Duplicate fact_id or doc_id; refusing non-unique join")

    b_units = units.loc[units.pred_label.eq("B"), ["accident_id", "fact_id", "sentence"]].copy()
    b_units["fact_id"] = b_units.fact_id.astype("Int64")
    b_ids, e_ids = set(b_units.fact_id.dropna()), set(emb.doc_id.dropna())
    # The normal export is for the full family (all roles); every B unit must
    # occur exactly once, while non-B IDs are expected and are filtered out.
    missing = b_ids - e_ids
    if missing:
        raise ValueError(f"Normal embeddings miss {len(missing)} B-unit IDs")
    if not set(refs.anchor_fact_id.dropna()).issubset(b_ids):
        raise ValueError("Reference IDs are not all present among B units")
    data = b_units.merge(emb, left_on="fact_id", right_on="doc_id", how="left", validate="one_to_one")
    X = data[vector_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite embedding values found")
    norms = np.linalg.norm(X, axis=1)
    if (norms < 1e-12).any():
        raise ValueError("Zero-norm embedding found")
    X /= norms[:, None]

    row_by_id = {int(fid): i for i, fid in enumerate(data.fact_id)}
    all_accidents = data.accident_id.astype(str).to_numpy()
    rows = []
    ref_rows = []
    for _, ref in refs.iterrows():
        idx = row_by_id[int(ref.anchor_fact_id)]
        anchor = data.iloc[idx]
        scores = X @ X[idx]
        eligible = all_accidents != str(anchor.accident_id)
        eligible[idx] = False
        scores[~eligible] = -np.inf
        top = np.argpartition(-scores, N_NEIGHBOURS - 1)[:N_NEIGHBOURS]
        top = top[np.argsort(-scores[top])]
        ref_rows.append({"anchor_no": int(ref.anchor_no), "anchor_fact_id": int(anchor.fact_id),
                         "anchor_accident_id": str(anchor.accident_id), "anchor_text": anchor.sentence,
                         "anchor_selection": "same references selected for instructed-embedding diagnostic"})
        for rank, ni in enumerate(top, 1):
            neighbour = data.iloc[int(ni)]
            rows.append({"anchor_no": int(ref.anchor_no), "anchor_fact_id": int(anchor.fact_id),
                         "anchor_accident_id": str(anchor.accident_id), "anchor_text": anchor.sentence,
                         "rank": rank, "neighbour_fact_id": int(neighbour.fact_id),
                         "neighbour_accident_id": str(neighbour.accident_id),
                         "cosine_similarity": float(scores[ni]), "neighbour_text": neighbour.sentence,
                         "mechanism_judgement": "", "review_note": ""})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "B_top10_cosine_neighbours_for_review.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ref_rows).to_csv(OUT_DIR / "B_reference_units.csv", index=False, encoding="utf-8-sig")
    summary = out.groupby("rank").cosine_similarity.agg(["mean", "median", "min", "max"]).reset_index()
    summary.to_csv(OUT_DIR / "B_similarity_by_rank.csv", index=False)

    instructed = pd.read_csv(INSTRUCTED_REVIEW_PATH)
    # Pair-level paired cosine comparison for the same anchor and rank numbers.
    paired = out.merge(instructed[["anchor_no", "rank", "cosine_similarity"]], on=["anchor_no", "rank"],
                       suffixes=("_normal", "_instructed"), validate="one_to_one")
    paired.to_csv(OUT_DIR / "B_paired_similarity_comparison.csv", index=False)
    audit = {
        "corpus": "BTP / Building construction and masonry", "role": "B",
        "embedding_file": str(EMBEDDINGS_PATH.relative_to(ROOT)),
        "embedding_type": "ordinary Qwen family embeddings (no instruction)",
        "rows_in_source_units": int(len(units)), "role_B_units": int(len(b_units)),
        "embedding_rows": int(len(emb)), "embedding_dimensions": int(len(vector_cols)),
        "B_unit_embedding_coverage": int(len(b_units)), "id_join": "fact_id = doc_id, one-to-one",
        "anchor_ids_reused": True, "anchor_count": int(len(refs)),
        "neighbours_per_anchor": N_NEIGHBOURS,
        "neighbour_pool": "role B only; exclude all units from the anchor accident",
        "pairs_exported": int(len(out)),
        "similarity": {"top1_mean": float(out.loc[out['rank'].eq(1), 'cosine_similarity'].mean()),
                       "top10_mean": float(out.cosine_similarity.mean()),
                       "top10_min": float(out.cosine_similarity.min()),
                       "top10_max": float(out.cosine_similarity.max())},
        "paired_cosine_delta_normal_minus_instructed": {
            "mean": float((paired.cosine_similarity_normal-paired.cosine_similarity_instructed).mean()),
            "top1_mean": float((paired.loc[paired['rank'].eq(1), 'cosine_similarity_normal']-
                                paired.loc[paired['rank'].eq(1), 'cosine_similarity_instructed']).mean())},
        "source_sha256": sha256(UNITS_PATH), "embedding_sha256": sha256(EMBEDDINGS_PATH),
        "reference_source": str(REFERENCE_PATH.relative_to(ROOT)),
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
