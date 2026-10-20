#!/usr/bin/env python3
"""Validate and stack the role-specific instructed embeddings for factor search."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TEXT_ROOT = HERE.parent
UNITS = TEXT_ROOT / "dataset/families/btp/data_btp_building_construction_and_masonry.csv"
EMBEDDING_DIR = TEXT_ROOT / "embeddings/instructed/btp_building_construction_and_masonry"
OUTPUT = EMBEDDING_DIR / "Qwen3-Embedding-0.6B_btp_all_roles_instruct.csv"
ROLE_FILES = {
    "B": "Qwen3-Embedding-0.6B_btp_B_mechanism_instruct.csv",
    "A0": "Qwen3-Embedding-0.6B_btp_A0_work_situation_instruct.csv",
    "A1": "Qwen3-Embedding-0.6B_btp_A1_adverse_condition_instruct.csv",
    "C": "Qwen3-Embedding-0.6B_btp_C_health_consequence_instruct.csv",
}


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def accepted_flag(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def main() -> None:
    if not UNITS.is_file():
        raise FileNotFoundError(f"Unit file is missing: {UNITS}")
    units = pd.read_csv(UNITS, dtype={"fact_id": str}, low_memory=False)
    required = {"fact_id", "sentence", "pred_label", "pred_ok"}
    missing = sorted(required - set(units.columns))
    if missing:
        raise ValueError(f"Missing required unit columns: {missing}")
    units["_id"] = normalize_id(units["fact_id"])
    units["_role"] = units["pred_label"].astype(str).str.strip().str.upper()
    eligible = accepted_flag(units["pred_ok"]) & units["_role"].isin(ROLE_FILES) & units["sentence"].fillna("").astype(str).str.strip().ne("")
    if units.loc[eligible, "_id"].duplicated().any():
        examples = units.loc[eligible & units["_id"].duplicated(keep=False), "_id"].head(10).tolist()
        raise ValueError(f"Eligible unit IDs are not unique: {examples}")

    merged: list[pd.DataFrame] = []
    shared_dimensions: list[str] | None = None
    for role, filename in ROLE_FILES.items():
        expected = set(units.loc[eligible & units["_role"].eq(role), "_id"])
        path = EMBEDDING_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing role-{role} instructed embeddings: {path}. Run the all-role embedding job first.")
        frame = pd.read_csv(path, dtype={"doc_id": str}, low_memory=False)
        if "doc_id" not in frame:
            raise ValueError(f"Missing doc_id in role-{role} embeddings: {path}")
        frame["_id"] = normalize_id(frame["doc_id"])
        dims = [col for col in frame.columns if str(col).startswith("dim_")]
        if not dims:
            raise ValueError(f"No dim_ embedding columns in {path}")
        if frame["_id"].duplicated().any():
            raise ValueError(f"Duplicate doc_id in role-{role} embeddings: {path}")
        actual = set(frame["_id"])
        missing_ids = sorted(expected - actual)
        extra_ids = sorted(actual - expected)
        if missing_ids or extra_ids:
            raise ValueError(
                f"Role-{role} embedding alignment mismatch: expected={len(expected)}, actual={len(actual)}, "
                f"missing examples={missing_ids[:5]}, extra examples={extra_ids[:5]}"
            )
        if shared_dimensions is None:
            shared_dimensions = dims
        elif dims != shared_dimensions:
            raise ValueError(f"Embedding dimensions/order differ for role-{role}; all role exports must share one schema")
        values = frame[dims].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite vector values in role-{role} embeddings: {path}")
        part = frame.loc[:, ["_id", *dims]].copy().rename(columns={"_id": "doc_id"})
        merged.append(part)
        print(f"[merge] role={role} eligible_units={len(expected)} embeddings={len(part)} dims={len(dims)}", flush=True)

    result = pd.concat(merged, ignore_index=True)
    result["doc_id"] = normalize_id(result["doc_id"])
    if result["doc_id"].duplicated().any():
        raise ValueError("The role-specific exports contain overlapping unit IDs")
    all_expected = set(units.loc[eligible, "_id"])
    if set(result["doc_id"]) != all_expected:
        raise ValueError(f"Combined export coverage mismatch: expected {len(all_expected)} IDs, got {len(result)}")
    result["doc_id"] = pd.to_numeric(result["doc_id"], errors="raise").astype(np.int64)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_name(f".{OUTPUT.name}.{uuid.uuid4().hex}.tmp")
    try:
        result.to_csv(temp, index=False, float_format="%.9g")
        os.replace(temp, OUTPUT)
    finally:
        temp.unlink(missing_ok=True)
    print(f"[merge] verified {len(result)} unique units across roles {','.join(ROLE_FILES)}; wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
