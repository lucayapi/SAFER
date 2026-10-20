#!/usr/bin/env python3
"""Validate and stack the role-specific instructed embeddings for factor search."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TEXT_ROOT = HERE.parent
UNITS = TEXT_ROOT / "dataset/families/btp/data_btp_building_construction_and_masonry.csv"
DEFAULT_EMBEDDING_DIR = TEXT_ROOT / "embeddings/instructed/btp_building_construction_and_masonry"
ROLE_SUFFIXES = {
    "B": "btp_B_mechanism_instruct.csv",
    "A0": "btp_A0_work_situation_instruct.csv",
    "A1": "btp_A1_adverse_condition_instruct.csv",
    "C": "btp_C_health_consequence_instruct.csv",
}


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def accepted_flag(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing the role-specific embedding CSVs.",
    )
    parser.add_argument(
        "--model-prefix", default="Qwen3-Embedding-0.6B",
        help="Filename prefix shared by role CSVs, e.g. Qwen3-Embedding-4B.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Merged output CSV (default: <embedding-dir>/<model-prefix>_btp_all_roles_instruct.csv).",
    )
    args = parser.parse_args()
    embedding_dir = args.embedding_dir if args.embedding_dir.is_absolute() else (TEXT_ROOT / args.embedding_dir).resolve()
    output = args.output if args.output is not None else embedding_dir / f"{args.model_prefix}_btp_all_roles_instruct.csv"
    if not output.is_absolute():
        output = (TEXT_ROOT / output).resolve()
    role_files = {role: f"{args.model_prefix}_{suffix}" for role, suffix in ROLE_SUFFIXES.items()}

    if not UNITS.is_file():
        raise FileNotFoundError(f"Unit file is missing: {UNITS}")
    units = pd.read_csv(UNITS, dtype={"fact_id": str}, low_memory=False)
    required = {"fact_id", "sentence", "pred_label", "pred_ok"}
    missing = sorted(required - set(units.columns))
    if missing:
        raise ValueError(f"Missing required unit columns: {missing}")
    units["_id"] = normalize_id(units["fact_id"])
    units["_role"] = units["pred_label"].astype(str).str.strip().str.upper()
    eligible = accepted_flag(units["pred_ok"]) & units["_role"].isin(role_files) & units["sentence"].fillna("").astype(str).str.strip().ne("")
    if units.loc[eligible, "_id"].duplicated().any():
        examples = units.loc[eligible & units["_id"].duplicated(keep=False), "_id"].head(10).tolist()
        raise ValueError(f"Eligible unit IDs are not unique: {examples}")

    merged: list[pd.DataFrame] = []
    shared_dimensions: list[str] | None = None
    for role, filename in role_files.items():
        expected = set(units.loc[eligible & units["_role"].eq(role), "_id"])
        path = embedding_dir / filename
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

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        result.to_csv(temp, index=False, float_format="%.9g")
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"[merge] verified {len(result)} unique units across roles {','.join(role_files)}; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
