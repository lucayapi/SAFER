"""Chargement et filtrage des métadonnées texte (BTP, corpus test)."""

from __future__ import annotations

from typing import List, Optional, Sequence
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from scgm_text.utils_io import create_doc_id_if_missing, parse_bool_column

LABEL2ID = {"A0": 0, "A1": 1, "B": 2, "C": 3, "AUTRE": 4}
GOLD_V3_SNAPSHOT_SHA256 = "4845dbec927c5cd1161360729382ac5497c3f5c41d1562b03d5020c2f20bdca8"
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
VALID_LABELS = set(LABEL2ID.keys())


def _metadata_usecols(
    header: Sequence[str],
    *,
    label_col: str,
    pred_ok_col: str,
    group_col: str,
    text_col: Optional[str],
) -> List[str]:
    wanted = {label_col, pred_ok_col, group_col, "doc_id", "fact_id"}
    if text_col:
        wanted.add(text_col)
    else:
        wanted.update(("sentence", "accident_summary", "text"))
    return [c for c in header if c in wanted]


def resolve_text_column(metadata_df: pd.DataFrame, text_col: Optional[str] = None) -> str:
    if text_col and text_col in metadata_df.columns:
        return text_col
    for candidate in ("sentence", "accident_summary", "text"):
        if candidate in metadata_df.columns:
            return candidate
    raise ValueError(
        "Colonne texte introuvable. Fournir text_col ou ajouter sentence / accident_summary."
    )


def load_filtered_metadata(
    data_csv: str,
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    text_col: Optional[str] = None,
) -> pd.DataFrame:
    header = pd.read_csv(data_csv, nrows=0).columns.tolist()
    usecols = _metadata_usecols(
        header,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        text_col=text_col,
    )
    if not usecols:
        raise ValueError(
            f"Aucune colonne utile dans {data_csv} "
            f"(attendu au moins {label_col}, {pred_ok_col}, {group_col} et une colonne texte)."
        )
    data_df = pd.read_csv(data_csv, usecols=usecols)
    data_df = create_doc_id_if_missing(data_df)

    ok_mask = parse_bool_column(data_df[pred_ok_col])
    label_series = data_df[label_col].astype(str).str.strip()
    invalid_str = label_series.str.lower().isin({"nan", "none", ""})
    valid_label_mask = label_series.notna() & ~invalid_str & label_series.isin(VALID_LABELS)
    filtered = data_df.loc[ok_mask & valid_label_mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    filtered["label_id"] = filtered[label_col].astype(str).str.strip().map(LABEL2ID)
    if filtered["label_id"].isna().any():
        bad = filtered.loc[filtered["label_id"].isna(), label_col].astype(str).unique().tolist()
        raise ValueError(f"Labels non mappés vers LABEL2ID : {bad[:20]}")
    filtered["label_id"] = filtered["label_id"].astype(np.int64)
    filtered["row_id"] = np.arange(len(filtered), dtype=np.int64)
    data_path = Path(data_csv)
    gold_manifest = data_path.parent / "gold_v3_manifest.json"
    if gold_manifest.is_file():
        manifest = json.loads(gold_manifest.read_text(encoding="utf-8"))
        if manifest.get("snapshot_sha256") != GOLD_V3_SNAPSHOT_SHA256:
            raise ValueError(f"Unexpected GOLD snapshot provenance: {gold_manifest}")
        entry = manifest.get("corpora", {}).get(data_path.stem.removeprefix("data_"))
        if data_path.name.startswith("data_") and data_path.suffix == ".csv":
            if not entry or entry.get("file") != data_path.name:
                raise ValueError(f"Dataset is not registered in GOLD v3 manifest: {data_path}")
            digest_hash = hashlib.sha256()
            with data_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_hash.update(block)
            digest = digest_hash.hexdigest()
            if digest != entry["sha256"] or len(filtered) != entry["units"]:
                raise ValueError(f"GOLD dataset modified or rows filtered: {data_path}")
            filtered.attrs["gold_dataset_sha256"] = digest
            filtered.attrs["gold_snapshot_sha256"] = manifest["snapshot_sha256"]
    if group_col not in filtered.columns:
        raise ValueError(f"Missing group column: {group_col}")
    return filtered
