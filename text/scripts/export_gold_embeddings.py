"""Validated, resumable and atomic frozen-encoder export for GOLD narrative_v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scripts.export_corpus_embeddings import (
    ROOT_DIR, build_contrastive_config, load_export_config,
)
from safer_core.embedding_paths import embedding_export_path
from safer_core.test_corpus import resolve_test_corpus
from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.export import export_text_embeddings
from scgm_text.data_metadata import GOLD_V3_SNAPSHOT_SHA256, load_filtered_metadata

CONFIGS = {
    "qwen3": "export_embeddings.yaml",
    "e5": "export_embeddings_multilingual_e5_large.yaml",
    "camembert": "export_embeddings_camembert.yaml",
}
EXPECTED_GOLD_SHA = GOLD_V3_SNAPSHOT_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_signature(data_csv: Path, corpus: str, cfg: dict) -> dict:
    manifest = json.loads((data_csv.parent / "gold_v3_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["corpora"][corpus]
    actual = sha256(data_csv)
    if (manifest["gold_version"] != "narrative_v3"
            or manifest["snapshot_sha256"] != EXPECTED_GOLD_SHA
            or entry["file"] != data_csv.name or entry["sha256"] != actual):
        raise ValueError(f"GOLD dataset/manifest mismatch: {data_csv}")
    return {
        "corpus": corpus,
        "gold_snapshot_sha256": manifest["snapshot_sha256"],
        "dataset_sha256": actual,
        "rows": entry["units"],
        "backbone_name": cfg["backbone_name"],
        "backbone_id": cfg["backbone_id"],
        "input_prefix": cfg.get("input_prefix", ""),
        "text_col": cfg.get("text_col", "sentence"),
        "max_seq_length": int(cfg.get("max_seq_length", 256)),
        "use_prompt": bool(cfg.get("use_prompt", False)),
    }


def validate_export(path: Path, expected_ids: list[int]) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    dims = [column for column in header if column.startswith("dim_")]
    if not dims or header != ["doc_id", *dims] or len(dims) != len(set(dims)):
        raise ValueError(f"Invalid embedding columns: {path}")
    actual = []
    for chunk in pd.read_csv(path, usecols=["doc_id"], chunksize=10000):
        actual.extend(chunk["doc_id"].astype(int).tolist())
    if actual != expected_ids or len(actual) != len(set(actual)):
        raise ValueError(f"Embedding IDs/order mismatch: {path}")
    return len(dims)


def export_gold(backbone: str, corpus: str, *, force: bool = False, root: Path = ROOT_DIR) -> Path:
    if backbone not in CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone}")
    cfg = load_export_config(root / "configs" / CONFIGS[backbone])
    if cfg["backbone_id"] != backbone or cfg["output_root"] != "embeddings/gold_v3":
        raise ValueError("Encoder configuration is not the GOLD v3 export configuration")
    spec = resolve_test_corpus(corpus, anchor=root, require_files=True, require_emb_csv=False)
    signature = source_signature(spec.data_csv, corpus, cfg)
    metadata = load_filtered_metadata(str(spec.data_csv), text_col=signature["text_col"])
    expected_ids = metadata["doc_id"].astype(int).tolist()
    if len(expected_ids) != signature["rows"] or len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"Filtered rows/IDs differ from the GOLD manifest: {corpus}")
    dest = embedding_export_path(corpus, backbone_name=cfg["backbone_name"],
                                 output_root=cfg["output_root"], backbone_id=backbone, anchor=root)
    sidecar = dest.with_suffix(".manifest.json")
    if dest.exists() and sidecar.exists() and not force:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if existing.get("signature") != signature or existing.get("export_sha256") != sha256(dest):
            raise ValueError(f"Stale export; inspect before re-running with --force: {dest}")
        validate_export(dest, expected_ids)
        print(f"[skip] verified {backbone}/{corpus}: {dest}", flush=True)
        return dest
    if dest.exists() and not sidecar.exists() and not force:
        print(f"[resume] incomplete export without manifest; recomputing {dest}", flush=True)
    cfg_obj = build_contrastive_config(cfg, spec.data_csv)
    dataset = prepare_text_dataset(cfg_obj)
    if dataset.metadata_df["doc_id"].astype(int).tolist() != expected_ids:
        raise ValueError("Encoder input order differs from verified GOLD")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    temp_sidecar = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        export_text_embeddings(cfg_obj, dataset, temp, batch_size=cfg_obj.encode_batch_size, show_progress=True)
        dims = validate_export(temp, expected_ids)
        record = {"signature": signature, "dimensions": dims, "export_sha256": sha256(temp)}
        temp_sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, dest)
        os.replace(temp_sidecar, sidecar)
    finally:
        temp.unlink(missing_ok=True)
        temp_sidecar.unlink(missing_ok=True)
    print(f"[ok] {backbone}/{corpus}: {len(expected_ids)} vectors, {dims} dimensions → {dest}", flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export_gold(args.backbone, args.corpus, force=args.force)


if __name__ == "__main__":
    main()
