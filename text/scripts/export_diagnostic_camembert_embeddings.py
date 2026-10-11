"""Export and verify frozen CamemBERT vectors for one diagnostic corpus.

The export is staged and installed only after its rows match the filtered
annotated corpus. This avoids accepting a truncated file from an interrupted
earlier job. No classifier is fitted and no target labels select a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safer_core.embedding_paths import embedding_export_path
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.data_metadata import load_filtered_metadata
from scripts.export_corpus_embeddings import export_corpus, load_export_config

CORPORA = ("btp", "metallurgie", "caou", "nicollin")
CONFIG = ROOT / "configs" / "export_embeddings_camembert.yaml"


def write_inventory(result: dict, corpus_id: str) -> None:
    metadata = ROOT / "output" / "diagnostics" / "embedding_exports"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / f"camembert_{corpus_id}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def verify_export(path: Path, corpus_id: str, expected_dim: int = 768) -> dict:
    """Check dimensions, finite values, and ordered one-to-one doc_id match."""
    spec = resolve_test_corpus(corpus_id, anchor=ROOT, require_files=True, require_emb_csv=False)
    expected = load_filtered_metadata(
        str(spec.data_csv), text_col="sentence", label_col="pred_label",
        pred_ok_col="pred_ok", group_col="accident_id",
    )
    expected_ids = expected["doc_id"].to_numpy(dtype=np.int64)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dims = [f"dim_{index:04d}" for index in range(1, expected_dim + 1)]
    if header != ["doc_id", *dims]:
        raise ValueError(f"{path}: expected doc_id and {expected_dim} ordered dimensions")
    offset = 0
    for chunk in pd.read_csv(path, chunksize=2048):
        n_rows = len(chunk)
        if offset + n_rows > len(expected_ids):
            raise ValueError(f"{path}: more rows than the filtered {corpus_id} corpus")
        actual_ids = chunk["doc_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_ids, expected_ids[offset:offset + n_rows]):
            raise ValueError(f"{path}: doc_id mismatch at row {offset}")
        values = chunk[dims].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite embedding values at row {offset}")
        offset += n_rows
    if offset != len(expected_ids):
        raise ValueError(f"{path}: {offset} rows, expected {len(expected_ids)}")
    return {"corpus": corpus_id, "rows": offset, "dimensions": expected_dim,
            "source_csv": str(spec.data_csv), "embedding_csv": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=CORPORA)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-export even if a valid file exists")
    args = parser.parse_args()
    cfg = load_export_config(CONFIG)
    if cfg.get("backbone_name") != "almanach/camembert-base":
        raise ValueError("Unexpected backbone in CamemBERT export config")
    final = embedding_export_path(
        args.corpus, backbone_name=cfg["backbone_name"],
        output_root=cfg["output_root"], backbone_id=cfg["backbone_id"], anchor=ROOT,
    )
    if args.check_only:
        print(json.dumps(verify_export(final, args.corpus), indent=2))
        return
    if final.is_file() and not args.force:
        try:
            result = verify_export(final, args.corpus)
        except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
            print(f"[re-export] existing file failed validation: {exc}", flush=True)
        else:
            write_inventory(result, args.corpus)
            print(f"[skip] verified existing export: {json.dumps(result)}", flush=True)
            return

    # Each array task has its own corpus, so staging paths cannot collide.
    staging_root = ROOT / "output" / "diagnostics" / "frozen_embedding_staging"
    staging_root = staging_root / f"{args.corpus}_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    staged_cfg = dict(cfg, output_root=str(staging_root), skip_existing=False)
    staged = export_corpus(args.corpus, staged_cfg, force=True, anchor=ROOT)
    result = verify_export(staged, args.corpus)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)
    result["embedding_csv"] = str(final)
    write_inventory(result, args.corpus)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
