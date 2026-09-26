"""Organize existing SoftTriple embeddings by corpus and company family.

The script never re-encodes texts. It subsets either available SoftTriple
configuration (`full_yes`, the 128-D projected model, or `full_no`, the
1024-D unprojected model) using stable `(accident_id, fact_id/doc_id)` pairs.
"""

from __future__ import annotations

import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_ROOT = Path(__file__).resolve().parents[1]
FAMILY_ROOT = TEXT_ROOT / "dataset" / "families"
CORPORA = ("btp", "caou", "metallurgie", "nicollin")


def source_root(combo: str) -> Path:
    return TEXT_ROOT / "output" / "softtriple" / "macro_ft_tuning" / "combos" / combo / "embeddings"


def output_root(combo: str) -> Path:
    return TEXT_ROOT / "embeddings" / "softtriple" / combo


def link_or_copy(source: Path, target: Path) -> None:
    """Expose an unchanged full-corpus artifact without duplicating its bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.samefile(source):
            return
        if target.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"Existing target differs from source: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_metadata(corpus: str, source: Path) -> pd.DataFrame:
    path = source / f"projected_{corpus}_metadata.csv"
    metadata = pd.read_csv(path, dtype={"accident_id": "string", "doc_id": "string"})
    required = {"accident_id", "doc_id", "sentence", "pred_label"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{corpus}: missing metadata columns {sorted(missing)}")
    metadata["accident_id"] = metadata["accident_id"].astype("string").str.strip()
    metadata["doc_id"] = metadata["doc_id"].astype("string").str.strip()
    if metadata.duplicated(["accident_id", "doc_id"]).any():
        raise ValueError(f"{corpus}: duplicate (accident_id, doc_id) in SoftTriple metadata")
    metadata["_source_row"] = np.arange(len(metadata), dtype=np.int64)
    return metadata


def write_pipeline_csv(
    output_path: Path,
    embeddings: np.ndarray,
    doc_ids: pd.Series,
    row_indices: np.ndarray | None = None,
) -> None:
    """Write the exact `doc_id, dim_0001, ...` format used by topic pipelines."""
    if len(doc_ids) != (len(embeddings) if row_indices is None else len(row_indices)):
        raise ValueError(f"CSV export mismatch for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = [f"dim_{index:04d}" for index in range(1, embeddings.shape[1] + 1)]
    chunk_size = 1024
    for start in range(0, len(doc_ids), chunk_size):
        stop = min(start + chunk_size, len(doc_ids))
        indices = slice(start, stop) if row_indices is None else row_indices[start:stop]
        table = pd.DataFrame(np.asarray(embeddings[indices]), columns=dimensions)
        table.insert(0, "doc_id", doc_ids.iloc[start:stop].astype("string").to_numpy())
        table.to_csv(output_path, index=False, mode="w" if start == 0 else "a", header=start == 0)


def export_family(
    corpus: str,
    dataset_id: str,
    label: str,
    source_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    destination: Path,
) -> dict[str, object]:
    units_path = FAMILY_ROOT / corpus / f"data_{dataset_id}.csv"
    units = pd.read_csv(units_path, dtype={"accident_id": "string", "fact_id": "string"})
    required = {"accident_id", "fact_id"}
    missing = required.difference(units.columns)
    if missing:
        raise ValueError(f"{dataset_id}: missing unit columns {sorted(missing)}")
    keys = units[["accident_id", "fact_id"]].copy()
    keys["accident_id"] = keys["accident_id"].astype("string").str.strip()
    keys["fact_id"] = keys["fact_id"].astype("string").str.strip()
    if keys.duplicated(["accident_id", "fact_id"]).any():
        raise ValueError(f"{dataset_id}: duplicate (accident_id, fact_id) in family units")

    matched = keys.merge(
        metadata,
        left_on=["accident_id", "fact_id"],
        right_on=["accident_id", "doc_id"],
        how="left",
        validate="one_to_one",
    )
    if matched["_source_row"].isna().any():
        raise ValueError(f"{dataset_id}: {int(matched['_source_row'].isna().sum())} units have no SoftTriple embedding")

    selected = matched.sort_values("_source_row", kind="stable").copy()
    row_indices = selected["_source_row"].astype(np.int64).to_numpy()
    output_dir = destination / corpus / "families" / dataset_id.removeprefix(f"{corpus}_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_embeddings = output_dir / "embeddings.npy"
    output_csv = output_dir / "embeddings.csv"
    output_metadata = output_dir / "metadata.csv"

    out = np.lib.format.open_memmap(
        output_embeddings,
        mode="w+",
        dtype=source_embeddings.dtype,
        shape=(len(row_indices), source_embeddings.shape[1]),
    )
    chunk_size = 2048
    for start in range(0, len(row_indices), chunk_size):
        stop = min(start + chunk_size, len(row_indices))
        out[start:stop] = source_embeddings[row_indices[start:stop]]
    del out

    selected["fact_id"] = selected["doc_id"]
    selected[["accident_id", "doc_id", "fact_id", "sentence", "pred_label"]].to_csv(output_metadata, index=False)
    write_pipeline_csv(output_csv, source_embeddings, selected["doc_id"].reset_index(drop=True), row_indices)
    return {
        "dataset_id": dataset_id,
        "macro_activity": label,
        "n_units": len(selected),
        "embedding_dim": int(source_embeddings.shape[1]),
        "embeddings_path": str(output_embeddings.relative_to(TEXT_ROOT)),
        "embeddings_csv_path": str(output_csv.relative_to(TEXT_ROOT)),
        "metadata_path": str(output_metadata.relative_to(TEXT_ROOT)),
    }


def export_corpus(corpus: str, source: Path, destination: Path) -> None:
    source_embeddings_path = source / f"projected_{corpus}.npy"
    metadata_path = source / f"projected_{corpus}_metadata.csv"
    if not source_embeddings_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing SoftTriple export for {corpus}")
    source_embeddings = np.load(source_embeddings_path, mmap_mode="r")
    metadata = read_metadata(corpus, source)
    if source_embeddings.ndim != 2 or source_embeddings.shape[0] != len(metadata):
        raise ValueError(f"{corpus}: array / metadata row mismatch")

    corpus_dir = destination / corpus
    link_or_copy(source_embeddings_path, corpus_dir / "all" / "embeddings.npy")
    link_or_copy(metadata_path, corpus_dir / "all" / "metadata.csv")
    all_csv_path = corpus_dir / "all" / "embeddings.csv"
    write_pipeline_csv(all_csv_path, source_embeddings, metadata["doc_id"])
    rows: list[dict[str, object]] = [{
        "dataset_id": corpus,
        "macro_activity": "All corpus units",
        "n_units": len(metadata),
        "embedding_dim": int(source_embeddings.shape[1]),
        "embeddings_path": str((corpus_dir / "all" / "embeddings.npy").relative_to(TEXT_ROOT)),
        "embeddings_csv_path": str(all_csv_path.relative_to(TEXT_ROOT)),
        "metadata_path": str((corpus_dir / "all" / "metadata.csv").relative_to(TEXT_ROOT)),
    }]

    manifest_path = FAMILY_ROOT / corpus / "topic_modeling_manifest.csv"
    if manifest_path.is_file():
        family_manifest = pd.read_csv(manifest_path)
        for record in family_manifest.to_dict("records"):
            rows.append(
                export_family(
                    corpus,
                    str(record["dataset_id"]),
                    str(record["macro_activity"]),
                    source_embeddings,
                    metadata,
                    destination,
                )
            )

    pd.DataFrame(rows).to_csv(corpus_dir / "manifest.csv", index=False)
    print(f"{corpus}: {len(rows) - 1} family exports + all-corpus export")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combo",
        choices=("full_yes", "full_no"),
        default="full_yes",
        help="SoftTriple configuration to export (default: full_yes).",
    )
    args = parser.parse_args()
    combo = args.combo
    source = source_root(combo)
    destination = output_root(combo)
    destination.mkdir(parents=True, exist_ok=True)
    for corpus in CORPORA:
        export_corpus(corpus, source, destination)
    model_description = (
        "the complete fine-tuned Qwen encoder and its 128-dimensional MLP projector"
        if combo == "full_yes"
        else "the complete fine-tuned Qwen encoder without a projector"
    )
    (destination / "README.md").write_text(
        f"# SoftTriple embeddings (`{combo}`)\n\n"
        f"These arrays use {model_description}. "
        "Each `metadata.csv` is row-aligned with the colocated `embeddings.npy` and `embeddings.csv`. "
        f"The CSV uses `doc_id, dim_0001, ..., dim_{'0128' if combo == 'full_yes' else '1024'}` and can be passed directly to topic pipelines. "
        "Full-corpus files are hard-linked to the original trained exports when supported.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
