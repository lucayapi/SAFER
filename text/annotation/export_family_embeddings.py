"""Export corpus embeddings aligned to existing family unit files.

The family assignments are not recomputed here. Each family unit table is
matched to its source embedding table by ``fact_id`` / ``doc_id`` and the
matching rows are written to the matching family embedding directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


TEXT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPORA = ("btp", "caou", "metallurgie")
EMBEDDING_VARIANTS = {
    "multilingual_e5_large": {
        "source_dir": "multilingual_e5_large",
        "filename_prefix": "multilingual_e5_large",
        "label": "E5",
    },
    "qwen3": {
        "source_dir": "qwen3",
        "filename_prefix": "Qwen3-Embedding-0.6B",
        "label": "Qwen",
    },
}


@dataclass(frozen=True)
class FamilyInput:
    dataset_id: str
    units_path: Path
    expected_units: int


def _read_family_manifest(corpus: str, dataset_root: Path) -> list[FamilyInput]:
    manifest = dataset_root / "families" / corpus / "topic_modeling_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Family manifest not found: {manifest}")

    families: list[FamilyInput] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_id", "n_units"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected family manifest schema: {manifest}")
        for row in reader:
            dataset_id = (row.get("dataset_id") or "").strip()
            if not re.fullmatch(rf"{re.escape(corpus)}_[a-z0-9_]+", dataset_id):
                raise ValueError(f"Invalid dataset_id {dataset_id!r} in {manifest}")
            units_path = dataset_root / "families" / corpus / f"data_{dataset_id}.csv"
            if not units_path.is_file():
                raise FileNotFoundError(f"Family unit file not found: {units_path}")
            families.append(FamilyInput(dataset_id, units_path, int(row["n_units"])))

    if not families:
        raise ValueError(f"No families listed in {manifest}")
    if len({family.dataset_id for family in families}) != len(families):
        raise ValueError(f"Duplicate family IDs in {manifest}")
    return families


def _read_unit_ids(family: FamilyInput) -> set[str]:
    ids: set[str] = set()
    with family.units_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "fact_id" not in reader.fieldnames:
            raise ValueError(f"Family units need a fact_id column: {family.units_path}")
        for row_number, row in enumerate(reader, start=2):
            fact_id = (row.get("fact_id") or "").strip()
            if not fact_id:
                raise ValueError(f"Empty fact_id at {family.units_path}:{row_number}")
            if fact_id in ids:
                raise ValueError(f"Duplicate fact_id {fact_id!r} in {family.units_path}")
            ids.add(fact_id)
    if len(ids) != family.expected_units:
        raise ValueError(
            f"Manifest/unit row mismatch for {family.dataset_id}: "
            f"manifest={family.expected_units}, file={len(ids)}"
        )
    return ids


def export_corpus(
    corpus: str,
    *,
    text_root: Path = TEXT_ROOT,
    embedding_variant: str = "multilingual_e5_large",
    replace_existing: bool = True,
) -> list[dict[str, object]]:
    """Split one source embedding table into pre-existing family memberships."""
    corpus = corpus.strip().lower()
    if corpus not in DEFAULT_CORPORA:
        raise ValueError(f"Unsupported family corpus {corpus!r}; choose {DEFAULT_CORPORA}")
    if embedding_variant not in EMBEDDING_VARIANTS:
        raise ValueError(
            f"Unsupported embedding variant {embedding_variant!r}; "
            f"choose {tuple(EMBEDDING_VARIANTS)}"
        )
    text_root = Path(text_root).resolve()
    variant = EMBEDDING_VARIANTS[embedding_variant]
    filename_prefix = str(variant["filename_prefix"])

    dataset_root = text_root / "dataset"
    embedding_source = (
        text_root / "embeddings" / str(variant["source_dir"]) / f"{corpus}.csv"
    )
    if not embedding_source.is_file():
        raise FileNotFoundError(
            f"Source {variant['label']} embeddings not found: {embedding_source}"
        )

    families = _read_family_manifest(corpus, dataset_root)
    requested_ids: dict[str, str] = {}
    family_counts: dict[str, int] = {}
    for family in families:
        ids = _read_unit_ids(family)
        family_counts[family.dataset_id] = len(ids)
        for fact_id in ids:
            prior = requested_ids.get(fact_id)
            if prior is not None:
                raise ValueError(
                    f"Unit {fact_id!r} occurs in both {prior} and {family.dataset_id}"
                )
            requested_ids[fact_id] = family.dataset_id

    output_root = text_root / "embeddings" / "families" / corpus
    output_root.mkdir(parents=True, exist_ok=True)
    temp_paths: dict[str, Path] = {}
    writers: dict[str, csv.DictWriter] = {}
    handles = []
    found_ids: dict[str, set[str]] = {family.dataset_id: set() for family in families}
    seen_source_ids: set[str] = set()
    row_counts: dict[str, int] = {family.dataset_id: 0 for family in families}

    try:
        with embedding_source.open("r", encoding="utf-8-sig", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            if not reader.fieldnames or "doc_id" not in reader.fieldnames:
                raise ValueError(f"Embedding file needs a doc_id column: {embedding_source}")
            dimension_columns = [name for name in reader.fieldnames if name.startswith("dim_")]
            if not dimension_columns:
                raise ValueError(f"No dim_* embedding columns found in {embedding_source}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Duplicate columns found in {embedding_source}")

            for family in families:
                destination = output_root / f"{filename_prefix}_{family.dataset_id}.csv"
                if destination.exists() and not replace_existing:
                    raise FileExistsError(destination)
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".tmp", dir=output_root
                )
                temp_path = Path(tmp_name)
                temp_paths[family.dataset_id] = temp_path
                handle = os.fdopen(fd, "w", encoding="utf-8", newline="")
                handles.append(handle)
                writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, extrasaction="raise")
                writer.writeheader()
                writers[family.dataset_id] = writer

            for row_number, row in enumerate(reader, start=2):
                doc_id = (row.get("doc_id") or "").strip()
                if not doc_id:
                    raise ValueError(f"Empty doc_id at {embedding_source}:{row_number}")
                if doc_id in seen_source_ids:
                    raise ValueError(f"Duplicate doc_id {doc_id!r} in {embedding_source}")
                seen_source_ids.add(doc_id)
                dataset_id = requested_ids.get(doc_id)
                if dataset_id is not None:
                    writers[dataset_id].writerow(row)
                    found_ids[dataset_id].add(doc_id)
                    row_counts[dataset_id] += 1

        missing = {
            dataset_id: family_counts[dataset_id] - len(found_ids[dataset_id])
            for dataset_id in family_counts
            if family_counts[dataset_id] != len(found_ids[dataset_id])
        }
        if missing:
            raise ValueError(
                f"{variant['label']} alignment incomplete for {corpus}; "
                f"missing embeddings by family: {missing}"
            )

        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        manifest_rows: list[dict[str, object]] = []
        for family in families:
            destination = output_root / f"{filename_prefix}_{family.dataset_id}.csv"
            os.replace(temp_paths[family.dataset_id], destination)
            manifest_rows.append({
                "dataset_id": family.dataset_id,
                "n_units": family_counts[family.dataset_id],
                "n_embeddings": row_counts[family.dataset_id],
                "embedding_dimensions": len(dimension_columns),
                "units_path": family.units_path.relative_to(text_root).as_posix(),
                "embeddings_path": destination.relative_to(text_root).as_posix(),
            })
        manifest_path = output_root / f"{filename_prefix}_family_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle:
            writer = csv.DictWriter(manifest_handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        return manifest_rows
    finally:
        for handle in handles:
            handle.close()
        for temp_path in temp_paths.values():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora", nargs="+", choices=DEFAULT_CORPORA, default=DEFAULT_CORPORA,
        help="Corpora with existing family memberships (default: all three).",
    )
    parser.add_argument(
        "--variant", choices=tuple(EMBEDDING_VARIANTS), default="multilingual_e5_large",
        help="Embedding source and output naming convention (default: multilingual E5).",
    )
    parser.add_argument("--text-root", type=Path, default=TEXT_ROOT)
    args = parser.parse_args()

    for corpus in args.corpora:
        rows = export_corpus(
            corpus,
            text_root=args.text_root.resolve(),
            embedding_variant=args.variant,
        )
        total_units = sum(int(row["n_units"]) for row in rows)
        dimensions = sorted({int(row["embedding_dimensions"]) for row in rows})
        print(
            f"{corpus} ({args.variant}): exported {len(rows)} families, "
            f"{total_units:,} aligned units, dimensions={dimensions}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
