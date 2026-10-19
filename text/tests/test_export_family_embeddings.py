import csv
import sys
from pathlib import Path

import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEXT_ROOT / "annotation"))

from export_family_embeddings import export_corpus  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _family_fixture(
    root: Path,
    *,
    include_second_embedding: bool = True,
    embedding_variant: str = "multilingual_e5_large",
) -> Path:
    family_root = root / "dataset" / "families" / "btp"
    _write_csv(
        family_root / "topic_modeling_manifest.csv",
        ["dataset_id", "n_units"],
        [{"dataset_id": "btp_family_one", "n_units": "2"}],
    )
    _write_csv(
        family_root / "data_btp_family_one.csv",
        ["accident_id", "fact_id"],
        [
            {"accident_id": "acc-1", "fact_id": "unit-1"},
            {"accident_id": "acc-2", "fact_id": "unit-2"},
        ],
    )
    embeddings = [
        {"doc_id": "unit-1", "dim_0001": "0.1", "dim_0002": "0.2"},
        {"doc_id": "unit-2", "dim_0001": "0.3", "dim_0002": "0.4"},
    ]
    if not include_second_embedding:
        embeddings.pop()
    embeddings.append({"doc_id": "unrelated", "dim_0001": "0.5", "dim_0002": "0.6"})
    source_directory = "qwen3" if embedding_variant == "qwen3" else "multilingual_e5_large"
    _write_csv(
        root / "embeddings" / source_directory / "btp.csv",
        ["doc_id", "dim_0001", "dim_0002"],
        embeddings,
    )
    return root


def test_exports_exact_family_membership_without_unrelated_rows(tmp_path: Path) -> None:
    root = _family_fixture(tmp_path)

    manifest = export_corpus("btp", text_root=root)

    assert manifest[0]["n_units"] == 2
    assert manifest[0]["n_embeddings"] == 2
    output = root / "embeddings" / "families" / "btp" / "multilingual_e5_large_btp_family_one.csv"
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["doc_id"] for row in rows] == ["unit-1", "unit-2"]
    assert (output.parent / "multilingual_e5_large_family_manifest.csv").is_file()


def test_fails_on_missing_embedding_and_leaves_no_partial_export(tmp_path: Path) -> None:
    root = _family_fixture(tmp_path, include_second_embedding=False)

    with pytest.raises(ValueError, match="alignment incomplete"):
        export_corpus("btp", text_root=root)

    output = root / "embeddings" / "families" / "btp" / "multilingual_e5_large_btp_family_one.csv"
    assert not output.exists()
    assert not list(output.parent.glob("*.tmp"))


def test_qwen_export_replaces_family_embeddings_at_manifest_paths(tmp_path: Path) -> None:
    root = _family_fixture(tmp_path, embedding_variant="qwen3")
    output = (
        root / "embeddings" / "families" / "btp"
        / "Qwen3-Embedding-0.6B_btp_family_one.csv"
    )
    _write_csv(output, ["doc_id", "dim_0001"], [{"doc_id": "stale", "dim_0001": "9"}])

    manifest = export_corpus("btp", text_root=root, embedding_variant="qwen3")

    assert manifest[0]["n_units"] == manifest[0]["n_embeddings"] == 2
    assert manifest[0]["embedding_dimensions"] == 2
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["doc_id"] for row in rows] == ["unit-1", "unit-2"]
    assert (output.parent / "Qwen3-Embedding-0.6B_family_manifest.csv").is_file()


def test_qwen_alignment_failure_preserves_existing_family_file(tmp_path: Path) -> None:
    root = _family_fixture(
        tmp_path,
        include_second_embedding=False,
        embedding_variant="qwen3",
    )
    output = (
        root / "embeddings" / "families" / "btp"
        / "Qwen3-Embedding-0.6B_btp_family_one.csv"
    )
    _write_csv(output, ["doc_id", "dim_0001"], [{"doc_id": "old", "dim_0001": "8"}])

    with pytest.raises(ValueError, match="alignment incomplete"):
        export_corpus("btp", text_root=root, embedding_variant="qwen3")

    with output.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["doc_id"] == "old"
    assert not list(output.parent.glob("*.tmp"))


def test_family_resolver_selects_e5_or_preserves_qwen_default() -> None:
    sys.path.insert(0, str(TEXT_ROOT / "recurrent_scenarios"))
    from scenario_pipeline import select_dataset_config

    base = {"data": {"dataset_id": "caou", "dataset_registry": {}}}

    qwen = select_dataset_config(base, "btp_electrical_installation")
    e5 = select_dataset_config(
        {"data": {"dataset_id": "caou", "embedding_variant": "multilingual_e5_large", "dataset_registry": {}}},
        "btp_electrical_installation",
    )

    assert qwen["data"]["embeddings_path"].endswith(
        "Qwen3-Embedding-0.6B_btp_electrical_installation.csv"
    )
    assert e5["data"]["embeddings_path"].endswith(
        "multilingual_e5_large_btp_electrical_installation.csv"
    )
    assert e5["data"]["units_path"] == qwen["data"]["units_path"]
