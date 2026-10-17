import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sat_lodo_low_thresholds as low


def test_source_only_reader_never_opens_target_test_and_deduplicates(tmp_path):
    domains = low.sat_lodo.CORPORA
    for target in domains:
        folder = tmp_path / "folds" / f"target_{target}"
        folder.mkdir(parents=True)
        rows = [{"corpus": domain, "document_id": domain,
                 "raw_text": "Un fait.", "units": [{"start": 0, "end": 8}]}
                for domain in domains if domain != target]
        (folder / "val.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (folder / "test.jsonl").write_text("INVALID TEST JSON", encoding="utf-8")
    rotations, unique = low.source_validation_rows(tmp_path)
    assert len(rotations) == len(domains)
    assert len(unique) == len(domains)
    assert all(all(row["corpus"] != target for row in rows)
               for target, rows in rotations.items())


def test_cache_refuses_different_bundle(tmp_path):
    db = low.cache_connection(tmp_path / "pred.sqlite3", "one")
    db.close()
    with pytest.raises(ValueError, match="another bundle"):
        low.cache_connection(tmp_path / "pred.sqlite3", "two")
