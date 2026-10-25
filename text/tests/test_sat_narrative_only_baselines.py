import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sat_narrative_only_baselines as baselines


def test_prediction_reader_requires_exact_test_id_coverage(tmp_path):
    rows = [{"corpus": "btp", "document_id": "1", "raw_text": "Fait."}]
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"corpus": "btp", "document_id": "1",
                                "segments": [{"start": 0, "end": 5}]}) + "\n",
                    encoding="utf-8")
    assert baselines.read_prediction_file(path, rows, "M0") == {("btp", "1"): [(0, 5)]}

    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover"):
        baselines.read_prediction_file(path, rows, "M0")


def test_prediction_reader_rejects_offsets_outside_literal_text(tmp_path):
    rows = [{"corpus": "btp", "document_id": "1", "raw_text": "Fait."}]
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"corpus": "btp", "document_id": "1",
                                "segments": [{"start": 0, "end": 6}]}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid M0 span offsets"):
        baselines.read_prediction_file(path, rows, "M0")


def test_edge_normalized_counts_follow_m3_scoring():
    text = "«  Fait, rapporté. »"
    row = {"document_id": "id", "raw_text": text,
           "units": [{"start": 3, "end": len(text) - 2, "role": "B"}]}
    assert baselines.counts_for([row], {"id": [(0, len(text))]})[baselines.METRIC] == 1.0

