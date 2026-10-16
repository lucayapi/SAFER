"""Critical invariants of the independent narrative preannotation pipeline."""

from argparse import Namespace
import csv
import json

from scripts import annotate_narratives_batch as batch


def test_prepared_request_is_independent_of_sat(tmp_path, monkeypatch):
    source = tmp_path / "pilot.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "accident_id", "fact_id", "sentence", "accident_summary"])
        writer.writeheader()
        writer.writerow({"accident_id": "a1", "fact_id": "1", "sentence": "Il travaille.",
                         "accident_summary": "Il travaille. Il tombe."})
        writer.writerow({"accident_id": "a1", "fact_id": "2", "sentence": "Il tombe.",
                         "accident_summary": "Il travaille. Il tombe."})
    monkeypatch.setattr(batch, "OUTPUT_ROOT", tmp_path / "output")
    batch.prepare(Namespace(corpus="test", input_csv=str(source), run_id="pilot",
                            all=True, limit=20, seed=42, model="gpt-5.4-mini",
                            reasoning_effort="low", max_completion_tokens=12000))
    run_dir = tmp_path / "output" / "test" / "pilot"
    doc = batch._read_jsonl(run_dir / "documents.jsonl")[0]
    request = batch._read_jsonl(run_dir / "batch_input.jsonl")[0]
    assert len(doc["sat_segments"]) == 2
    assert doc["sat_segments"][1]["start"] == 14
    messages = request["body"]["messages"]
    assert doc["raw_text"] in messages[1]["content"]
    assert "sat_segments" not in json.dumps(messages)
    assert "sentence" not in json.dumps(messages)


def test_invalid_offsets_and_overlapping_exclusions_are_rejected():
    raw = "Il travaille. Il tombe."
    annotation = {
        "units": [{"start": 0, "end": 12, "text": "Il travaille.",
                   "role": "A0", "assertion": "AFFIRME"}],
        "excluded_spans": [{"start": 10, "end": 14, "reason": "test"}],
    }
    assert any("chevauchement" in error for error in
               batch.validate_annotation(raw, annotation))
    annotation["excluded_spans"] = []
    annotation["units"][0]["start"] = 1
    assert any("texte différent" in error for error in
               batch.validate_annotation(raw, annotation))
