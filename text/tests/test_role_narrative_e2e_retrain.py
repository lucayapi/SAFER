"""Audit the end-to-end scoring and the checkpoint-pruning gate."""

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import role_narrative_e2e_baselines as scoring
import role_narrative_e2e_retrain as campaign


def test_role_aware_scoring_uses_role_blind_span_alignment():
    row = {"document_id": "example", "raw_text": "alpha beta",
           "units": [{"start": 0, "end": 5, "role": "A0"},
                     {"start": 6, "end": 10, "role": "B"}]}
    score, records = scoring.score_target(
        "btp", "M3", "ce", [row], [[(0, 5), (6, 10)]], ["B", "B"])
    assert score["span_matches"] == 2
    assert score["correct_role_matches"] == 1
    assert score["unit_f1"] == 1.0
    assert score["role_aware_f1"] == 0.5
    assert len(records) == 2


def test_incomplete_completion_marker_cannot_authorize_checkpoint_deletion(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(campaign, "manifest_path", lambda: manifest)
    dest = tmp_path / "result"
    out = dest / "e2e"
    out.mkdir(parents=True)
    marker = out / "complete.json"
    common = {"retrain_manifest_sha256": campaign.base.digest(manifest),
              "checkpoint_retention_policy": "prune_after_verified_span_inference"}
    marker.write_text(json.dumps({**common, "files_sha256": {}}), encoding="utf-8")
    assert not campaign.captured(dest)
    for name in ("scores.csv", "predictions.csv"):
        (out / name).write_text("verified", encoding="utf-8")
    hashes = {name: campaign.base.digest(out / name)
              for name in ("scores.csv", "predictions.csv")}
    marker.write_text(json.dumps({**common, "files_sha256": hashes}), encoding="utf-8")
    assert campaign.captured(dest)
    (out / "predictions.csv").write_text("changed", encoding="utf-8")
    assert not campaign.captured(dest)
