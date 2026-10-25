from __future__ import annotations

import sys
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rq1_e2_excluded as method  # noqa: E402


def example_row():
    return {"corpus": "btp", "document_id": "example", "raw_text": "AB??CD??EF",
            "units": [{"start": 0, "end": 2, "role": "OTHER"},
                      {"start": 8, "end": 10, "role": "B"}],
            "excluded_spans": [{"start": 4, "end": 6}]}


def test_reference_edges_and_unknown_gaps_are_distinct():
    row = example_row()
    labels = method.boundary_labels(row, list(range(1, 10)))
    assert np.flatnonzero(labels == 1).tolist() == [1, 3, 5, 7]
    assert np.flatnonzero(labels == 0).tolist() == [0, 4, 8]
    assert np.flatnonzero(labels == -1).tolist() == [2, 6]


def test_exclusion_is_counted_only_when_its_interval_is_recovered():
    row = example_row()
    partial = method.exclusion_counts(row, [(3, 7)], [3, 5])
    assert partial == {"excluded_ref": 1, "excluded_exact": 0,
                       "excluded_iou50": 1, "excluded_endpoint_available": 0,
                       "fact_endpoint_available": 0}
    exact = method.exclusion_counts(row, [(4, 6)], [4, 6])
    assert exact["excluded_exact"] == 1
    assert exact["excluded_endpoint_available"] == 1


def test_overlapping_fact_and_exclusion_is_rejected():
    row = example_row()
    row["excluded_spans"][0]["start"] = 1
    with pytest.raises(ValueError, match="Overlapping"):
        method.boundary_labels(row, [1, 2, 4, 6, 8])


def test_inference_features_do_not_require_reference_labels():
    row = example_row()
    raw = row["raw_text"]
    feature = (np.full(len(raw), .05, dtype=np.float32),
               np.zeros((len(raw), 128), dtype=np.float16),
               np.column_stack((np.arange(len(raw)), np.arange(1, len(raw) + 1))))
    supervised = method.boundary_example(row, feature, .02)
    without_labels = {key: value for key, value in row.items()
                      if key not in ("units", "excluded_spans")}
    inference = method.inference_example(without_labels, feature, .02)
    assert inference.ranks == supervised.ranks
    np.testing.assert_array_equal(inference.features, supervised.features)
    assert np.all(inference.labels == -1)


def test_six_domain_summary_reports_only_exclusion_aware_refiner(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    source = tmp_path / "source"
    output = tmp_path / "output"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    rows_by_target = {}
    spans = [(0, 2), (4, 6), (8, 10)]
    for target in method.sat.CORPORA:
        row = example_row()
        row["corpus"] = target
        row["document_id"] = target + "_example"
        rows_by_target[target] = row
        target_source = source / f"target_{target}"
        target_source.mkdir(parents=True)
        (target_source / "m3_final.json").write_text("{}", encoding="utf-8")
        folder = output / f"target_{target}"
        method.upstream.save_crossfit_prediction(
            folder / "predictions" / "contextual_refinement.jsonl",
            [row], [spans], method.METHOD, .02, method.SEED)
        score = method.sat.score(row, spans)
        exclusions = method.exclusion_counts(row, spans)
        stored = {"factual_exact_f1": 2 * score["span_tp"] /
                  (score["span_pred"] + score["span_ref"]),
                  "excluded_ref": 1,
                  "excluded_exact_recall": exclusions["excluded_exact"],
                  "excluded_iou50_recall": exclusions["excluded_iou50"]}
        method.upstream.json_write(folder / "metrics.json", {
            "n_test": 1, "sat_training_seed": 500, "method": method.METHOD,
            "scores": stored,
            "candidate_coverage": {"excluded_both_endpoints": 1,
                                   "excluded_ref": 1,
                                   "factual_both_endpoints": 2,
                                   "factual_ref": 2}})

    monkeypatch.setattr(method, "check_binding", lambda *_: None)
    monkeypatch.setattr(method, "completed", lambda *_: True)
    monkeypatch.setattr(method.upstream, "bundle_groups",
                        lambda _, target: {"test": [rows_by_target[target]]})
    method.summary(SimpleNamespace(bundle=bundle, source=source, out=output))
    with (output / "summary" / "per_domain.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with (output / "summary" / "macro.csv").open(encoding="utf-8", newline="") as f:
        macro = list(csv.DictReader(f))
    assert len(rows) == 6
    assert all(row["method"] == method.METHOD for row in rows)
    assert len(macro) == 1
    assert float(macro[0]["macro_excluded_exact_recall"]) == 1.0
    assert not (output / "summary" / "paired_bootstrap.csv").exists()
