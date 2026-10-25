from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rq1_e2_excluded_holdout as holdout  # noqa: E402


def test_one_source_holdout_keeps_target_out_and_facts_exclusions_separate(monkeypatch):
    target = holdout.sat.CORPORA[0]
    rows = []
    for corpus in holdout.sat.CORPORA[1:]:
        for index in range(12):
            rows.append({"corpus": corpus, "document_id": f"{corpus}_{index}",
                         "raw_text": f"Unique text {corpus} {index}.",
                         "units": [{"start": 0, "end": 6, "role": "OTHER"}],
                         "excluded_spans": [{"start": 7, "end": 11}]})
    monkeypatch.setattr(holdout.x, "bundle_groups",
                        lambda _bundle, _target: {"train": rows, "val": [], "test": []})
    groups, sat_rows, e2_rows = holdout.split(Path("unused"), target)
    assert len(sat_rows) + len(e2_rows) == len(groups["train"])
    assert {holdout.x.key(row) for row in sat_rows}.isdisjoint(
        {holdout.x.key(row) for row in e2_rows})
    assert {row["corpus"] for row in sat_rows} == set(holdout.sat.CORPORA) - {target}
    assert {row["corpus"] for row in e2_rows} == set(holdout.sat.CORPORA) - {target}
    assert all(row["corpus"] != target for row in sat_rows + e2_rows)


def test_summary_contains_six_targets_and_one_method(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "run"
    root.mkdir()
    (root / "binding.json").write_text("{}", encoding="utf-8")
    records = {}
    rows = {}
    for target in holdout.sat.CORPORA:
        records[target] = {"target_test_sha256": target}
        row = {"corpus": target, "document_id": target + "_1",
               "raw_text": "A fact and context", "units": [], "excluded_spans": []}
        rows[target] = row
        folder = root / f"target_{target}"
        holdout.x.save_crossfit_prediction(
            folder / "predictions" / "contextual_refinement.jsonl",
            [row], [[(0, 6)]], holdout.e2.METHOD, .02, holdout.SEED)
        holdout.x.json_write(folder / "metrics.json", {
            "target": target, "method": holdout.e2.METHOD, "refiner_seed": holdout.SEED,
            "sat_auxiliary_seed": 113, "sat_final_seed": 513, "n_test": 1,
            "factual_exact_f1": .5, "excluded_ref": 2,
            "excluded_exact_recall": .5, "excluded_iou50_recall": .75,
            "candidate_excluded_endpoint_coverage": .8,
            "candidate_factual_endpoint_coverage": .9})
    monkeypatch.setattr(holdout, "verify", lambda *_: records)
    monkeypatch.setattr(holdout, "complete", lambda *_: True)
    monkeypatch.setattr(holdout.x, "bundle_groups",
                        lambda _bundle, target: {"test": [rows[target]]})
    holdout.summary(SimpleNamespace(bundle=bundle, out=root))
    with (root / "summary" / "per_domain.csv").open(encoding="utf-8", newline="") as stream:
        per_domain = list(csv.DictReader(stream))
    with (root / "summary" / "macro.csv").open(encoding="utf-8", newline="") as stream:
        macro = list(csv.DictReader(stream))
    assert len(per_domain) == 6
    assert {row["method"] for row in per_domain} == {holdout.e2.METHOD}
    assert len(macro) == 1
    assert float(macro[0]["macro_excluded_exact_recall"]) == .5
