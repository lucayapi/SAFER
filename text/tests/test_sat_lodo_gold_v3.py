"""Checks for the narrative_v3 RQ1 LoRA comparison and its immutable bundle."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
import unittest
from collections import Counter
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))
sys.path.insert(0, str(ROOT / "annotation"))

import rq1_calibrate_sat_v3 as rq1  # noqa: E402
import sat_lodo_gold_v3 as experiment  # noqa: E402


class GoldV3ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = experiment.BUNDLE
        cls.manifest = experiment.check_bundle(cls.bundle)

    def test_bundle_uses_rq1_partition_with_no_target_leakage(self) -> None:
        self.assertEqual(self.manifest["reference"], experiment.REFERENCE)
        self.assertEqual(sum(item["gold_documents"] for item in
                             self.manifest["source_report"].values()), 17927)
        target = "btp"
        folder = self.bundle / "folds" / f"target_{target}"
        train = experiment.read_jsonl(folder / "train.jsonl")
        val = experiment.read_jsonl(folder / "val.jsonl")
        test = experiment.read_jsonl(folder / "test.jsonl")
        self.assertEqual({row["corpus"] for row in test}, {target})
        self.assertNotIn(target, {row["corpus"] for row in train + val})
        self.assertFalse({(row["corpus"], row["document_id"]) for row in train} &
                         {(row["corpus"], row["document_id"]) for row in val})

    def test_scores_use_the_identical_rq1_whitespace_convention(self) -> None:
        row = {"raw_text": "Alpha.  Beta.",
               "units": [{"start": 0, "end": 6, "role": "A0"},
                         {"start": 8, "end": 13, "role": "C"}]}
        predicted = [(0, 8), (8, 13)]
        self.assertEqual(experiment.score(row, predicted), rq1.basic_score(row, predicted))
        self.assertEqual(experiment.score(row, predicted)["span_tp"], 2)
        self.assertEqual(experiment.score(row, [(0, 13)], details=True)["fusion"], 1)

    def test_training_never_promotes_excluded_text_into_facts(self) -> None:
        row = {"raw_text": "A fait. Commentaire. B blessé.",
               "units": [{"start": 0, "end": 7}, {"start": 21, "end": 30}]}
        chunks, reason, omitted = experiment.trainer_chunks(row)
        self.assertEqual(chunks, ["A fait.", "B blessé."])
        self.assertEqual(reason, "literal_facts_only")
        self.assertEqual(omitted, len(row["raw_text"]) - len("A fait.") - len("B blessé."))
        row["units"] = [{"start": 0, "end": 8}]
        row["raw_text"] = "A fait.\n"
        self.assertEqual(experiment.trainer_chunks(row)[1], "newline_inside_unit")

    def test_m1_and_m2_select_identically_when_given_the_same_probabilities(self) -> None:
        folder = self.bundle / "folds" / "target_btp"
        val = experiment.read_jsonl(folder / "val.jsonl")
        sample = [next(row for row in val if row["corpus"] == source and row["units"])
                  for source in experiment.CORPORA if source != "btp"]
        with closing(sqlite3.connect(self.bundle / "base_probabilities.sqlite3")) as db:
            selected, curves = experiment.choose_thresholds(sample, db, db)
        self.assertEqual(selected["M1"], selected["M2"])
        by_model = {model: [row["source_macro_span_f1"] for row in curves
                            if row["model"] == model] for model in ("M1", "M2")}
        self.assertEqual(by_model["M1"], by_model["M2"])

    def test_m0_reproduces_previous_rq1_on_all_target_documents(self) -> None:
        with (self.bundle / "rq1_baseline_metrics.csv").open(
                encoding="utf-8-sig", newline="") as stream:
            old = {row["target"]: row for row in csv.DictReader(stream)}
        with closing(sqlite3.connect(self.bundle / "base_probabilities.sqlite3")) as db:
            for target in experiment.CORPORA:
                rows = experiment.read_jsonl(
                    self.bundle / "folds" / f"target_{target}" / "test.jsonl")
                total = Counter()
                for row in rows:
                    pred = experiment.legacy.spans_from_probabilities(
                        row["raw_text"], experiment.get_probs(db, row), experiment.DEFAULT)
                    total.update(experiment.score(row, pred))
                span = experiment.prf(total["span_tp"], total["span_pred"],
                                      total["span_ref"])[2]
                boundary = experiment.prf(total["boundary_tp"], total["boundary_pred"],
                                          total["boundary_ref"])[2]
                self.assertTrue(math.isclose(span, float(old[target]["M0_span_f1"]),
                                             abs_tol=1e-12), target)
                self.assertTrue(math.isclose(boundary, float(old[target]["M0_boundary_f1"]),
                                             abs_tol=1e-12), target)


if __name__ == "__main__":
    unittest.main()
