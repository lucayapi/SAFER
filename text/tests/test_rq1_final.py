"""Checks for the frozen RQ1 comparison, independent of GPU availability."""

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rq1_final as campaign
import rq1_final_report as report
import sat_lodo_gold_v3 as sat


class SpanScoringTests(unittest.TestCase):
    def test_soft_exact_and_partial(self):
        gold = [(0, 10)]
        self.assertEqual(report.soft_credit(gold, gold), 1.0)
        self.assertEqual(report.soft_credit([(1, 10)], gold), 0.9)
        self.assertEqual(report.soft_credit([(1, 10)], gold, 0.95), 0.0)

    def test_merged_prediction_gets_only_one_match(self):
        gold = [(0, 5), (5, 10)]
        self.assertEqual(report.soft_credit([(0, 10)], gold), 0.5)
        self.assertEqual(report.soft_credit([(0, 10)], gold, 0.75), 0.0)

    def test_trimming_and_diagnostic_roles(self):
        row = {"raw_text": "  alpha beta  ", "units": [
            {"start": 2, "end": 7, "role": "A0"},
            {"start": 8, "end": 12, "role": "B"}]}
        counts = report.score(row, [(0, 14)])
        self.assertEqual(counts["merge_cross_role"], 1)
        self.assertEqual(counts["span_tp"], 0)
        self.assertEqual(counts["span_ref"], 2)

    def test_invalid_offsets_are_rejected(self):
        row = {"corpus": "btp", "document_id": "x", "raw_text": "abc"}
        with self.assertRaises(ValueError):
            report.checked_spans(row, [{"start": 0, "end": 2, "text": "xx"}])

    def test_paired_bootstrap_covers_all_adjacent_models(self):
        archive = {}
        for target in sat.CORPORA:
            for model, tp in (("M0", 0), ("M1", 1), ("M2", 1), ("M3", 0)):
                archive[target, model] = [
                    {"document_id": str(i), "span_tp": tp, "span_pred": 1, "span_ref": 1}
                    for i in range(2)]
        estimates = report.bootstrap(archive, reps=20)
        self.assertEqual(len(estimates), 21)
        macro = {row["comparison"]: row["delta_exact_f1"] for row in estimates
                 if row["target"] == "MACRO_6"}
        self.assertEqual(macro, {"M1_minus_M0": 1.0, "M2_minus_M1": 0.0,
                                 "M3_minus_M2": -1.0})
        archive[sat.CORPORA[0], "M3"].reverse()
        with self.assertRaisesRegex(ValueError, "not paired"):
            report.bootstrap(archive, reps=20)


@unittest.skipUnless(campaign.SOURCE.is_dir() and campaign.OUTPUT.is_dir(),
                     "Private, fixed narrative_v3 bundles are unavailable")
class ExistingBundleTests(unittest.TestCase):
    def test_common_test_ids_and_no_target_leakage(self):
        manifest = campaign.check_bundle(campaign.OUTPUT)
        self.assertEqual(manifest["gold_snapshot_sha256"], campaign.EXPECTED_GOLD)
        self.assertEqual(sum(x[1] for x in campaign.EXPECTED_COUNTS.values()), 2690)

    def test_selected_single_source_cache_reproduces_old_score(self):
        source, target = "btp", "metallurgie"
        run = campaign.SINGLE_RUNS / f"source_{source}"
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        old = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        rows = sat.read_jsonl(campaign.OUTPUT / "folds" / f"target_{target}" / "test.jsonl")
        with sqlite3.connect(run / "heldout_probabilities.sqlite3") as db:
            predicted = report.probability_predictions(db, rows, selection["M2_threshold"])
        scored = report.metric_rows(rows, predicted, "single_source", target, "M3", source)
        actual = report.aggregate(scored, experiment="single_source", source=source,
                                  target=target, model="M3")
        self.assertAlmostEqual(actual["exact_f1"], old["test_sets"][target]["M2"]["span_f1"], places=12)


if __name__ == "__main__":
    unittest.main()
