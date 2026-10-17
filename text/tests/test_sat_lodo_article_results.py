"""Checks for the article metrics that differ from the original strict scores."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sat_lodo_article_results as results


class ArticleMetricsTests(unittest.TestCase):
    def test_whitespace_gaps_collapse_to_one_internal_boundary(self):
        text = "A.  B."
        row = {"document_id": "example", "raw_text": text, "units": [
            {"start": 0, "end": 2, "role": "A0"},
            {"start": 4, "end": 6, "role": "B"},
        ]}
        counts = results.score_document(row, [(0, 4), (4, 6)])
        self.assertEqual((counts["span_tp"], counts["span_pred"], counts["span_ref"]),
                         (2, 2, 2))
        self.assertEqual((counts["boundary_tp"], counts["boundary_pred"],
                          counts["boundary_ref"]), (1, 1, 1))

    def test_fusion_and_fragmentation_are_different_errors(self):
        row = {"document_id": "example", "raw_text": "aa bb cc", "units": [
            {"start": 0, "end": 2, "role": "A0"},
            {"start": 3, "end": 5, "role": "B"},
            {"start": 6, "end": 8, "role": "C"},
        ]}
        fused = results.score_document(row, [(0, 5), (5, 8)])
        self.assertEqual(fused["fusion"], 1)
        self.assertEqual(fused["fragmented"], 0)
        fragmented = results.score_document(row, [(0, 1), (1, 3), (3, 5), (5, 8)])
        self.assertEqual(fragmented["fragmented"], 1)
        self.assertEqual(fragmented["fusion"], 0)

    def test_paired_bootstrap_zero_for_identical_predictions(self):
        rows = [{"span_tp": 2, "span_pred": 3, "span_ref": 4,
                 "boundary_tp": 3, "boundary_pred": 5, "boundary_ref": 6},
                {"span_tp": 1, "span_pred": 2, "span_ref": 3,
                 "boundary_tp": 2, "boundary_pred": 4, "boundary_ref": 5}]
        span, boundary = results.bootstrap_gain(rows, rows, 1, 100)
        self.assertTrue((span == 0).all())
        self.assertTrue((boundary == 0).all())


if __name__ == "__main__":
    unittest.main()
