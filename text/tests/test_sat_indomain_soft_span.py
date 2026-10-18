"""Checks for the diagnostic one-to-one partial-span score."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sat_indomain_soft_span as soft


class SoftSpanTests(unittest.TestCase):
    def test_exact_and_peripheral_overlap(self):
        self.assertEqual(soft.optimal_credit([(0, 10)], [(0, 10)], 0.75), (1.0, 1))
        credit, matches = soft.optimal_credit([(1, 10)], [(0, 10)], 0.75)
        self.assertAlmostEqual(credit, 0.9)
        self.assertEqual(matches, 1)
        self.assertEqual(soft.optimal_credit([(1, 10)], [(0, 10)], 0.90), (0.9, 1))

    def test_fusion_has_at_most_one_match(self):
        self.assertEqual(soft.optimal_credit([(0, 10)], [(0, 5), (5, 10)], 0.5), (0.5, 1))
        self.assertEqual(soft.optimal_credit([(0, 10)], [(0, 5), (5, 10)], 0.75), (0.0, 0))
        self.assertEqual(soft.metric(0.5, 1, 2), (0.5, 0.25, 1 / 3))

    def test_optimal_matching_beats_first_overlap(self):
        # The first predicted span overlaps two references; reserving the
        # second reference for the second prediction gives a better score.
        pred = [(0, 8), (8, 12)]
        gold = [(0, 4), (4, 12)]
        credit, matches = soft.optimal_credit(pred, gold, 0.5)
        self.assertAlmostEqual(credit, 1.0)
        self.assertEqual(matches, 2)

    def test_normalize_trims_only_peripheral_spaces(self):
        self.assertEqual(soft.normalize("  abc def  ", [(0, 11)]), [(2, 9)])
        with self.assertRaises(ValueError):
            soft.normalize("abcd", [(0, 3), (2, 4)])


if __name__ == "__main__":
    unittest.main()
