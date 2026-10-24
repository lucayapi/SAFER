"""Checks for the constrained source sampling used by RQ1 equal-volume fits."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rq1_seg_equal_volume as equal  # noqa: E402


class EqualVolumeSamplingTest(unittest.TestCase):
    def test_reproducible_character_budget_and_five_sources(self):
        target, source = "btp", "nicollin"
        corpora = [c for c in equal.sat.CORPORA if c != target]
        eligible = {
            corpus: [({"corpus": corpus, "document_id": f"{corpus}-{index}",
                       "raw_text": "x" * (75 + (index * 17 + offset) % 65)},
                      ["x" * (75 + (index * 17 + offset) % 65)])
                     for index in range(60)]
            for offset, corpus in enumerate(corpora)
        }
        one = equal.balanced_sample(eligible, target, source, 7000)
        two = equal.balanced_sample(eligible, target, source, 7000)
        self.assertEqual([equal.key(row) for row, _ in one],
                         [equal.key(row) for row, _ in two])
        self.assertEqual({row["corpus"] for row, _ in one}, set(corpora))
        self.assertEqual(len({equal.key(row) for row, _ in one}), len(one))
        self.assertLessEqual(7000 - sum(equal.input_chars(item) for item in one), 70)

    def test_rejects_insufficient_source_volume(self):
        target, source = "btp", "nicollin"
        eligible = {corpus: [({"corpus": corpus, "document_id": corpus,
                              "raw_text": "x" * 10}, ["x"])]
                    for corpus in equal.sat.CORPORA if corpus != target}
        with self.assertRaises(ValueError):
            equal.balanced_sample(eligible, target, source, 1000)

    def test_exact_f1_and_transfer_figure(self):
        self.assertAlmostEqual(equal.f1([3, 5, 6]), 6 / 11)
        matrix = [{"source": corpus, **{target: ("N/A" if corpus == target else .05)
                                          for target in equal.sat.CORPORA}}
                  for corpus in equal.sat.CORPORA]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transfer.svg"
            equal.transfer_svg(path, matrix)
            self.assertIn("Multisource minus monosource", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
