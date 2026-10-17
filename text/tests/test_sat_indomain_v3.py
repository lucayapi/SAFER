"""Checks for the fixed single-source splits and selection protocol."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))

import sat_indomain_v3 as experiment  # noqa: E402


class SingleSourceTests(unittest.TestCase):
    def test_grouped_split_is_order_independent_and_keeps_duplicates_together(self) -> None:
        rows = [{"corpus": "btp", "document_id": str(i), "raw_text": f"narrative-{i}"}
                for i in range(100)]
        rows.extend([{"corpus": "btp", "document_id": f"dup-{i}",
                      "raw_text": "same narrative"} for i in range(3)])
        first = experiment.grouped_split(rows, "btp")
        second = experiment.grouped_split(list(reversed(rows)), "btp")
        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first.values())), len(rows))
        self.assertEqual(sum(any(row["raw_text"] == "same narrative" for row in part)
                             for part in first.values()), 1)
        self.assertTrue(all(first.values()))

    def test_checkpoint_and_threshold_ties_are_deterministic(self) -> None:
        curve = [{"exact_f1": 0.4, "threshold": 0.1, "step": 100},
                 {"exact_f1": 0.5, "threshold": 0.2, "step": 200},
                 {"exact_f1": 0.5, "threshold": 0.3, "step": 100},
                 {"exact_f1": 0.5, "threshold": 0.2, "step": 100}]
        selected = experiment.best_candidate(curve)
        self.assertEqual((selected["threshold"], selected["step"]), (0.2, 100))

    def test_real_bundle_has_fixed_disjoint_tests_and_exact_gold(self) -> None:
        bundle = experiment.BUNDLE
        if not bundle.is_dir():
            self.skipTest("Private GOLD bundle has not been prepared locally")
        manifest = experiment.check_bundle(bundle)
        self.assertEqual(sum(item["total"] for item in manifest["report"].values()), 17927)
        for corpus in experiment.base.CORPORA:
            split = manifest["report"][corpus]["split"]
            self.assertEqual(sum(split.values()), manifest["report"][corpus]["total"])
            self.assertLessEqual(abs(split["train"] / sum(split.values()) - 0.70), 0.01)
            self.assertLessEqual(abs(split["test"] / sum(split.values()) - 0.15), 0.01)


if __name__ == "__main__":
    unittest.main()
