"""Regression checks for factual and explicitly excluded boundary targets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))

import sat_elementary_lodo as experiment  # noqa: E402


def example() -> dict:
    raw = "A agit. ?? NB. B tombe."
    first = raw.index("A agit.")
    excluded = raw.index("NB.")
    last = raw.index("B tombe.")
    return {"raw_text": raw, "corpus": "btp", "document_id": "toy",
            "validation_method": "INDIVIDUAL_APP_VALIDATION",
            "units": [{"start": first, "end": first + len("A agit."), "role": "A0"},
                      {"start": last, "end": last + len("B tombe."), "role": "B"}],
            "excluded_spans": [{"start": excluded,
                                "end": excluded + len("NB.")}]}


class ElementaryBoundaryTests(unittest.TestCase):
    def test_training_keeps_fact_and_exclusion_in_textual_order(self) -> None:
        row = example()
        chunks, reason, omitted = experiment.elementary_chunks(row)
        self.assertEqual(chunks, ["A agit.", "NB.", "B tombe."])
        self.assertEqual(reason, "literal_elementary_units")
        self.assertEqual(omitted, len(row["raw_text"]) - sum(map(len, chunks)))
        self.assertNotIn("??", "".join(chunks))

    def test_exact_exclusion_and_factual_metrics_are_distinct(self) -> None:
        row = example()
        pred = [(item["start"], item["end"])
                for field in ("units", "excluded_spans") for item in row[field]]
        counts = experiment.score(row, pred)
        self.assertEqual((counts["fact_tp"], counts["fact_pred"], counts["fact_ref"]),
                         (2, 3, 2))
        self.assertEqual((counts["excluded_tp"], counts["excluded_ref"]), (1, 1))
        self.assertEqual(experiment.rates(counts)["elementary_exact_f1"], 1.0)
        self.assertEqual(experiment.rates(counts)["factual_exact_recall"], 1.0)
        self.assertEqual(experiment.rates(counts)["excluded_exact_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
