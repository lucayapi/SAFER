"""Source-only checkpoint-selection invariants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text" / "scripts"))

import sat_elementary_checkpoint_lodo as experiment  # noqa: E402


class CheckpointSelectionTests(unittest.TestCase):
    def test_best_source_f1_precedes_threshold_and_step(self) -> None:
        rows = [
            {"candidate": "early", "step": 1000, "threshold": 0.25,
             "macro_elementary_exact_f1": 0.40},
            {"candidate": "late", "step": 3000, "threshold": 0.05,
             "macro_elementary_exact_f1": 0.41},
        ]
        self.assertEqual(experiment.best_candidate(rows)["candidate"], "late")

    def test_ties_prefer_default_threshold_then_earlier_checkpoint(self) -> None:
        rows = [
            {"candidate": "late", "step": 3000, "threshold": 0.25,
             "macro_elementary_exact_f1": 0.40},
            {"candidate": "early", "step": 1000, "threshold": 0.25,
             "macro_elementary_exact_f1": 0.40},
            {"candidate": "remote", "step": 500, "threshold": 0.05,
             "macro_elementary_exact_f1": 0.40},
        ]
        self.assertEqual(experiment.best_candidate(rows)["candidate"], "early")

    def test_curve_uses_elementary_source_metric(self) -> None:
        candidate = {"candidate": "checkpoint-1000", "step": 1000, "epoch": 12.0}
        source_curve = [{"threshold": 0.25, "macro_elementary_exact_f1": 0.52,
                         "macro_factual_exact_recall": 0.44,
                         "macro_excluded_exact_recall": 0.36}]
        with patch.object(experiment.elementary, "select_threshold",
                          return_value=(0.25, source_curve)) as select:
            result = experiment.validation_curve([{"corpus": "source"}], object(), candidate)
        select.assert_called_once()
        self.assertEqual(result[0]["macro_elementary_exact_f1"], 0.52)
        self.assertEqual(result[0]["candidate"], "checkpoint-1000")


if __name__ == "__main__":
    unittest.main()
