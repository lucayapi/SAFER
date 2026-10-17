"""Focused checks for PySBD offsets and RQ1-compatible scoring."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pysbd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rq1_pysbd_baseline.py"
spec = importlib.util.spec_from_file_location("rq1_pysbd_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class PySBDBaselineTests(unittest.TestCase):
    def test_native_offsets_keep_original_text_and_whitespace(self) -> None:
        raw = "  L'ouvrier travaillait.\n\nIl est tombé.  "
        segmenter = pysbd.Segmenter(language="fr", clean=False, char_span=True)
        spans = baseline.pysbd_spans(raw, segmenter)
        self.assertEqual([raw[a:b].strip() for a, b in spans],
                         ["L'ouvrier travaillait.", "Il est tombé."])
        self.assertEqual(baseline.check_cached_spans(raw, [list(item) for item in spans]),
                         spans)

    def test_neutralized_boundaries_match_rq1_convention(self) -> None:
        row = {"raw_text": "A fait.  B blessé.",
               "units": [{"start": 0, "end": 7, "role": "A0"},
                         {"start": 9, "end": 18, "role": "C"}]}
        predicted = [(0, 9), (9, 18)]
        score = baseline.full_score(row, predicted)
        self.assertEqual((score["span_tp"], score["span_pred"], score["span_ref"]),
                         (2, 2, 2))
        self.assertEqual((score["boundary_tp"], score["boundary_pred"],
                          score["boundary_ref"]), (1, 1, 1))
        self.assertEqual(baseline.full_score(row, [(0, 18)])["fusion"], 1)

    def test_informative_gaps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "omit informative text"):
            baseline.check_cached_spans("Un fait. Puis un autre.", [[0, 8], [14, 22]])

    def test_cache_resume_keeps_existing_predictions(self) -> None:
        signature = {"test": "same input and code"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with closing(baseline.open_cache(path, signature)) as db:
                db.execute("INSERT INTO predictions VALUES (?,?,?,?)",
                           ("btp", "one", "abc", "[[0,1]]"))
                db.commit()
            with closing(baseline.open_cache(path, signature)) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM predictions").fetchone()[0], 1)
            with self.assertRaisesRegex(ValueError, "different data or code"):
                baseline.open_cache(path, {"test": "changed"})


if __name__ == "__main__":
    unittest.main()
