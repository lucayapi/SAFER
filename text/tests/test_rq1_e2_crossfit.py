import unittest
import hashlib
import sqlite3
from unittest.mock import patch

import numpy as np

from rq1_e2_crossfit import (E2_FEATURE_SLICES, baseline_threshold_from_features,
                            boundary_example, cached_probabilities, hash_group_split, sat,
                            key, variant_features)


class CrossfitSplitTests(unittest.TestCase):
    def test_feature_cache_probabilities_work_without_legacy_table(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE features (corpus TEXT, document_id TEXT, text_sha256 TEXT, "
                   "prob BLOB, states BLOB, offsets BLOB, n_tokens INTEGER, "
                   "PRIMARY KEY(corpus,document_id))")
        rows = []
        for corpus in ("btp", "metallurgie", "caou_chimie_plas", "bois_textille", "service_temporaire"):
            raw = "ab"
            row = {"corpus": corpus, "document_id": "d1", "raw_text": raw,
                   "units": [{"start": 0, "end": 2}], "excluded_spans": []}
            rows.append(row)
            probabilities = np.asarray([0.1, 0.9], dtype=np.float32)
            states = np.zeros((2, 128), dtype=np.float16)
            offsets = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
            db.execute("INSERT INTO features VALUES (?,?,?,?,?,?,?)",
                (corpus, "d1", hashlib.sha256(raw.encode()).hexdigest(), probabilities.tobytes(),
                 states.tobytes(), offsets.tobytes(), 2))
        db.commit()
        np.testing.assert_allclose(cached_probabilities(db, rows[0]), [0.1, 0.9], rtol=1e-6)
        with patch.object(sat.legacy, "spans_from_probabilities", return_value=[(0, 2)]), \
                patch.object(sat, "THRESHOLDS", (0.05,)):
            selected, curve = baseline_threshold_from_features(db, rows)
        self.assertEqual(selected["threshold"], 0.05)
        self.assertEqual(len(curve), 1)
        db.close()

    def test_identical_narratives_are_kept_in_one_fold(self):
        rows = []
        for corpus in ("btp", "metallurgie", "caou_chimie_plas", "bois_textille", "nicollin"):
            for index in range(6):
                rows.append({"corpus": corpus, "document_id": f"{corpus}_{index}",
                             "raw_text": f"unique {corpus} {index}"})
            rows += [
                {"corpus": corpus, "document_id": f"{corpus}_duplicate_a", "raw_text": "same text"},
                {"corpus": corpus, "document_id": f"{corpus}_duplicate_b", "raw_text": " same   text "},
            ]
        folds = hash_group_split(rows, 5, 12345)
        for corpus in {row["corpus"] for row in rows}:
            seen = {folds[key(row)] for row in rows if row["corpus"] == corpus}
            self.assertEqual(seen, set(range(5)))
            self.assertEqual(folds[(corpus, f"{corpus}_duplicate_a")],
                             folds[(corpus, f"{corpus}_duplicate_b")])
        duplicate_folds = {folds[key(row)] for row in rows if row["document_id"].endswith("duplicate_a")}
        self.assertEqual(len(duplicate_folds), 1)
        inner = hash_group_split(rows, 2, 54321, fraction=0.1)
        for corpus in {row["corpus"] for row in rows}:
            self.assertEqual({inner[key(row)] for row in rows if row["corpus"] == corpus}, {0, 1})
        self.assertEqual(len({inner[key(row)] for row in rows
                              if row["document_id"].endswith("duplicate_a")}), 1)

    def test_internal_gold_unit_cut_is_known_negative_but_uncovered_cut_is_unknown(self):
        raw = "abc def ghi"
        row = {"corpus": "btp", "document_id": "d1", "raw_text": raw,
               "units": [{"start": 0, "end": 3}, {"start": 8, "end": 11}],
               "excluded_spans": [{"start": 4, "end": 7}]}
        probs = np.full(len(raw), 0.0, dtype=np.float32)
        probs[0:3] = 0.9
        probs[4] = 0.9
        probs[7:10] = 0.9
        states = np.zeros((len(raw), 128), dtype=np.float16)
        offsets = np.column_stack((np.arange(len(raw)), np.arange(1, len(raw)+1))).astype(np.int32)
        example = boundary_example(row, (probs, states, offsets), 0.5)
        labels = dict(zip(example.ranks, example.labels, strict=True))
        self.assertEqual(labels[1], 0.0)
        self.assertEqual(labels[2], 0.0)
        self.assertEqual(labels[3], 1.0)
        self.assertEqual(labels[6], 1.0)
        self.assertEqual(labels[8], 0.0)
        self.assertEqual(labels[4], -1.0)

    def test_e2_ablation_feature_dimensions(self):
        raw = "abc"
        row = {"corpus": "btp", "document_id": "d", "raw_text": raw,
               "units": [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
               "excluded_spans": []}
        probabilities = np.asarray([0.0, 0.9, 0.0], dtype=np.float32)
        states = np.zeros((3, 128), dtype=np.float16)
        offsets = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        base = boundary_example(row, (probabilities, states, offsets), 0.5)
        for name, (start, end) in E2_FEATURE_SLICES.items():
            example = variant_features(base, name)
            self.assertEqual(example.features.shape[1], end-start)


if __name__ == "__main__":
    unittest.main()
