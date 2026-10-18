"""Offset and upper-bound invariants for the M3 candidate diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rq1_candidate_boundaries import document_counts  # noqa: E402


def test_threshold_and_newline_candidates_are_distinguished():
    raw = "A B\nC D"
    row = {"corpus": "fixture", "document_id": "one", "raw_text": raw,
           "units": [{"start": 0, "end": 3}, {"start": 4, "end": 7}]}
    probabilities = np.zeros(len(raw), dtype=np.float32)
    probabilities[2] = 0.8  # Cut after B, before the newline.
    low, candidates, gold = document_counts(row, probabilities, 0.5)
    high, _, _ = document_counts(row, probabilities, 0.9)
    assert candidates == gold == {2}
    assert low["score_candidates"] == 1
    assert low["forced_added"] == 0
    assert low["available_spans"] == 2
    assert high["score_candidates"] == 0
    assert high["forced_added"] == 1
    assert high["available_spans"] == 2
    assert high["score_available_spans"] == 0


def test_informative_gap_needs_both_endpoints():
    raw = "A X B"
    row = {"corpus": "fixture", "document_id": "two", "raw_text": raw,
           "units": [{"start": 0, "end": 1}, {"start": 4, "end": 5}]}
    probabilities = np.zeros(len(raw), dtype=np.float32)
    probabilities[0] = 0.8  # End of first fact.
    one, candidates, gold = document_counts(row, probabilities, 0.5)
    assert candidates == {1}
    assert gold == {1, 2}
    assert one["gap_gold_spans"] == 2
    assert one["available_spans"] == one["gap_available_spans"] == 1
    probabilities[2] = 0.8  # Start of second fact after the excluded X.
    both, _, _ = document_counts(row, probabilities, 0.5)
    assert both["available_spans"] == both["gap_available_spans"] == 2


def test_whitespace_boundary_ranks_are_neutralized():
    raw = "A  B"
    row = {"corpus": "fixture", "document_id": "three", "raw_text": raw,
           "units": [{"start": 0, "end": 2}, {"start": 3, "end": 4}]}
    probabilities = np.zeros(len(raw), dtype=np.float32)
    probabilities[1] = 0.5  # Cut inside peripheral whitespace.
    strict, _, _ = document_counts(row, probabilities, 0.5)
    accepted, candidates, gold = document_counts(row, probabilities, 0.49)
    assert strict["available_spans"] == 0
    assert candidates == gold == {1}
    assert accepted["available_spans"] == 2
