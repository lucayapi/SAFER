#!/usr/bin/env python3
"""Boundary-normalized exact-span scoring for narrative-only experiments.

Leading and trailing Unicode whitespace/separators, punctuation, symbols, and
control/format characters are ignored. Characters inside a span are preserved.
"""

from __future__ import annotations

import unicodedata


NORMALIZATION = (
    "trim Unicode categories P/S/C/Z at span edges only; preserve internal characters"
)
METRIC = "boundary_normalized_exact_f1"
COUNTS = ("span_tp", "span_pred", "span_ref")


def is_ignorable_edge_char(char: str) -> bool:
    category = unicodedata.category(char)
    return char.isspace() or category[0] in {"P", "S", "C", "Z"}


def trim_edges(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and is_ignorable_edge_char(text[start]):
        start += 1
    while end > start and is_ignorable_edge_char(text[end - 1]):
        end -= 1
    return start, end


def score(row: dict, predicted: list[tuple[int, int]]) -> dict[str, int]:
    raw = row["raw_text"]
    gold = {
        span for unit in row["units"]
        if (span := trim_edges(raw, int(unit["start"]), int(unit["end"])))[0] < span[1]
    }
    pred = {
        span for start, end in predicted
        if (span := trim_edges(raw, int(start), int(end)))[0] < span[1]
    }
    return {"span_tp": len(gold & pred), "span_pred": len(pred), "span_ref": len(gold)}


def prf(counts: dict[str, int]) -> tuple[float, float, float]:
    tp, predicted, reference = counts["span_tp"], counts["span_pred"], counts["span_ref"]
    precision = tp / predicted if predicted else 0.0
    recall = tp / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
