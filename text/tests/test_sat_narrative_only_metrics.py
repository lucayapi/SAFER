import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sat_narrative_only_lodo import score


def test_exact_span_ignores_only_edge_whitespace_and_unicode_punctuation():
    text = '«  La victime tombe. »'
    row = {"raw_text": text, "units": [{"start": 3, "end": 20, "role": "B"}]}
    # Predicted span includes outer French quotes and spaces.
    assert score(row, [(0, len(text))]) == {"span_tp": 1, "span_pred": 1, "span_ref": 1}


def test_internal_punctuation_difference_remains_an_error():
    text = "Il tombe, puis se relève."
    row = {"raw_text": text, "units": [{"start": 0, "end": len(text), "role": "B"}]}
    assert score(row, [(0, text.index(","))])["span_tp"] == 0
