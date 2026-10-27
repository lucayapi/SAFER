from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_narrative_only_saved_runs import partial_counts


def test_heterogeneous_merge_and_one_to_one_iou():
    row = {"corpus": "demo", "document_id": "1", "raw_text": "AAA BBB CCC",
           "units": [{"start": 0, "end": 3, "role": "A0"},
                     {"start": 4, "end": 7, "role": "B"},
                     {"start": 8, "end": 11, "role": "C"}]}
    counts = partial_counts(row, [(0, 7), (8, 11)])
    assert counts["heterogeneous_merges"] == 1
    assert counts["all_merges"] == 1
    assert counts["iou_tp"] == 1  # The 7-character merge has IoU 3/7 with each unit.
    assert counts["span_tp"] == 1
    assert counts["span_pred"] == 2
    assert counts["span_ref"] == 3


def test_unicode_edge_cleanup_does_not_change_internal_content():
    row = {"corpus": "demo", "document_id": "2", "raw_text": "(AAA) BBB",
           "units": [{"start": 1, "end": 4, "role": "A0"},
                     {"start": 6, "end": 9, "role": "B"}]}
    counts = partial_counts(row, [(0, 5), (6, 9)])
    assert counts["span_tp"] == 2
    assert counts["iou_tp"] == 2
    assert counts["heterogeneous_merges"] == 0
