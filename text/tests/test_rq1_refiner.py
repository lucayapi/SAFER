"""Small, deterministic checks for the refiner's offset and decoding rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rq1_refiner as refiner  # noqa: E402


def test_nonspace_ranks_and_original_offsets():
    raw = "A  B\nC"
    prefix, starts, ends = refiner.nonspace_geometry(raw)
    assert prefix.tolist() == [0, 1, 1, 1, 2, 2, 3]
    assert refiner.spans_from_ranks(raw, [(0, 1), (1, 2), (2, 3)]) == [
        (0, 1), (3, 4), (5, 6)]
    probabilities = np.zeros(len(raw), dtype=np.float32)
    probabilities[0] = 0.02
    probabilities[3] = 0.03
    assert refiner.candidate_ranks(raw, probabilities, prefix, 0.02) == [0, 2, 3]


def test_decoder_prefers_two_facts_to_fusion_and_can_leave_gap():
    spans = [(0, 1), (0, 3), (2, 3)]
    assert refiner.decode_spans(spans, np.array([2.0, 2.5, 2.0]), 0.0) == [
        (0, 1), (2, 3)]
    assert refiner.decode_spans(spans, np.array([-1.0, -2.0, -1.0]), 0.0) == []


def test_unknown_gap_is_not_a_negative_and_autre_is_positive():
    row = {"corpus": "example", "document_id": "x", "raw_text": "A B C",
           "units": [{"start": 0, "end": 1, "role": "A0"},
                     {"start": 4, "end": 5, "role": "AUTRE"}],
           "excluded_spans": []}
    probabilities = np.array([0.03, 0.0, 0.03, 0.0, 0.03], dtype=np.float32)
    states = np.ones((3, refiner.PROJECTION_DIM), dtype=np.float16)
    offsets = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    example = refiner.make_example(row, (probabilities, states, offsets),
                                    0.02, 5, "E3")
    labels = dict(zip(example.spans, example.labels.tolist()))
    assert labels[(0, 1)] == 1
    assert labels[(2, 3)] == 1
    assert labels[(1, 2)] == -1
    assert labels[(0, 2)] == 0


def test_source_only_threshold_grid_does_not_include_target():
    assert refiner.THRESHOLDS == (0.02, 0.05, 0.10)
    assert refiner.SEEDS == (13, 23, 37)


def test_small_cpu_fit_saves_source_selected_checkpoint(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(refiner, "EPOCHS", 1)
    examples = []
    for index in range(5):
        row = {"corpus": f"source_{index}", "document_id": str(index),
               "raw_text": "AB", "units": [{"start": 0, "end": 1, "role": "A0"}]}
        features = np.zeros((3, 5 * refiner.PROJECTION_DIM + 3), dtype=np.float16)
        features[0, 0], features[1, 1], features[2, 2] = 1, 1, 1
        examples.append(refiner.Example(row, 0.02, [0, 1, 2],
                        [(0, 1), (0, 2), (1, 2)], features,
                        np.array([1, 0, 0], dtype=np.int8), [], [], "E3"))
    selection = refiner.train_network("E3", 13, examples,
        {threshold: examples for threshold in refiner.THRESHOLDS}, tmp_path, "cpu")
    saved = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert saved["source_validation"]["threshold"] == selection["threshold"]
    assert saved["source_validation"]["macro_exact_f1"] == selection["macro_exact_f1"]
    model = refiner.new_network("E3")
    model.load_state_dict(saved["state_dict"])
    assert (tmp_path / "history.json").is_file()
