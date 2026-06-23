"""Tests sous-graphe slide BN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bn_pipeline.bn_visualization import (
    extract_subgraph_for_slide,
    path_nodes_from_scenario_row,
    plot_bn_scenario_slide,
)


@dataclass
class _MockBnModel:
    edges_list: list[tuple[str, str]]

    def edges(self) -> list[tuple[str, str]]:
        return list(self.edges_list)


def test_extract_subgraph_for_slide_returns_path_only():
    model = _MockBnModel(
        [
            ("macro_topic_A0_01", "macro_topic_A1_01"),
            ("macro_topic_A1_01", "macro_topic_B_01"),
            ("macro_topic_B_01", "macro_topic_C_01"),
        ]
    )
    path = [
        "macro_topic_A0_01",
        "macro_topic_A1_01",
        "macro_topic_B_01",
        "macro_topic_C_01",
    ]
    macro_map = {
        "macro_topic_A0_01": "A0",
        "macro_topic_A1_01": "A1",
        "macro_topic_B_01": "B",
        "macro_topic_C_01": "C",
    }
    nodes, edges = extract_subgraph_for_slide(model, path, macro_map)
    assert nodes == path
    assert edges == [
        ("macro_topic_A0_01", "macro_topic_A1_01"),
        ("macro_topic_A1_01", "macro_topic_B_01"),
        ("macro_topic_B_01", "macro_topic_C_01"),
    ]


def test_path_nodes_from_scenario_row_path_nodes():
    row = pd.Series({"path_nodes": "a -> b -> c"})
    assert path_nodes_from_scenario_row(row) == ["a", "b", "c"]


def test_path_nodes_from_scenario_row_topics_present_fallback():
    row = pd.Series({"topics_present": "x+y+z"})
    assert path_nodes_from_scenario_row(row) == ["x", "y", "z"]


@patch("bn_pipeline.bn_visualization.plot_bn_graph_cpd_boxes")
def test_plot_bn_scenario_slide_delegates(mock_plot):
    model = _MockBnModel([("a", "b")])
    row = pd.Series({"path_nodes": "a -> b", "macro_path": "A0→B"})
    out = Path("slide.png")
    ok = plot_bn_scenario_slide(model, row, {"a": "A0", "b": "B"}, out, rank=0)
    assert ok is True
    mock_plot.assert_called_once()
    assert mock_plot.call_args.kwargs["nodes_subset"] == ["a", "b"]
