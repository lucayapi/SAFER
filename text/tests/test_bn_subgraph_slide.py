"""Tests sous-graphe slide BN."""

from __future__ import annotations

from dataclasses import dataclass

from bn_pipeline.bn_visualization import extract_subgraph_for_slide


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
