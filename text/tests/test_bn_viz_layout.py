"""Tests disposition colonnes LR et boîtes CPD (bn_visualization)."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from bn_pipeline.bn_visualization import (
    BBox,
    _layout_bn_columns_lr,
    _macro_of_node,
)


def _topic_var_map(nodes: list[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in nodes:
        parts = str(n).split("_")
        if len(parts) >= 4 and parts[0] == "macro" and parts[1] == "topic":
            out[n] = parts[2]
    return out


def _bboxes_overlap(a: BBox, b: BBox, margin: float = 0.02) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    return not (
        ax1 + margin <= bx0
        or bx1 + margin <= ax0
        or ay1 + margin <= by0
        or by1 + margin <= ay0
    )


@pytest.fixture
def sample_topic_nodes() -> list[str]:
    return [
        "macro_topic_A0_01",
        "macro_topic_A0_02",
        "macro_topic_A1_01",
        "macro_topic_A1_02",
        "macro_topic_B_01",
        "macro_topic_C_01",
        "macro_topic_C_02",
    ]


def test_layout_columns_lr_macro_order(sample_topic_nodes: list[str]):
    var_map = _topic_var_map(sample_topic_nodes)
    pos, bboxes = _layout_bn_columns_lr(sample_topic_nodes, var_map, row_gap=1.4)

    xs_by_macro: Dict[str, list[float]] = {}
    for n in sample_topic_nodes:
        m = _macro_of_node(n, var_map)
        xs_by_macro.setdefault(m, []).append(pos[n][0])

    assert max(xs_by_macro["A0"]) < min(xs_by_macro["A1"])
    assert max(xs_by_macro["A1"]) < min(xs_by_macro["B"])
    assert max(xs_by_macro["B"]) < min(xs_by_macro["C"])
    assert min(xs_by_macro["C"]) == max(pos[n][0] for n in sample_topic_nodes)

    assert len(pos) == len(bboxes) == len(sample_topic_nodes)
    for n in sample_topic_nodes:
        x0, y0, w, h = bboxes[n]
        cx, cy = pos[n]
        assert abs(cx - (x0 + w / 2.0)) < 1e-6
        assert abs(cy - (y0 + h / 2.0)) < 1e-6


def test_layout_no_vertical_overlap_same_column(sample_topic_nodes: list[str]):
    var_map = _topic_var_map(sample_topic_nodes)
    _pos, bboxes = _layout_bn_columns_lr(sample_topic_nodes, var_map, row_gap=1.4)

    by_macro: Dict[str, list[str]] = {}
    for n in sample_topic_nodes:
        by_macro.setdefault(_macro_of_node(n, var_map), []).append(n)

    for macro, group in by_macro.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda n: bboxes[n][1])
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            _, y0a, _, ha = bboxes[a]
            _, y0b, _, _ = bboxes[b]
            assert y0a + ha <= y0b + 0.01, f"chevauchement vertical {macro}: {a} vs {b}"


def test_bbox_anchor_lr_sides():
    from bn_pipeline.bn_visualization import _bbox_anchor_lr

    bbox: BBox = (1.0, 2.0, 1.0, 0.8)
    assert _bbox_anchor_lr(bbox, "right") == (2.0, 2.4)
    assert _bbox_anchor_lr(bbox, "left") == (1.0, 2.4)
