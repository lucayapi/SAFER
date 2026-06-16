"""Tests squelette macro chain et interdiction A0→B."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_st_path = Path(__file__).resolve().parents[1] / "bn_pipeline" / "bn_structure.py"
_spec = importlib.util.spec_from_file_location("bn_structure", _st_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

build_blacklist = _mod.build_blacklist
ensure_inter_macro_chain_edges = _mod.ensure_inter_macro_chain_edges


def _topic_map() -> dict[str, str]:
    return {
        "macro_topic_A0_01": "A0",
        "macro_topic_A1_02": "A1",
        "macro_topic_B_03": "B",
        "macro_topic_C_04": "C",
    }


def test_blacklist_allows_a0_to_b_when_not_disallowed():
    nodes = list(_topic_map().keys())
    bl = set(build_blacklist(nodes, _topic_map(), disallow_a0_to_b_direct=False))
    assert ("macro_topic_A0_01", "macro_topic_B_03") not in bl


def test_blacklist_disallow_a0_to_b():
    nodes = list(_topic_map().keys())
    bl = set(build_blacklist(nodes, _topic_map(), disallow_a0_to_b_direct=True))
    assert ("macro_topic_A0_01", "macro_topic_B_03") in bl
    assert ("macro_topic_A0_01", "macro_topic_A1_02") not in bl


def test_ensure_inter_macro_chain_edges_adds_missing_tiers():
    nodes = list(_topic_map().keys())
    var_map = _topic_map()
    bl = set(build_blacklist(nodes, var_map))
    df = pd.DataFrame(
        [
            {"macro_topic_A0_01": 1, "macro_topic_A1_02": 1, "macro_topic_B_03": 0, "macro_topic_C_04": 0},
            {"macro_topic_A0_01": 1, "macro_topic_A1_02": 1, "macro_topic_B_03": 1, "macro_topic_C_04": 1},
            {"macro_topic_A0_01": 0, "macro_topic_A1_02": 0, "macro_topic_B_03": 1, "macro_topic_C_04": 1},
        ]
    )
    edges = [("macro_topic_B_03", "macro_topic_C_04")]
    out = ensure_inter_macro_chain_edges(edges, nodes, var_map, df, bl)
    edge_set = set(out)
    assert ("macro_topic_A0_01", "macro_topic_A1_02") in edge_set
    assert ("macro_topic_A1_02", "macro_topic_B_03") in edge_set
    assert ("macro_topic_B_03", "macro_topic_C_04") in edge_set
