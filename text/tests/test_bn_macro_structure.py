"""Tests contraintes DAG macro BN (A0→A1, A0→B, A1→B, B→C)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_st_path = Path(__file__).resolve().parents[1] / "bn_pipeline" / "bn_structure.py"
_spec = importlib.util.spec_from_file_location("bn_structure", _st_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

build_blacklist = _mod.build_blacklist
macro_chain_edge_list = _mod.macro_chain_edge_list
STANDARD_ALLOWED_MACRO_EDGES = _mod.STANDARD_ALLOWED_MACRO_EDGES


def _topic_map():
    return {
        "macro_topic_A0_01": "A0",
        "macro_topic_A1_02": "A1",
        "macro_topic_B_03": "B",
        "macro_topic_C_04": "C",
    }


def test_standard_allowed_edges():
    assert STANDARD_ALLOWED_MACRO_EDGES == {("A0", "A1"), ("A0", "B"), ("A1", "B"), ("B", "C")}


def test_blacklist_blocks_skip_to_c():
    nodes = list(_topic_map().keys())
    bl = set(build_blacklist(nodes, _topic_map()))
    assert ("macro_topic_A0_01", "macro_topic_C_04") in bl
    assert ("macro_topic_A1_02", "macro_topic_C_04") in bl


def test_blacklist_allows_required_edges():
    nodes = list(_topic_map().keys())
    bl = set(build_blacklist(nodes, _topic_map()))
    for edge in [
        ("macro_topic_A0_01", "macro_topic_A1_02"),
        ("macro_topic_A0_01", "macro_topic_B_03"),
        ("macro_topic_A1_02", "macro_topic_B_03"),
        ("macro_topic_B_03", "macro_topic_C_04"),
    ]:
        assert edge not in bl


def test_macro_chain_model_edges():
    edges = macro_chain_edge_list(severity_node=None)
    assert set(edges) == {
        ("M_A0", "M_A1"),
        ("M_A0", "M_B"),
        ("M_A1", "M_B"),
        ("M_B", "M_C"),
    }
