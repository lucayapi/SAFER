"""Tests chargement configs/accident_macros.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.macro_definitions import (
    MACRO_ORDER,
    format_macro_context_for_prompt,
    get_macro_definition,
    load_macro_definitions,
)


def test_load_all_macros():
    registry = load_macro_definitions(anchor=TEXT_ROOT)
    assert set(registry.keys()) == set(MACRO_ORDER)
    for mid in MACRO_ORDER:
        spec = registry[mid]
        assert spec.title
        assert spec.question
        assert spec.label_guidance
        assert spec.description


def test_format_macro_context_a0():
    block = format_macro_context_for_prompt("A0", anchor=TEXT_ROOT)
    assert block is not None
    assert "Macro : A0" in block
    assert "Contexte de travail" in block
    assert "Dans quelle situation de travail" in block
    assert "Consigne pour le libellé" in block


def test_get_macro_unknown():
    assert get_macro_definition("ZZ", anchor=TEXT_ROOT) is None
    assert format_macro_context_for_prompt("ZZ", anchor=TEXT_ROOT) is None
