"""Tests formatage cartes CPD et libellés courts BN."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from bn_malt.bn_visualization import (
    build_node_short_title,
    build_node_summary_label,
    cpd_binary_marginal,
    format_node_card,
    format_prob_bar,
    load_openai_themes_for_bn,
)


def test_format_prob_bar():
    bar = format_prob_bar(0.332, width=10)
    assert "33.2%" in bar
    assert "█" in bar
    assert "░" in bar
    assert len(bar.split("  ")[0]) == 10


def test_format_node_card():
    card = format_node_card("A1_Defaut_protection", [(0, 0.332), (1, 0.668)])
    lines = card.splitlines()
    assert lines[0] == "A1_Defaut_protection"
    assert lines[1].startswith("0  ")
    assert lines[2].startswith("1  ")
    assert "66.8%" in lines[2]


def test_build_node_short_title_with_themes():
    themes = pd.DataFrame(
        {
            "z_id": [12],
            "theme_summary": ["défaut protection équipement chantier"],
            "dominant_macro": ["A1"],
        }
    )
    title = build_node_short_title("Z_12_A1", themes, {"Z_12_A1": "A1"})
    assert title == "défaut protection équipement chantier"


def test_build_node_short_title_fallback():
    title = build_node_short_title("Z_3_B", None, {"Z_3_B": "B"})
    assert title.startswith("B")
    assert "motif z=3" in title


def test_build_node_summary_ignores_keywords_and_top_words():
    themes = pd.DataFrame(
        {
            "z_id": [5],
            "theme_summary": ["chute hauteur échafaudage sécurité"],
            "theme_keywords": ["chute;hauteur;échafaudage;ligne;garde"],
            "top_words": "chute hauteur ligne garde corps",
            "theme_title": "Chutes",
        }
    )
    title = build_node_summary_label("Z_05_B", themes, {"Z_05_B": "B"})
    assert title == "chute hauteur échafaudage sécurité"
    assert "chute;hauteur" not in title
    assert "ligne garde" not in title


def test_load_openai_themes_requires_theme_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bad = root / "themes_by_z.csv"
        pd.DataFrame(
            {"z_id": [1], "top_words": ["foo bar"], "top_sentences": [""]}
        ).to_csv(bad, index=False)
        try:
            load_openai_themes_for_bn(root)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
        good = root / "themes_by_z_openai.csv"
        pd.DataFrame(
            {
                "z_id": [1],
                "dominant_macro": ["A0"],
                "theme_summary": ["foo bar baz qux quux corge grault"],
            }
        ).to_csv(good, index=False)
        df = load_openai_themes_for_bn(root)
        assert len(df) == 1
        assert df.iloc[0]["theme_summary"].startswith("foo bar")


def test_cpd_binary_marginal_root():
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import BayesianNetwork

    model = BayesianNetwork([("A", "B")])
    cpd_a = TabularCPD("A", 2, values=[[0.7], [0.3]])
    cpd_b = TabularCPD("B", 2, values=[[0.6, 0.4], [0.2, 0.8]], evidence=["A"], evidence_card=[2])
    model.add_cpds(cpd_a, cpd_b)
    probs = cpd_binary_marginal(model, "A")
    assert len(probs) == 2
    assert abs(probs[0][1] + probs[1][1] - 1.0) < 1e-6
    assert abs(probs[0][1] - 0.7) < 1e-6
