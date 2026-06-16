"""Tests formatage cartes CPD et libellés courts BN."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from bn_pipeline.bn_visualization import (
    _draw_cpd_node_box,
    _wrap_node_title,
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
    assert lines[1].startswith("█") or lines[1].startswith("░")
    assert "33.2%" in lines[1]
    assert lines[2].startswith("█") or lines[2].startswith("░")
    assert "66.8%" in lines[2]
    assert not any(line.startswith(("0  ", "1  ", "Absent", "Présent")) for line in lines[1:])


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


def test_wrap_node_title():
    wrapped = _wrap_node_title("Intervention de maintenance sur une ligne de production métallique", width=22)
    lines = wrapped.splitlines()
    assert len(lines) >= 2
    assert all(len(line) <= 22 for line in lines)


def test_draw_cpd_node_box_title_only():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1.5))
    long_title = "Intervention de maintenance sur une ligne de production métallique"
    _draw_cpd_node_box(ax, (0.0, 0.0, 1.8, 1.0), long_title, macro="A0", wrap_width=22)
    assert len(fig.axes) == 1
    text_obj = ax.texts[0]
    assert "\n" in text_obj.get_text()
    assert "maintenance" in text_obj.get_text()
    assert text_obj.get_color() == "#2E6F9E"
    patch = ax.patches[0]
    assert patch.get_edgecolor()[:3] != patch.get_facecolor()[:3]
    plt.close(fig)
