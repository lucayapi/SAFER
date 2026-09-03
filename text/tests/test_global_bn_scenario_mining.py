import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "text/recurrent_scenarios")

import global_bn
import scenario_mining
from scenario_pipeline import StructuralEMResult, _edge_conditional_contrast_signed, _edge_conditional_strength


def _toy_matrix() -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    rows = [
        {"accident_id": "0", "A0__T01": 1, "A0__T02": 0, "A1__T01": 0, "B__T01": 1, "C__T01": 1},
        {"accident_id": "1", "A0__T01": 1, "A0__T02": 0, "A1__T01": 0, "B__T01": 1, "C__T01": 1},
        {"accident_id": "2", "A0__T01": 1, "A0__T02": 0, "A1__T01": 0, "B__T01": 1, "C__T01": 0},
        {"accident_id": "3", "A0__T01": 0, "A0__T02": 1, "A1__T01": 1, "B__T01": 1, "C__T01": 1},
        {"accident_id": "4", "A0__T01": 0, "A0__T02": 0, "A1__T01": 0, "B__T01": 0, "C__T01": 0},
        {"accident_id": "5", "A0__T01": 1, "A0__T02": 0, "A1__T01": 0, "B__T01": 1, "C__T01": 1},
        {"accident_id": "6", "A0__T01": 1, "A0__T02": 0, "A1__T01": 0, "B__T01": 1, "C__T01": 1},
        {"accident_id": "7", "A0__T01": 1, "A0__T02": 0, "A1__T01": 0, "B__T01": 1, "C__T01": 1},
    ]
    matrix = pd.DataFrame(rows)
    roles = {
        "A0__T01": "A0", "A0__T02": "A0", "A1__T01": "A1",
        "B__T01": "B", "C__T01": "C",
    }
    dictionary = pd.DataFrame([
        {"variable_name": name, "topic_label": name, "role": roles[name]}
        for name in roles
    ])
    return matrix, roles, dictionary


def _toy_result(roles: dict[str, str], n_rows: int = 8) -> StructuralEMResult:
    nodes = list(roles)
    edges = [("A0__T01", "B__T01"), ("B__T01", "C__T01")]
    downstream = {
        ("B__T01", (1,)): 0.8,
        ("B__T01", (0,)): 0.4,
        ("C__T01", (1,)): 0.9,
        ("C__T01", (0,)): 0.1,
    }
    return StructuralEMResult(
        nodes, roles, edges, 1, np.array([1.0]), np.ones((n_rows, 1)),
        {}, downstream, 0.0, 0.0, 1, True, 0, "empty", None, [], 0.0, 0.0, True, 0, 0,
    )


def test_signed_conditional_contrast():
    _, roles, _ = _toy_matrix()
    result = _toy_result(roles)
    signed = _edge_conditional_contrast_signed(result, "B__T01", "C__T01")
    strength = _edge_conditional_strength(result, "B__T01", "C__T01")
    assert signed > 0
    assert abs(strength - abs(signed)) < 1e-12


def test_global_bn_fit_no_latent():
    matrix, roles, _ = _toy_matrix()
    config = {
        "random_state": 0,
        "bayesian_networks": {
            "d_max": 2,
            "structure_max_iter": 2,
            "structure_epsilon": 1e-6,
            "em_max_iter": 5,
            "em_tol": 1e-6,
            "graph_stability_patience": 1,
            "probability_floor": 1e-12,
            "alpha": 0.5,
            "latent_scope": "upstream_only",
            "bn_structure_bootstrap": {"enabled": False},
        },
    }
    result = global_bn.fit_global_bn(matrix, roles, config)
    assert result.n_states == 1
    assert "Z" not in result.nodes


def test_scenario_mining_recurrence_selection_no_pareto():
    matrix, roles, dictionary = _toy_matrix()
    result = _toy_result(roles, n_rows=len(matrix))
    bootstrap = pd.DataFrame([
        {"parent": "A0__T01", "child": "B__T01", "selection_frequency": 1.0},
        {"parent": "B__T01", "child": "C__T01", "selection_frequency": 0.8},
    ])
    config = {
        "scenario_mining": {
            "scenario_min_accident_count": 5,
            "max_upstream_factors_per_scenario": 2,
            "n_article_scenarios": 6,
        },
        "bayesian_networks": {"bn_display_bootstrap_threshold": 0.60},
    }
    with tempfile.TemporaryDirectory(dir="text") as tmp:
        out = Path(tmp)
        mining = scenario_mining.mine_recurrent_scenarios(
            matrix, roles, result, bootstrap, dictionary, config, out,
        )
        candidates = mining["candidates_all"]
        recurrent = mining["recurrent_all"]
        article = mining["article"]
        assert "is_pareto" not in candidates.columns
        assert "is_pareto" not in recurrent.columns
        assert not (out / "scenario_pareto.csv").exists()
        assert (out / "recurrent_scenarios_all.csv").is_file()
        assert (out / "scenarios_article.csv").is_file()
        assert (recurrent["scenario_accident_count"] >= 5).all()
        assert recurrent["is_closed_pattern"].all()
        counts = article["scenario_accident_count"].tolist()
        assert counts == sorted(counts, reverse=True)
        summary = pd.read_csv(out / "scenario_threshold_summary.csv")
        assert {"threshold", "n_candidates", "n_closed_patterns"}.issubset(summary.columns)
        assert "n_pareto" not in summary.columns


def test_article_ranked_by_recurrence_only():
    frame = pd.DataFrame([
        {"scenario_id": "A", "is_closed_pattern": True, "scenario_accident_count": 10, "scenario_support": 0.10, "lift": 1.1, "confidence": 0.5},
        {"scenario_id": "B", "is_closed_pattern": True, "scenario_accident_count": 8, "scenario_support": 0.08, "lift": 9.0, "confidence": 0.9},
        {"scenario_id": "C", "is_closed_pattern": True, "scenario_accident_count": 12, "scenario_support": 0.12, "lift": 1.2, "confidence": 0.4},
    ])
    cfg = {"n_article_scenarios": 2, "min_accident_count": 5}
    article = scenario_mining._select_article_scenarios(frame, cfg)
    assert list(article["scenario_id"]) == ["C", "A"]


def test_matrix_reporting_exports():
    import matrix_reporting

    matrix, roles, dictionary = _toy_matrix()
    audit = pd.DataFrame([
        {"accident_id": str(i), "included_in_bn": True, "exclusion_reason": "", "n_observed_factors": 3, "n_A0": 1, "n_A1": 0, "n_B": 1, "n_C": 1}
        for i in range(len(matrix))
    ])
    with tempfile.TemporaryDirectory(dir="text") as tmp:
        out = Path(tmp)
        pd.DataFrame([{
            "variable_name": name,
            "topic_id": name,
            "role": roles[name],
            "topic_label": name,
            "observation_prevalence": float(matrix[name].mean()),
            "n_accidents_with_factor": int(matrix[name].sum()),
            "n_accidents_total": len(matrix),
        } for name in roles]).to_csv(out / "factor_prevalence.csv", index=False)
        exports = matrix_reporting.export_matrix_artifacts(matrix, roles, dictionary, audit, out)
        assert (out / "factor_pair_cooccurrence.csv").is_file()
        assert (out / "figures" / "conceptual_role_architecture.png").is_file()
        assert "factor_pair_cooccurrence" in exports
