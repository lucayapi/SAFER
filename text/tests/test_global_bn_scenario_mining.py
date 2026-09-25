import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "text/recurrent_scenarios")

import global_bn
import scenario_mining
from scenario_pipeline import (
    StructuralEMResult,
    _edge_conditional_contrast_signed,
    _edge_conditional_contrast_strata,
    _edge_conditional_strength,
)


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
    matrix, _, _ = _toy_matrix()
    observed_data = matrix[nodes].to_numpy(dtype=np.int8)
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
        observed_data=observed_data,
    )


def test_signed_conditional_contrast():
    _, roles, _ = _toy_matrix()
    result = _toy_result(roles)
    signed = _edge_conditional_contrast_signed(result, "B__T01", "C__T01")
    strength = _edge_conditional_strength(result, "B__T01", "C__T01")
    assert signed > 0
    assert abs(strength - abs(signed)) < 1e-12


def test_conditional_contrast_excludes_unobserved_parent_strata():
    nodes = ["P", "Q", "Y"]
    roles = {"P": "A0", "Q": "A1", "Y": "B"}
    # Q=1, P=1 is never observed. Its conventional CPT value must not enter
    # the mean contrast.
    observed_data = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.int8)
    result = StructuralEMResult(
        nodes,
        roles,
        [("P", "Y"), ("Q", "Y")],
        1,
        np.array([1.0]),
        np.ones((len(observed_data), 1)),
        {},
        {
            ("Y", (0, 0)): 0.2,
            ("Y", (1, 0)): 0.8,
            ("Y", (0, 1)): 0.9,
            ("Y", (1, 1)): 0.5,
        },
        0.0,
        0.0,
        1,
        True,
        0,
        "empty",
        observed_data=observed_data,
    )
    strata = _edge_conditional_contrast_strata(result, "P", "Y")
    assert list(strata["estimable"]) == [True, False]
    assert np.isnan(strata.loc[1, "conditional_contrast"])
    assert _edge_conditional_contrast_signed(result, "P", "Y") == pytest.approx(0.6)


def test_global_bn_edge_export_includes_contrast_estimability_audit():
    matrix, roles, _ = _toy_matrix()
    result = _toy_result(roles)
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output_dir = Path(temporary_directory)
        edges = global_bn.write_global_bn_edges(
            result,
            matrix,
            {},
            roles,
            None,
            output_dir,
        )
        strata_path = output_dir / "global_bn_edge_contrast_strata.csv"
        assert strata_path.is_file()
        strata = pd.read_csv(strata_path)
        assert {
            "n_parent_0",
            "n_parent_1",
            "estimable",
            "conditional_contrast",
        }.issubset(strata.columns)
        assert {
            "conditional_contrast_n_estimable_strata",
            "conditional_contrast_n_total_strata",
        }.issubset(edges.columns)


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
    assert result.cpt_estimation_method == "MLE"
    assert result.mle_model is not None


def test_exact_learner_enumerates_all_local_parent_sets_and_uses_lexical_tie_break():
    matrix = pd.DataFrame({
        "accident_id": [str(index) for index in range(8)],
        "A0__P": [0, 0, 0, 0, 1, 1, 1, 1],
        "A0__Q": [0, 0, 0, 0, 1, 1, 1, 1],
        "A1__Y": [0, 0, 0, 0, 1, 1, 1, 1],
        "B__Y": [0, 0, 0, 0, 1, 1, 1, 1],
        "C__Y": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    roles = {"A0__P": "A0", "A0__Q": "A0", "A1__Y": "A1", "B__Y": "B", "C__Y": "C"}
    result = global_bn.fit_global_bn(matrix, roles, {"bayesian_networks": {"d_max": 2}})
    # 2 A0 roots (1 each), A1 (1+2+1), B (1+3+3), C (1+1).
    assert len(result.exact_local_scores) == 15
    selected = result.exact_selected_parent_sets.set_index("child_factor")
    assert selected.loc["A1__Y", "parent_factor_ids"] == "A0__P"
    assert selected.loc["A1__Y", "tie_break"] == "fewest_parents_then_factor_id_order"


def test_jeffreys_cpt_uses_half_for_an_unobserved_parent_configuration():
    data = np.array([[0, 0], [0, 0], [0, 1]], dtype=np.int8)
    probabilities, cpts = global_bn._jeffreys_cpts(
        data, ["A0__P", "A1__Y"], {"A0__P": "A0", "A1__Y": "A1"}, [("A0__P", "A1__Y")],
    )
    assert probabilities[("A1__Y", (1,))] == pytest.approx(0.5)
    assert probabilities[("A1__Y", (0,))] == pytest.approx(0.375)
    assert not cpts.empty


def test_cpt_support_diagnostic_retains_mle_when_all_selected_rows_are_observed():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    cpts = global_bn._postselection_cpt_counts(
        data, ["A0__P", "A1__Y"], {"A0__P": "A0", "A1__Y": "A1"}, [("A0__P", "A1__Y")],
    )
    diagnostic = global_bn._cpt_support_diagnostics(cpts, min_observed_count=0).iloc[0]
    mle = global_bn._cpt_probabilities(cpts, "MLE_P_child_1")
    assert diagnostic["n_rows_N_eq_0"] == 0
    assert diagnostic["n_rows_N_le_2"] == 2
    assert diagnostic["final_CPT_estimation"] == "MLE"
    assert mle[("A1__Y", (0,))] == pytest.approx(0.5)
    assert mle[("A1__Y", (1,))] == pytest.approx(0.5)


def test_cpt_support_diagnostic_requires_jeffreys_for_empty_selected_row():
    data = np.array([[0, 0], [0, 0], [0, 1]], dtype=np.int8)
    cpts = global_bn._postselection_cpt_counts(
        data, ["A0__P", "A1__Y"], {"A0__P": "A0", "A1__Y": "A1"}, [("A0__P", "A1__Y")],
    )
    diagnostic = global_bn._cpt_support_diagnostics(cpts, min_observed_count=0).iloc[0]
    assert diagnostic["n_rows_N_eq_0"] == 1
    assert diagnostic["final_CPT_estimation"] == "Jeffreys"
    with pytest.raises(ValueError, match="unobserved"):
        global_bn._cpt_probabilities(cpts, "MLE_P_child_1")


def test_exact_bootstrap_uses_full_accident_samples_and_is_deterministic():
    matrix, roles, _ = _toy_matrix()
    config = {"bayesian_networks": {"d_max": 2, "bn_structure_bootstrap": {"enabled": True, "n_resamples": 3, "random_state": 7}}}
    result = global_bn.fit_global_bn(matrix, roles, config)
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        first = global_bn.run_global_bn_bootstrap(matrix, roles, config, result, Path(temporary_directory))
        second = global_bn.run_global_bn_bootstrap(matrix, roles, config, result, Path(temporary_directory))
    assert first["bootstrap_sample_size"].eq(len(matrix)).all()
    assert first["n_resamples"].eq(3).all()
    pd.testing.assert_frame_equal(first, second)


def test_scenario_bn_support_marginalizes_unmentioned_variables():
    matrix = pd.DataFrame({
        "accident_id": [str(index) for index in range(8)],
        "A0__T01": [0, 0, 0, 0, 1, 1, 1, 1],
        "B__T01": [0, 0, 0, 0, 1, 1, 1, 1],
        "C__T01": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    roles = {"A0__T01": "A0", "B__T01": "B", "C__T01": "C"}
    config = {"bayesian_networks": {"d_max": 2}}
    result = global_bn.fit_global_bn(matrix, roles, config)
    scenarios = pd.DataFrame([{
        "scenario_id": "SC_1", "upstream_factor_ids": "A0__T01",
        "B_factor_id": "B__T01", "C_factor_id": "C__T01", "scenario_support": 0.5,
    }])
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        enriched = global_bn.add_scenario_bn_discrepancy(scenarios, result, len(matrix), Path(temporary_directory))
    assert result.edges == [("A0__T01", "B__T01"), ("B__T01", "C__T01")]
    expected = (
        result.downstream_probabilities[("A0__T01", ())]
        * result.downstream_probabilities[("B__T01", (1,))]
        * result.downstream_probabilities[("C__T01", (1,))]
    )
    assert enriched.loc[0, "BN_implied_support"] == pytest.approx(expected)
    assert enriched.loc[0, "BN_expected_accident_count"] == pytest.approx(len(matrix) * expected)


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
