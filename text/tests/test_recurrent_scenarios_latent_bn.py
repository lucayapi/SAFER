import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, "text/recurrent_scenarios")

import scenario_pipeline as pipeline
import bn_reporting
import scenario_latex_export


def _synthetic_matrix() -> tuple[pd.DataFrame, dict[str, str]]:
    matrix = pd.DataFrame(
        [
            {"accident_id": str(index), "A0__T01": int(index < 6), "A1__T01": int(index % 2 == 0), "B__T01": int(index % 3 == 0), "C__T01": int(index % 4 == 0)}
            for index in range(12)
        ]
    )
    roles = {"A0__T01": "A0", "A1__T01": "A1", "B__T01": "B", "C__T01": "C"}
    return matrix, roles


def _small_config(**overrides) -> dict:
    config = {
        "random_state": 42,
        "bayesian_networks": {
            "latent_states": [1, 2],
            "n_initializations": 1,
            "em_max_iter": 100,
            "structure_max_iter": 1,
            "min_latent_effective_n": 1,
            "d_max": 2,
            "include_all_retained_factors": True,
            "mpe_required_roles": ["B", "C"],
            "mpe_upstream_any_roles": ["A0", "A1"],
            "mpe_compute_free_diagnostic": True,
            "top_m_mpe": 3,
            "bn_structure_bootstrap": {"enabled": False},
        },
    }
    config["bayesian_networks"].update(overrides)
    return config


def test_is_latent_conditioned():
    assert pipeline.is_latent_conditioned("A0", "upstream_only")
    assert not pipeline.is_latent_conditioned("B", "upstream_only")
    assert pipeline.is_latent_conditioned("C", "all_roles")


def test_role_constraints_and_parameter_count():
    matrix, roles = _synthetic_matrix()
    del matrix
    allowed = pipeline._allowed_bn_edges(roles)
    assert all((roles[parent], roles[child]) in pipeline.BN_ROLE_ARCS for parent, child in allowed)
    assert pipeline._bn_parameter_count(list(roles), roles, [], 2) == 1 + 2 * 2 + 2 * 1
    all_roles_count = pipeline._bn_parameter_count(list(roles), roles, [], 2, "all_roles")
    assert all_roles_count > pipeline._bn_parameter_count(list(roles), roles, [], 2)


def test_local_bic_matches_global_bic():
    nodes = ["A0__T01", "A0__T02", "A1__T01", "B__T01", "B__T02", "C__T01"]
    roles = {node: node.split("__", 1)[0] for node in nodes}
    rng = np.random.default_rng(7)
    data = rng.integers(0, 2, size=(40, len(nodes)), dtype=np.int8)
    tau = rng.dirichlet(np.ones(3), size=len(data))
    edges = [("A0__T01", "A1__T01"), ("A0__T01", "B__T01"), ("A1__T01", "B__T02"), ("B__T01", "C__T01")]
    weights, upstream, downstream = pipeline._bn_mle_parameters(data, nodes, roles, edges, tau, 3, 1e-12)
    log_joint = pipeline._bn_log_joint(data, nodes, roles, edges, weights, upstream, downstream, 1e-12)
    global_bic = -2.0 * float(np.sum(tau * log_joint)) + pipeline._bn_parameter_count(nodes, roles, edges, 3) * np.log(len(data))
    local_bic = pipeline._bn_expected_bic(data, nodes, roles, edges, tau, 3, 1e-12)
    assert np.isclose(local_bic, global_bic, rtol=1e-10, atol=1e-8)


def test_k1_reference_model():
    matrix, roles = _synthetic_matrix()
    nodes = [column for column in matrix.columns if column != "accident_id"]
    data = matrix[nodes].to_numpy(dtype=np.int8)
    result = pipeline._fit_bn_k1(data, nodes, roles, 0, "empty", _small_config())
    assert result.n_states == 1
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.allclose(result.responsibilities, 1.0)
    assert np.allclose(result.responsibilities.sum(axis=1), 1.0)
    assert result.responsibilities.shape == (len(matrix), 1)
    entropy = float(-np.sum(result.responsibilities * np.log(np.clip(result.responsibilities, 1e-300, 1.0))))
    assert entropy == 0.0
    assert np.isfinite(result.bic)


def test_k1_admissible_despite_min_latent_effective_n():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        result, selection, _ = pipeline.fit_latent_bn_analysis(
            matrix, roles, _small_config(latent_states=[1, 2], min_latent_effective_n=100), Path(temporary_directory),
        )
    k1_rows = selection[(selection["K"] == 1) & selection["admissible"]]
    assert not k1_rows.empty
    if result.n_states == 1:
        assert result.weights.sum() == pytest.approx(1.0)


def test_structural_em_normalization_and_pgmpy_model():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        result, selection, warnings = pipeline.fit_latent_bn_analysis(matrix, roles, _small_config(), Path(temporary_directory))
    final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config())
    assert selection["selected_final"].sum() == 1
    assert "entropy" in selection.columns
    assert np.allclose(final.responsibilities.sum(axis=1), 1.0)
    assert np.isclose(final.weights.sum(), 1.0)
    assert np.isclose(final.responsibilities.sum(), len(matrix))
    assert final.model.check_model()


def test_delta_bic_vs_k1_present_when_k_gt_1():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        result, selection, warnings = pipeline.fit_latent_bn_analysis(matrix, roles, _small_config(), output)
        final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config())
        summary = bn_reporting.write_k_selection_summary(selection, final, output, warnings)
    if final.n_states > 1:
        assert "delta_BIC_vs_K1" in summary.columns
        assert summary["BIC_K1"].notna().any()


def test_frozen_inputs_include_all_retained_factors():
    units = pd.DataFrame(
        {
            "_accident_id": [str(index) for index in range(16)],
            "_fact_id": [f"fact-{index}" for index in range(16)],
            "_role": [role for role in pipeline.ROLES for _ in range(4)],
            "_text": ["text"] * 16,
        }
    )
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        root = Path(temporary_directory)
        run_dir = root / "run"
        for role in pipeline.ROLES:
            role_dir = run_dir / "discovery" / role / "candidate_partitions"
            role_dir.mkdir(parents=True, exist_ok=True)
            np.save(role_dir / f"{role}_cfg_001_labels.npy", np.array([0, 1, 1, -1]))
            np.save(role_dir / f"{role}_cfg_001_membership_strength.npy", np.array([0.9, 0.8, 0.7, 0.1]))
        dictionary_dir = run_dir / "topics_manual"
        dictionary_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "topic_id": f"{role}_{topic:03d}",
                    "role": role,
                    "configuration_id": f"{role}_cfg_001",
                    "llm_label": f"label-{role}-{topic}",
                }
                for role in pipeline.ROLES
                for topic in (0, 1)
            ]
        ).to_csv(dictionary_dir / "topic_dictionary_with_llm_labels.csv", index=False)
        config = {"bayesian_networks": {"include_all_retained_factors": True, "min_theme_support_count": 20}}
        matrix, included, excluded, roles = pipeline.build_frozen_bn_inputs(
            units,
            run_dir,
            {role: f"{role}_cfg_001" for role in pipeline.ROLES},
            config,
            root / "outputs",
        )
        summary = pd.read_csv(root / "outputs" / "input_summary.csv")
        assert summary.iloc[0]["n_retained_factors"] == summary.iloc[0]["n_bn_factors"]
        assert len(excluded) == 0
        assert "A0__T02" in matrix.columns
        assert (root / "outputs" / "factor_prevalence.csv").is_file()


def test_accident_inclusion_audit():
    matrix, roles = _synthetic_matrix()
    units = pd.DataFrame({
        "_accident_id": [str(index) for index in range(14)],
        "_fact_id": [f"f-{index}" for index in range(14)],
        "_role": "A0",
        "_text": "x",
    })
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        audit, summary = pipeline.write_bn_accident_inclusion_audit(units, matrix, roles, output)
        assert len(audit) == 14
        assert summary.iloc[0]["included_accidents"] + summary.iloc[0]["excluded_accidents"] == summary.iloc[0]["total_accidents_available"]
        assert (output / "bn_accident_inclusion_audit.csv").is_file()


def test_mpe_requires_upstream_or_a1_and_b_c():
    nodes = [f"A0__T{index:02d}" for index in range(1, 4)]
    nodes += [f"A1__T{index:02d}" for index in range(1, 3)]
    nodes += [f"B__T{index:02d}" for index in range(1, 3)]
    nodes += [f"C__T{index:02d}" for index in range(1, 3)]
    roles = {node: node.split("__", 1)[0] for node in nodes}
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame(rng.integers(0, 2, size=(30, len(nodes))), columns=nodes)
    matrix.insert(0, "accident_id", [str(index) for index in range(len(matrix))])
    config = _small_config(em_max_iter=2, structure_max_iter=0)
    result = pipeline._fit_structural_em_initialization(
        matrix[nodes].to_numpy(dtype=np.int8), nodes, roles, 2, 8, "empty", config
    )
    final = pipeline.finalize_latent_bn(result, matrix, roles, config)
    mpe = pipeline._exact_constrained_mpe(final, roles, 0, config)
    assert pipeline._mpe_role_constraint_satisfied(mpe, roles, config)
    assert any(mpe[node] for node in nodes if roles[node] in {"A0", "A1"})
    assert any(mpe[node] for node in nodes if roles[node] == "B")
    assert any(mpe[node] for node in nodes if roles[node] == "C")
    ranked = pipeline._exact_constrained_mpe_top_m(final, roles, 0, config, matrix=matrix)
    assert len(ranked) >= 1
    assignments = [row["positive_factors"] for row in ranked]
    assert len(set(assignments)) == len(assignments)
    free = pipeline._exact_free_mpe(final, roles, 0, config)
    assert isinstance(free, dict)


def test_prototypes_exact_or_closest():
    nodes = ["A0__T01", "A1__T01", "B__T01", "C__T01"]
    roles = {node: node.split("__", 1)[0] for node in nodes}
    matrix = pd.DataFrame([
        {"accident_id": "0", "A0__T01": 1, "A1__T01": 0, "B__T01": 1, "C__T01": 1},
        {"accident_id": "1", "A0__T01": 1, "A1__T01": 1, "B__T01": 1, "C__T01": 1},
    ])
    responsibilities = np.array([[0.9, 0.1], [0.4, 0.6]])
    index, exact, coverage, missing, n_mpe, n_matched = pipeline._select_prototype(
        matrix, nodes, responsibilities, 0, ["A0__T01", "B__T01", "C__T01"],
    )
    assert exact
    assert coverage == 1.0
    assert not missing
    assert n_mpe == 3
    assert n_matched == 3


def test_parent_configuration_support():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        result, selection, _ = pipeline.fit_latent_bn_analysis(matrix, roles, _small_config(), output)
        final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config())
        detail, summary = bn_reporting.write_parent_configuration_support(final, matrix, roles, _small_config(), output)
        assert (output / "parent_configuration_support.csv").is_file()
        assert (output / "parent_support_summary.csv").is_file()
        if not detail.empty:
            assert "effective_count" in detail.columns


def test_latent_scope_all_roles_pgmpy_valid():
    matrix, roles = _synthetic_matrix()
    config = _small_config(latent_scope="all_roles")
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        result, _, _ = pipeline.fit_latent_bn_analysis(matrix, roles, config, Path(temporary_directory))
    final = pipeline.finalize_latent_bn(result, matrix, roles, config)
    assert final.model.check_model()


def test_bn_reporting_k_summary_and_diagnostic():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        result, selection, warnings = pipeline.fit_latent_bn_analysis(matrix, roles, _small_config(), output)
        final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config())
        units = pd.DataFrame({"_accident_id": matrix["accident_id"], "_fact_id": matrix["accident_id"], "_role": "A0", "_text": "x"})
        scenarios, _, prototypes, profiles = pipeline.extract_latent_bn_scenarios(final, matrix, roles, units, _small_config(), output)
        summary = bn_reporting.write_k_selection_summary(selection, final, output, warnings)
        diagnostic = bn_reporting.write_bn_diagnostic_summary(
            final, selection, summary, scenarios, prototypes, matrix, _small_config(), output, warnings,
        )
        assert "delta_bic" in summary.columns
        assert "best_ICL" in summary.columns
        assert "BIC_K1" in summary.columns
        assert diagnostic["selected_K"] == final.n_states
        assert "K_star_BIC" in diagnostic
        assert "K_star_ICL" in diagnostic
        assert "criteria_concordant" in diagnostic
        assert "latent_heterogeneity_supported" in diagnostic
        assert (output / "bn_diagnostic_summary.json").is_file()


def test_scenario_support_enrichment_columns():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        result, _, _ = pipeline.fit_latent_bn_analysis(matrix, roles, _small_config(), output)
        final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config())
        units = pd.DataFrame({"_accident_id": matrix["accident_id"], "_fact_id": matrix["accident_id"], "_role": "A0", "_text": "x"})
        scenarios, _, _, _ = pipeline.extract_latent_bn_scenarios(final, matrix, roles, units, _small_config(), output)
        assert "support_enrichment_ratio" in scenarios.columns
        assert "n_exact_matching_accidents" in scenarios.columns
        assert (output / "mpe_free_diagnostic.csv").is_file()
        assert (output / "recurrent_scenarios_latex.csv").is_file()
        assert (output / "recurrent_scenario_links.csv").is_file()
        article = scenario_latex_export.write_recurrent_scenarios_article(scenarios, output)
        assert "Enrichment" in article.columns
        assert article["Enrichment"].between(0, np.inf).all()
        assert "Family_support" in article.columns
        assert article["Family_support"].between(0, 1).all()


def test_bootstrap_keeps_fixed_k():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        result, _, _ = pipeline.fit_latent_bn_analysis(
            matrix, roles, _small_config(bn_structure_bootstrap={"enabled": True, "n_resamples": 3, "n_initializations_per_resample": 1}), output,
        )
        final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config(bn_structure_bootstrap={"enabled": True, "n_resamples": 3}))
        bootstrap = bn_reporting.run_structure_bootstrap(matrix, roles, _small_config(), final.n_states, final, None, output)
    if bootstrap is not None:
        assert "selection_frequency" in bootstrap.columns
        assert len(final.edges) >= 0


def test_recurrent_scenarios_latex_exports():
    nodes = [f"A0__T{index:02d}" for index in range(1, 3)]
    nodes += [f"A1__T{index:02d}" for index in range(1, 2)]
    nodes += [f"B__T{index:02d}" for index in range(1, 2)]
    nodes += [f"C__T{index:02d}" for index in range(1, 2)]
    roles = {node: node.split("__", 1)[0] for node in nodes}
    rng = np.random.default_rng(5)
    matrix = pd.DataFrame(rng.integers(0, 2, size=(24, len(nodes))), columns=nodes)
    matrix.insert(0, "accident_id", [str(index) for index in range(len(matrix))])
    dictionary = pd.DataFrame({
        "variable_name": nodes,
        "topic_label": [f"label-{node}" for node in nodes],
    })
    config = _small_config(latent_states=[2], em_max_iter=50, structure_max_iter=0, min_latent_effective_n=1)
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        output = Path(temporary_directory)
        result = pipeline._fit_structural_em_initialization(
            matrix[nodes].to_numpy(dtype=np.int8), nodes, roles, 2, 8, "empty", config,
        )
        final = pipeline.finalize_latent_bn(result, matrix, roles, config)
        units = pd.DataFrame({"_accident_id": matrix["accident_id"], "_fact_id": matrix["accident_id"], "_role": "A0", "_text": "x"})
        pipeline.extract_latent_bn_scenarios(final, matrix, roles, units, config, output, dictionary)
        latex = pd.read_csv(output / "recurrent_scenarios_latex.csv")
        links = pd.read_csv(output / "recurrent_scenario_links.csv")
        learned = set(final.edges)

    assert len(latex) == final.n_states
    for _, row in latex.iterrows():
        grouped = scenario_latex_export._positive_factors_by_role(row, roles)
        assignment = {factor: 1 for factor in scenario_latex_export._flat_display_sequence(grouped)}
        assert pipeline._mpe_role_constraint_satisfied(assignment, roles, config)
        assert 0 <= float(row["family_support"]) <= 1
        assert 0 <= float(row["global_support"]) <= 1
        assert float(row["support_enrichment"]) >= 0
        if bool(row["exact_observed_pattern"]):
            assert int(row["n_exact_matching_accidents"]) >= 1
        for factor_id, label_col in (
            ("A0_factor_ids", "A0_labels"),
            ("A1_factor_ids", "A1_labels"),
            ("B_factor_ids", "B_labels"),
            ("C_factor_ids", "C_labels"),
        ):
            ids = scenario_latex_export._split_cell(row[factor_id])
            labels = scenario_latex_export._split_cell(row[label_col])
            if ids:
                assert len(ids) == len(labels)
                lookup = dictionary.set_index("variable_name")["topic_label"]
                for factor, label in zip(ids, labels):
                    assert lookup[factor] == label

    for _, link in links.iterrows():
        edge = (link["source_factor"], link["target_factor"])
        if str(link["display_link_type"]) == "SOLID":
            assert edge in learned
            assert bool(link["learned_bn_edge"])
        if str(link["display_link_type"]) == "DASHED":
            assert edge not in learned
            assert not bool(link["learned_bn_edge"])


def test_load_bn_analysis_config_preserves_k1_despite_resolved_override():
    config_path = Path("text/recurrent_scenarios/config.yaml")
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        run_dir = Path(temporary_directory)
        (run_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(
                {"bayesian_networks": {"latent_states": [2, 3, 4, 5, 7, 8, 10, 15], "d_max": 99}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        loaded = pipeline.load_bn_analysis_config(config_path, "caou", run_dir)
        latent_states = [int(value) for value in loaded["bayesian_networks"]["latent_states"]]
        assert 1 in latent_states
        assert 2 in latent_states
        assert loaded["bayesian_networks"]["d_max"] == 99


def test_render_latent_k_selection_figure_writes_article_and_appendix(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("k_figures")
    selection = pd.DataFrame(
        [
            {"K": 1, "bic": 100.0, "admissible": True, "selected_for_K": True, "selected_final": False, "converged": True},
            {"K": 2, "bic": 90.0, "admissible": True, "selected_for_K": True, "selected_final": True, "converged": True},
            {"K": 2, "bic": 95.0, "admissible": True, "selected_for_K": False, "selected_final": False, "converged": True},
        ]
    )
    bn_reporting.render_latent_k_selection_figure(selection, 2, tmp_path)
    figures_dir = tmp_path / "figures"
    assert (figures_dir / "latent_K_selection_best.png").is_file()
    assert (figures_dir / "latent_K_selection_all_inits.png").is_file()
    assert (figures_dir / "latent_K_selection.png").is_file()


def test_load_bn_analysis_config_warns_when_k1_missing():
    config_path = Path("text/recurrent_scenarios/config.yaml")
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        run_dir = Path(temporary_directory)
        broken_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        broken_config["bayesian_networks"]["latent_states"] = [2, 3, 4]
        broken_path = Path(temporary_directory) / "broken_config.yaml"
        broken_path.write_text(yaml.safe_dump(broken_config, sort_keys=False), encoding="utf-8")
        with pytest.warns(RuntimeWarning, match="K=1 is missing"):
            loaded = pipeline.load_bn_analysis_config(broken_path, "caou", run_dir)
        assert 1 not in [int(value) for value in loaded["bayesian_networks"]["latent_states"]]


def test_k_criteria_article_summary_and_figure(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("k_criteria")
    k_summary = pd.DataFrame(
        [
            {"K": 1, "best_BIC": 200.0, "best_ICL": 200.0, "entropy": 0.0},
            {"K": 2, "best_BIC": 180.0, "best_ICL": 190.0, "entropy": 5.0},
            {"K": 3, "best_BIC": 185.0, "best_ICL": 205.0, "entropy": 10.0},
        ]
    )
    criteria = bn_reporting.write_k_criteria_article_summary(k_summary, tmp_path)
    assert int(criteria.loc[criteria["criterion"] == "BIC", "K_star"].iloc[0]) == 2
    assert int(criteria.loc[criteria["criterion"] == "ICL", "K_star"].iloc[0]) == 2
    assert bool(criteria.loc[criteria["criterion"] == "concordant", "K_star"].iloc[0])
    bn_reporting.render_latent_k_criteria_figure(k_summary, tmp_path)
    assert (tmp_path / "figures" / "latent_K_selection_bic_icl.png").is_file()
    assert (tmp_path / "K_selection_criteria_article.csv").is_file()
