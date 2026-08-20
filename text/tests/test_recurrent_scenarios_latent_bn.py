import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "text/recurrent_scenarios")

import scenario_pipeline as pipeline


def _synthetic_matrix() -> tuple[pd.DataFrame, dict[str, str]]:
    matrix = pd.DataFrame(
        [
            {"accident_id": str(index), "A0__T01": int(index < 6), "A1__T01": int(index % 2 == 0), "B__T01": int(index % 3 == 0), "C__T01": int(index % 4 == 0)}
            for index in range(12)
        ]
    )
    roles = {"A0__T01": "A0", "A1__T01": "A1", "B__T01": "B", "C__T01": "C"}
    return matrix, roles


def _small_config() -> dict:
    return {
        "random_state": 42,
        "bayesian_networks": {
            "latent_states": [2],
            "n_initializations": 1,
            "em_max_iter": 100,
            "structure_max_iter": 1,
            "min_latent_effective_n": 1,
            "d_max": 2,
        },
    }


def test_role_constraints_and_parameter_count():
    matrix, roles = _synthetic_matrix()
    del matrix
    allowed = pipeline._allowed_bn_edges(roles)
    assert all((roles[parent], roles[child]) in pipeline.BN_ROLE_ARCS for parent, child in allowed)
    assert pipeline._bn_parameter_count(list(roles), roles, [], 2) == 1 + 2 * 2 + 2 * 1


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


def test_structural_em_normalization_and_pgmpy_model():
    matrix, roles = _synthetic_matrix()
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        result, selection = pipeline.fit_latent_bn_analysis(matrix, roles, _small_config(), Path(temporary_directory))
    final = pipeline.finalize_latent_bn(result, matrix, roles, _small_config())
    assert selection["selected_final"].sum() == 1
    assert np.allclose(final.responsibilities.sum(axis=1), 1.0)
    assert np.isclose(final.weights.sum(), 1.0)
    assert final.model.check_model()


def test_frozen_inputs_exclude_noise_and_preserve_configuration():
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
            role_dir = run_dir / "pareto" / role / "candidate_partitions"
            role_dir.mkdir(parents=True, exist_ok=True)
            np.save(role_dir / f"{role}_cfg_001_labels.npy", np.array([0, 1, 1, -1]))
            np.save(role_dir / f"{role}_cfg_001_membership_strength.npy", np.array([0.9, 0.8, 0.7, 0.1]))
        dictionary_dir = run_dir / "topics_manual"
        dictionary_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {"topic_id": f"{role}_000", "role": role, "configuration_id": f"{role}_cfg_001", "llm_label": f"label-{role}"}
                for role in pipeline.ROLES
            ]
        ).to_csv(dictionary_dir / "topic_dictionary_with_llm_labels.csv", index=False)
        config = {"bayesian_networks": {"min_theme_support_count": 2}}
        matrix, included, excluded, roles = pipeline.build_frozen_bn_inputs(
            units,
            run_dir,
            {role: f"{role}_cfg_001" for role in pipeline.ROLES},
            config,
            root / "outputs",
        )
    assert "A0__T02" in matrix.columns
    assert set(included["role"]) == set(pipeline.ROLES)
    assert set(excluded["topic_id"]) == {f"{role}_000" for role in pipeline.ROLES}
    assert set(roles.values()) == set(pipeline.ROLES)


def test_exact_mpe_and_positive_support_above_twenty_variables():
    nodes = [f"A0__T{index:02d}" for index in range(1, 10)]
    nodes += [f"A1__T{index:02d}" for index in range(1, 5)]
    nodes += [f"B__T{index:02d}" for index in range(1, 5)]
    nodes += [f"C__T{index:02d}" for index in range(1, 5)]
    roles = {node: node.split("__", 1)[0] for node in nodes}
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame(rng.integers(0, 2, size=(30, len(nodes))), columns=nodes)
    matrix.insert(0, "accident_id", [str(index) for index in range(len(matrix))])
    config = {
        "random_state": 4,
        "bayesian_networks": {
            "latent_states": [2],
            "n_initializations": 1,
            "em_max_iter": 2,
            "structure_max_iter": 0,
            "min_latent_effective_n": 1,
            "d_max": 2,
            "show_progress": False,
        },
    }
    result = pipeline._fit_structural_em_initialization(
        matrix[nodes].to_numpy(dtype=np.int8), nodes, roles, 2, 8, "empty", config
    )
    final = pipeline.finalize_latent_bn(result, matrix, roles, config)
    units = pd.DataFrame(
        {
            "_accident_id": matrix["accident_id"],
            "_fact_id": matrix["accident_id"].map(lambda value: f"fact-{value}"),
            "_role": ["A0"] * len(matrix),
            "_text": ["text"] * len(matrix),
        }
    )
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        scenarios, supports, _, _ = pipeline.extract_latent_bn_scenarios(
            final, matrix, roles, units, config, Path(temporary_directory)
        )
    assert (scenarios["mpe_method"] == "exact_variable_elimination").all()
    assert "exact_vector_support" not in scenarios.columns
    assert "exact_vector_support" not in supports.columns
