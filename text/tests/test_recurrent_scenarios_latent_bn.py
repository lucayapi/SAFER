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
            "exact_inference": True,
        },
    }


def test_role_constraints_and_parameter_count():
    matrix, roles = _synthetic_matrix()
    del matrix
    allowed = pipeline._allowed_bn_edges(roles)
    assert all((roles[parent], roles[child]) in pipeline.BN_ROLE_ARCS for parent, child in allowed)
    assert pipeline._bn_parameter_count(list(roles), roles, [], 2) == 1 + 2 * 2 + 2 * 1


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
