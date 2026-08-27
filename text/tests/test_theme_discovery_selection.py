import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "text/recurrent_scenarios")

import scenario_pipeline as pipeline


def test_aggregate_resampling_stability_mean_and_observability():
    rows = []
    for repetition in range(4):
        for cluster_label, n_ref, jaccard in (
            (0, 5 if repetition < 3 else 0, 0.8 if repetition < 3 else 0.0),
            (1, 4, 0.5),
        ):
            rows.append(
                {
                    "role": "A0",
                    "configuration_id": "A0_cfg_001",
                    "repetition": repetition,
                    "cluster_label": cluster_label,
                    "n_reference_units": n_ref,
                    "best_jaccard": jaccard,
                }
            )
    theme_frame = pd.DataFrame(rows)
    theme_out, summary = pipeline._aggregate_resampling_stability(theme_frame, n_repetitions=4)
    assert np.isclose(summary.loc[0, "stability"], (0.8 + 0.5) / 2.0)
    factor0 = theme_out.drop_duplicates(["configuration_id", "cluster_label"])
    factor0 = factor0.set_index("cluster_label")
    assert np.isclose(factor0.loc[0, "theme_stability"], 0.8)
    assert np.isclose(factor0.loc[0, "observability"], 0.75)
    assert np.isclose(factor0.loc[1, "observability"], 1.0)


def test_select_configuration_by_stability_tie_break_dbcv():
    merged = pd.DataFrame(
        [
            {"configuration_id": "A0_cfg_001", "stability": 0.90, "dbcv_umap": 0.40},
            {"configuration_id": "A0_cfg_002", "stability": 0.91, "dbcv_umap": 0.10},
            {"configuration_id": "A0_cfg_003", "stability": 0.91, "dbcv_umap": 0.55},
        ]
    )
    table, selected = pipeline.select_configuration_by_stability(merged)
    assert selected == "A0_cfg_003"
    assert bool(table.loc[table["configuration_id"].eq(selected), "selected"].iloc[0])


def test_materialize_and_load_selected_configurations(tmp_path=None):
    with tempfile.TemporaryDirectory(dir="text") as temporary_directory:
        root = Path(temporary_directory)
        run_dir = root / "run"
        role = "A0"
        candidate_dir = run_dir / "discovery" / role / "candidate_partitions"
        candidate_dir.mkdir(parents=True)
        labels = np.array([0, 0, 1, -1], dtype=np.int32)
        strengths = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
        np.save(candidate_dir / "A0_cfg_001_labels.npy", labels)
        np.save(candidate_dir / "A0_cfg_001_membership_strength.npy", strengths)
        units = pd.DataFrame(
            {
                "_accident_id": ["1", "1", "2", "3"],
                "_fact_id": ["f1", "f2", "f3", "f4"],
                "_role": [role] * 4,
                "_text": ["a", "b", "c", "d"],
            }
        )
        theme = pd.DataFrame(
            [
                {
                    "role": role,
                    "configuration_id": "A0_cfg_001",
                    "cluster_label": 0,
                    "theme_stability": 0.9,
                    "observability": 1.0,
                    "n_reference_units": 2,
                    "best_jaccard": 0.9,
                    "repetition": 0,
                }
            ]
        )
        pipeline.materialize_selected_partition(role, units, "A0_cfg_001", run_dir, theme)
        selected_dir = run_dir / "discovery" / role / "selected"
        assert (selected_dir / "labels.npy").is_file()
        assert (selected_dir / "topic_assignments.csv").is_file()
        pd.DataFrame(
            [{"role": role, "configuration_id": "A0_cfg_001"}]
            + [{"role": other, "configuration_id": f"{other}_cfg_001"} for other in ("A1", "B", "C")]
        ).to_csv(run_dir / "selected_configurations.csv", index=False)
        loaded = pipeline.load_selected_configurations(run_dir)
        assert loaded[role] == "A0_cfg_001"


def test_frozen_inputs_read_discovery_selected_paths():
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
                    "topic_id": f"{role}_000",
                    "role": role,
                    "configuration_id": f"{role}_cfg_001",
                    "llm_label": f"label-{role}",
                }
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
