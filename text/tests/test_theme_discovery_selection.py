import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recurrent_scenarios"))

import pareto_knee_selection as pareto_knee
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


def test_identify_pareto_front_non_dominance():
    frame = pd.DataFrame(
        [
            {"configuration_id": "a", "stability": 0.90, "dbcv_umap": 0.10},
            {"configuration_id": "b", "stability": 0.80, "dbcv_umap": 0.30},
            {"configuration_id": "c", "stability": 0.70, "dbcv_umap": 0.20},
            {"configuration_id": "d", "stability": 0.85, "dbcv_umap": 0.25},
        ]
    )
    marked = pareto_knee.identify_pareto_front(frame)
    pareto_ids = set(marked.loc[marked["is_pareto"], "configuration_id"])
    assert pareto_ids == {"a", "b", "d"}


def test_select_single_pareto_without_knee():
    frame = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "A0_cfg_001", "stability": 0.9, "dbcv_umap": 0.4},
            {"role": "A0", "configuration_id": "A0_cfg_002", "stability": 0.7, "dbcv_umap": 0.1},
        ]
    )
    table, selected, rule = pipeline.select_configuration_for_role(frame)
    assert selected == "A0_cfg_001"
    assert rule == "single_pareto"
    assert bool(table.loc[table["configuration_id"].eq(selected), "is_selected_knee"].iloc[0])


def test_geometric_knee_selects_compromise():
    candidates = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "A0_cfg_001", "stability": 0.95, "dbcv_umap": 0.05},
            {"role": "A0", "configuration_id": "A0_cfg_002", "stability": 0.75, "dbcv_umap": 0.35},
            {"role": "A0", "configuration_id": "A0_cfg_003", "stability": 0.45, "dbcv_umap": 0.55},
            {"role": "A0", "configuration_id": "A0_cfg_004", "stability": 0.20, "dbcv_umap": 0.90},
        ]
    )
    table, selected, rule = pipeline.select_configuration_for_role(candidates)
    assert rule == "geometric_knee"
    assert selected == "A0_cfg_002"
    knee_row = table.loc[table["configuration_id"].eq(selected)].iloc[0]
    assert knee_row["knee_distance"] == pytest.approx(
        table.loc[table["is_pareto"], "knee_distance"].max()
    )


def test_knee_tie_break_by_stability():
    frame = pd.DataFrame(
        [
            {"role": "B", "configuration_id": "B_cfg_001", "stability": 0.80, "dbcv_umap": 0.40},
            {"role": "B", "configuration_id": "B_cfg_002", "stability": 0.90, "dbcv_umap": 0.20},
            {"role": "B", "configuration_id": "B_cfg_003", "stability": 0.70, "dbcv_umap": 0.60},
        ]
    )
    marked = pareto_knee.identify_pareto_front(frame)
    pareto = marked.loc[marked["is_pareto"]].copy()
    normalized = pareto_knee.normalize_pareto_objectives(pareto, role="B")
    with_knee = pareto_knee.compute_geometric_knee(normalized)
    with_knee["knee_distance"] = 0.25
    selected_id = pareto_knee._select_knee_from_pareto(with_knee)
    assert selected_id in {"B_cfg_001", "B_cfg_002", "B_cfg_003"}


def test_normalize_handles_constant_objective():
    frame = pd.DataFrame(
        [
            {"role": "C", "configuration_id": "C_cfg_001", "stability": 0.5, "dbcv_umap": 0.4},
            {"role": "C", "configuration_id": "C_cfg_002", "stability": 0.7, "dbcv_umap": 0.4},
        ]
    )
    with pytest.warns(UserWarning, match="DBCV is constant"):
        normalized = pareto_knee.normalize_pareto_objectives(frame, role="C")
    assert normalized["dbcv_normalized"].isna().all()


def test_roles_with_multi_point_pareto_front():
    tables = {
        "A0": pd.DataFrame({"is_pareto": [True, True, False]}),
        "B": pd.DataFrame({"is_pareto": [True, False, False]}),
    }
    assert pareto_knee.roles_with_multi_point_pareto_front(tables) == ("A0",)


def test_single_pareto_leaves_normalized_columns_nan():
    frame = pd.DataFrame(
        [
            {"role": "B", "configuration_id": "B_cfg_001", "stability": 0.9, "dbcv_umap": 0.4},
            {"role": "B", "configuration_id": "B_cfg_002", "stability": 0.7, "dbcv_umap": 0.1},
        ]
    )
    marked = pareto_knee.identify_pareto_front(frame)
    table, selected, rule = pareto_knee.select_knee_configuration(marked)
    assert rule == "single_pareto"
    row = table.loc[table["configuration_id"].eq(selected)].iloc[0]
    assert pd.isna(row["stability_normalized"])
    assert pd.isna(row["dbcv_normalized"])
    assert pd.isna(row["knee_distance"])


def test_projection_onto_reference_line():
    x_h, y_h = pareto_knee.project_knee_to_reference_line(0.7, 0.8)
    assert x_h + y_h == pytest.approx(1.0, abs=1e-12)


def test_selection_reproducible_under_row_shuffle():
    frame = pd.DataFrame(
        [
            {"role": "A1", "configuration_id": "A1_cfg_001", "stability": 0.95, "dbcv_umap": 0.10},
            {"role": "A1", "configuration_id": "A1_cfg_002", "stability": 0.60, "dbcv_umap": 0.50},
            {"role": "A1", "configuration_id": "A1_cfg_003", "stability": 0.25, "dbcv_umap": 0.90},
        ]
    )
    _, selected_a, _ = pipeline.select_configuration_for_role(frame)
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    _, selected_b, _ = pipeline.select_configuration_for_role(shuffled)
    assert selected_a == selected_b


def test_select_configuration_by_stability_legacy_alias():
    merged = pd.DataFrame(
        [
            {"configuration_id": "A0_cfg_001", "stability": 0.90, "dbcv_umap": 0.40},
            {"configuration_id": "A0_cfg_002", "stability": 0.91, "dbcv_umap": 0.10},
            {"configuration_id": "A0_cfg_003", "stability": 0.91, "dbcv_umap": 0.55},
        ]
    )
    table, selected = pipeline.select_configuration_by_stability(merged)
    assert selected
    assert bool(table.loc[table["configuration_id"].eq(selected), "is_selected_knee"].iloc[0])


def test_materialize_and_load_selected_configurations():
    with tempfile.TemporaryDirectory() as temporary_directory:
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


def test_parameter_plan_has_thirty_six_configurations():
    config = {
        "screening": {
            "umap": {
                "n_neighbors": [10, 20, 40],
                "n_components": [5, 10, 15],
                "min_dist": [0.0],
            },
            "hdbscan": {
                "min_cluster_size": [25, 50],
                "min_samples": [5, 10],
                "cluster_selection_method": ["leaf"],
            },
        }
    }
    assert len(pipeline.parameter_plan(config)) == 36


def test_resampling_tasks_use_fixed_primary_umap_seed():
    config = {"validation": {"random_state": 42, "n_resampling": 3, "resampling_fraction": 0.8}}
    role_units = pd.DataFrame({
        "_accident_id": [str(i) for i in range(10)],
        "_role": ["A0"] * 10,
    })
    candidates = pd.DataFrame([
        {
            "configuration_id": "A0_cfg_001",
            "role": "A0",
            "umap_n_neighbors": 10,
            "umap_n_components": 5,
            "umap_min_dist": 0.0,
            "hdbscan_min_cluster_size": 25,
            "hdbscan_min_samples": 5,
            "hdbscan_cluster_selection_method": "leaf",
        }
    ])
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        run_dir = root / "run"
        role_dir = run_dir / "discovery" / "A0" / "candidate_partitions"
        role_dir.mkdir(parents=True)
        np.save(role_dir / "A0_cfg_001_labels.npy", np.array([0, 0, 1, 1, -1, 0, 1, 1, 0, -1]))
        embeddings = np.random.default_rng(0).normal(size=(10, 4))
        captured_states: list[int] = []

        def fake_parallel(func, tasks, config, progress_label=""):
            del func, config, progress_label
            for task in tasks:
                captured_states.append(int(task["random_state"]))
            return [[] for _ in tasks]

        original_parallel = pipeline._parallel_map
        pipeline._parallel_map = fake_parallel
        try:
            pipeline.evaluate_resampling_stability(
                "A0",
                role_units,
                embeddings,
                config,
                run_dir,
                candidates,
                reestimate=True,
            )
        finally:
            pipeline._parallel_map = original_parallel
    assert captured_states
    assert set(captured_states) == {42}


def test_frozen_inputs_read_discovery_selected_paths():
    units = pd.DataFrame(
        {
            "_accident_id": [str(index) for index in range(16)],
            "_fact_id": [f"fact-{index}" for index in range(16)],
            "_role": [role for role in pipeline.ROLES for _ in range(4)],
            "_text": ["text"] * 16,
        }
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
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
