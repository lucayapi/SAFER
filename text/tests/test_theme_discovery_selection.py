import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recurrent_scenarios"))

import pareto_knee_selection as pareto_knee
import manuscript_reporting as reporting
import scenario_pipeline as pipeline


def test_manuscript_role_palette_matches_latex_definitions():
    assert reporting.ROLE_NODE_FILL == {
        "A0": "#DCEAF7",
        "A1": "#FFF0C7",
        "B": "#FAD7C5",
        "C": "#EBCDD2",
        "Z": "#E7E0EC",
    }
    assert reporting.ROLE_COLORS == {
        "A0": "#3F6F9F",
        "A1": "#B98920",
        "B": "#C65D32",
        "C": "#8F3446",
        "Z": "#CAB2D6",
    }


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
    assert bool(table.loc[table["configuration_id"].eq(selected), "is_selected_tchebycheff"].iloc[0])


def test_normalized_tchebycheff_selects_compromise():
    candidates = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "A0_cfg_001", "stability": 0.95, "dbcv_umap": 0.05},
            {"role": "A0", "configuration_id": "A0_cfg_002", "stability": 0.75, "dbcv_umap": 0.35},
            {"role": "A0", "configuration_id": "A0_cfg_003", "stability": 0.45, "dbcv_umap": 0.55},
            {"role": "A0", "configuration_id": "A0_cfg_004", "stability": 0.20, "dbcv_umap": 0.90},
        ]
    )
    table, selected, rule = pipeline.select_configuration_for_role(candidates)
    assert rule == "normalized_tchebycheff"
    assert selected == "A0_cfg_002"
    selected_row = table.loc[table["configuration_id"].eq(selected)].iloc[0]
    assert selected_row["tchebycheff_max_shortfall"] == pytest.approx(
        table.loc[table["is_pareto"], "tchebycheff_max_shortfall"].min()
    )
    assert selected_row["selection_tie_break"] == "not_required"


def test_tchebycheff_tie_break_uses_total_shortfall():
    frame = pd.DataFrame(
        [
            {"role": "B", "configuration_id": "B_cfg_000", "stability": 1.00, "dbcv_umap": 0.00},
            {"role": "B", "configuration_id": "B_cfg_001", "stability": 0.90, "dbcv_umap": 0.50},
            {"role": "B", "configuration_id": "B_cfg_002", "stability": 0.50, "dbcv_umap": 0.80},
            {"role": "B", "configuration_id": "B_cfg_003", "stability": 0.00, "dbcv_umap": 1.00},
        ]
    )
    table, selected_id, rule = pareto_knee.select_tchebycheff_configuration(frame)
    assert rule == "normalized_tchebycheff"
    assert selected_id == "B_cfg_001"
    row = table.loc[table["configuration_id"].eq(selected_id)].iloc[0]
    assert row["selection_tie_break"] == "total_normalized_shortfall"
    assert not table["is_stability_tie_break_candidate"].any()


def test_tchebycheff_tie_break_uses_max_stability_before_configuration_order():
    frame = pd.DataFrame(
        [
            {"role": "B", "configuration_id": "B_cfg_010", "stability": 0.8, "dbcv_umap": 0.2},
            {"role": "B", "configuration_id": "B_cfg_002", "stability": 0.2, "dbcv_umap": 0.8},
        ]
    )
    table, selected_id, rule = pareto_knee.select_tchebycheff_configuration(frame)
    assert rule == "normalized_tchebycheff"
    assert selected_id == "B_cfg_010"
    assert table["is_stability_tie_break_candidate"].all()
    row = table.loc[table["configuration_id"].eq(selected_id)].iloc[0]
    assert row["selection_tie_break"] == "max_stability"


def test_normalize_handles_constant_objective():
    frame = pd.DataFrame(
        [
            {"role": "C", "configuration_id": "C_cfg_001", "stability": 0.5, "dbcv_umap": 0.4},
            {"role": "C", "configuration_id": "C_cfg_002", "stability": 0.7, "dbcv_umap": 0.4},
        ]
    )
    with pytest.warns(UserWarning, match="DBCV is constant"):
        normalized = pareto_knee.normalize_pareto_objectives(frame, role="C")
    assert normalized["dbcv_normalized"].eq(1.0).all()


def test_roles_with_multi_point_pareto_front():
    tables = {
        "A0": pd.DataFrame({"is_pareto": [True, True, False]}),
        "B": pd.DataFrame({"is_pareto": [True, False, False]}),
    }
    assert pareto_knee.roles_with_multi_point_pareto_front(tables) == ("A0",)


def test_pareto_figures_render_with_editable_labels():
    candidates = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "A0_cfg_000", "stability": 0.95, "dbcv_umap": 0.10},
            {"role": "A0", "configuration_id": "A0_cfg_001", "stability": 0.72, "dbcv_umap": 0.65},
            {"role": "A0", "configuration_id": "A0_cfg_002", "stability": 0.40, "dbcv_umap": 0.90},
            {"role": "A0", "configuration_id": "A0_cfg_003", "stability": 0.30, "dbcv_umap": 0.20},
        ]
    )
    table, selected, _ = pareto_knee.select_tchebycheff_configuration(candidates)
    assert selected == "A0_cfg_001"
    tables = {"A0": table}
    labels = {"dbcv_raw": "Structural validity", "stability_raw": "Reproducibility"}
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory)
        pareto_knee.plot_pareto_raw(
            tables,
            output_dir,
            roles=("A0",),
            filename="raw.png",
            role_labels={"A0": "Antecedents"},
            axis_labels=labels,
        )
        pareto_knee.plot_pareto_normalized_tchebycheff(
            tables,
            output_dir,
            roles=("A0",),
            filename="normalized.png",
            role_labels={"A0": "Antecedents"},
        )
        assert (output_dir / "raw.png").is_file()
        assert (output_dir / "normalized.png").is_file()


def test_raw_pareto_uses_common_axes_and_reproducibility_label(monkeypatch):
    tables = {}
    for role, offset in (("A0", 0.0), ("A1", 0.1)):
        candidates = pd.DataFrame(
            [
                {"role": role, "configuration_id": f"{role}_cfg_000", "stability": 0.8 - offset, "dbcv_umap": 0.1},
                {"role": role, "configuration_id": f"{role}_cfg_001", "stability": 0.5 - offset, "dbcv_umap": 0.6},
            ]
        )
        tables[role], _, _ = pareto_knee.select_tchebycheff_configuration(candidates)

    captured = []
    monkeypatch.setattr(reporting, "save_manuscript_figure", lambda figure, *_args, **_kwargs: captured.append(figure))
    with tempfile.TemporaryDirectory() as temporary_directory:
        pareto_knee.plot_pareto_raw(
            tables,
            Path(temporary_directory),
            roles=("A0", "A1"),
        )
    axes = captured[0].axes[:2]
    assert axes[0].get_xlim() == pytest.approx(axes[1].get_xlim())
    assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())
    assert axes[0].get_ylabel() == r"Resampling reproducibility $S_R$"
    assert "Pareto front" in [text.get_text() for text in captured[0].legends[0].get_texts()]


def test_factor_resampling_and_seed_figures_use_manuscript_labels():
    rows = []
    for role in ("A1", "B", "C"):
        for cluster_label, stability in ((0, 0.8), (1, 0.6)):
            for repetition, jaccard in enumerate((0.5, 0.7, 0.9)):
                rows.append({
                    "role": role,
                    "configuration_id": f"{role}_cfg_001",
                    "cluster_label": cluster_label,
                    "repetition": repetition,
                    "n_reference_units": 10,
                    "best_jaccard": jaccard - 0.05 * cluster_label,
                    "theme_stability": stability,
                })
    frame = pd.DataFrame(rows)
    figure = reporting.plot_factor_resampling_multi_panel(
        {role: frame.loc[frame["role"].eq(role)] for role in ("A1", "B", "C")},
        roles=("A1", "B", "C"),
        configuration_ids={role: f"{role}_cfg_001" for role in ("A1", "B", "C")},
    )
    assert figure is not None
    assert all(axis.get_xlim() == pytest.approx((0.0, 1.0)) for axis in figure.axes)
    assert all(axis.get_xlabel() == "" for axis in figure.axes)
    assert figure._supxlabel.get_text() == "Best-match Jaccard similarity"
    assert len(figure.legends) == 1
    single_figure = reporting.plot_factor_resampling_reproducibility(
        frame.loc[frame["role"].eq("A1")],
        role="A1",
        configuration_id="A1_cfg_001",
    )
    assert single_figure is not None
    assert single_figure.axes[0].get_title() == ""

    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory)
        for role in reporting.ROLES:
            seed_dir = run_dir / "discovery" / role / "seed_sensitivity"
            seed_dir.mkdir(parents=True)
            pd.DataFrame({
                "seed": [1, 2, 3],
                "seed_stability": [0.72, 0.78, 0.75],
            }).to_csv(seed_dir / "seed_summary.csv", index=False)
        seed_figure = reporting.plot_umap_seed_sensitivity_all_roles(run_dir)
    assert seed_figure is not None
    assert seed_figure._suptitle is None
    assert all(axis.get_ylim() == pytest.approx((0.0, 1.0)) for axis in seed_figure.axes)
    assert seed_figure.axes[0].get_ylabel() == "Mean best-match Jaccard similarity"


def test_single_pareto_skips_normalization_and_scalarization():
    frame = pd.DataFrame(
        [
            {"role": "B", "configuration_id": "B_cfg_001", "stability": 0.9, "dbcv_umap": 0.4},
            {"role": "B", "configuration_id": "B_cfg_002", "stability": 0.7, "dbcv_umap": 0.1},
        ]
    )
    marked = pareto_knee.identify_pareto_front(frame)
    table, selected, rule = pareto_knee.select_tchebycheff_configuration(marked)
    assert rule == "single_pareto"
    row = table.loc[table["configuration_id"].eq(selected)].iloc[0]
    assert pd.isna(row["stability_normalized"])
    assert pd.isna(row["dbcv_normalized"])
    assert pd.isna(row["tchebycheff_max_shortfall"])
    assert pd.isna(row["total_normalized_shortfall"])


def test_tchebycheff_scores_are_shortfalls_from_ideal():
    frame = pd.DataFrame({"dbcv_normalized": [0.7], "stability_normalized": [0.8]})
    scored = pareto_knee.compute_tchebycheff_scores(frame)
    assert scored.loc[0, "tchebycheff_max_shortfall"] == pytest.approx(0.3)
    assert scored.loc[0, "total_normalized_shortfall"] == pytest.approx(0.5)


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


def test_final_tie_break_uses_configuration_order_only_after_stability_tie():
    frame = pd.DataFrame(
        [
            {"role": "A1", "configuration_id": "A1_cfg_010", "stability": 0.8, "dbcv_umap": 0.4},
            {"role": "A1", "configuration_id": "A1_cfg_002", "stability": 0.8, "dbcv_umap": 0.4},
        ]
    )
    table, selected, rule = pareto_knee.select_tchebycheff_configuration(
        frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    )
    assert rule == "normalized_tchebycheff"
    assert selected == "A1_cfg_002"
    assert table["is_stability_tie_break_candidate"].all()
    row = table.loc[table["configuration_id"].eq(selected)].iloc[0]
    assert row["selection_tie_break"] == "configuration_order_after_stability"


def test_stability_tie_break_is_stable_under_row_shuffle():
    frame = pd.DataFrame(
        [
            {"role": "A1", "configuration_id": "A1_cfg_010", "stability": 0.8, "dbcv_umap": 0.2},
            {"role": "A1", "configuration_id": "A1_cfg_002", "stability": 0.2, "dbcv_umap": 0.8},
        ]
    )
    _, selected_a, _ = pareto_knee.select_tchebycheff_configuration(frame)
    _, selected_b, _ = pareto_knee.select_tchebycheff_configuration(
        frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    )
    assert selected_a == selected_b == "A1_cfg_010"


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
    assert bool(table.loc[table["configuration_id"].eq(selected), "is_selected_tchebycheff"].iloc[0])


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


def test_parameter_plan_has_seventy_two_configurations():
    config = {
        "screening": {
            "umap": {
                "n_neighbors": [10, 20, 40, 80],
                "n_components": [5, 10, 15],
                "min_dist": [0.0],
            },
            "hdbscan": {
                "min_cluster_size": [15, 25, 50],
                "min_samples": [5, 10],
                "cluster_selection_method": ["leaf"],
            },
        }
    }
    assert len(pipeline.parameter_plan(config)) == 72


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
        config = {
            "bayesian_networks": {
                "include_all_retained_factors": False,
                "min_theme_support_count": 2,
            }
        }
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
