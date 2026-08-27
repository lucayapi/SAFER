import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "text/recurrent_scenarios")

import scenario_pipeline as pipeline
import semantic_evaluation as semantic


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


def test_mark_pareto_front_non_dominance():
    frame = pd.DataFrame(
        [
            {"configuration_id": "a", "stability": 0.90, "dbcv_umap": 0.10},
            {"configuration_id": "b", "stability": 0.80, "dbcv_umap": 0.30},
            {"configuration_id": "c", "stability": 0.70, "dbcv_umap": 0.20},
            {"configuration_id": "d", "stability": 0.85, "dbcv_umap": 0.25},
        ]
    )
    marked = semantic.mark_pareto_front(frame)
    pareto_ids = set(marked.loc[marked["on_pareto"], "configuration_id"])
    assert pareto_ids == {"a", "b", "d"}


def test_select_single_pareto_without_scores():
    frame = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "A0_cfg_001", "stability": 0.9, "dbcv_umap": 0.4},
            {"role": "A0", "configuration_id": "A0_cfg_002", "stability": 0.7, "dbcv_umap": 0.1},
        ]
    )
    table, selected, rule = pipeline.select_configuration_for_role(frame, pd.DataFrame())
    assert selected == "A0_cfg_001"
    assert rule == "single_pareto"
    assert bool(table.loc[table["configuration_id"].eq(selected), "selected"].iloc[0])


def test_select_semantic_score_with_sr_tie_break():
    candidates = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "A0_cfg_001", "stability": 0.80, "dbcv_umap": 0.40},
            {"role": "A0", "configuration_id": "A0_cfg_002", "stability": 0.90, "dbcv_umap": 0.20},
            {"role": "A0", "configuration_id": "A0_cfg_003", "stability": 0.70, "dbcv_umap": 0.10},
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "role": "A0",
                "configuration_id": "A0_cfg_001",
                "coherence": 4.0,
                "distinctiveness": 4.0,
                "prevention_relevance": 4.0,
                "semantic_score": 4.0,
                "evaluator_1_score": 4.0,
                "evaluator_2_score": 4.0,
            },
            {
                "role": "A0",
                "configuration_id": "A0_cfg_002",
                "coherence": 4.0,
                "distinctiveness": 4.0,
                "prevention_relevance": 4.0,
                "semantic_score": 4.0,
                "evaluator_1_score": 4.1,
                "evaluator_2_score": 3.9,
            },
        ]
    )
    table, selected, rule = pipeline.select_configuration_for_role(candidates, scores)
    assert selected == "A0_cfg_002"
    assert rule == "semantic_score"
    assert bool(table.loc[table["configuration_id"].eq(selected), "selected"].iloc[0])


def test_aggregate_and_agreement():
    factor_scores = pd.DataFrame(
        [
            {"role": "A0", "configuration_id": "c1", "evaluator_id": "evaluator_1", "coherence": 5, "distinctiveness": 4, "prevention_relevance": 3},
            {"role": "A0", "configuration_id": "c1", "evaluator_id": "evaluator_2", "coherence": 4, "distinctiveness": 4, "prevention_relevance": 4},
            {"role": "A0", "configuration_id": "c2", "evaluator_id": "evaluator_1", "coherence": 3, "distinctiveness": 3, "prevention_relevance": 3},
            {"role": "A0", "configuration_id": "c2", "evaluator_id": "evaluator_2", "coherence": 2, "distinctiveness": 2, "prevention_relevance": 2},
        ]
    )
    aggregated = semantic.aggregate_semantic_scores(factor_scores)
    assert set(aggregated["configuration_id"]) == {"c1", "c2"}
    c1 = aggregated.loc[aggregated["configuration_id"].eq("c1")].iloc[0]
    assert np.isclose(c1["semantic_score"], 4.0)
    agreement = semantic.compute_evaluator_agreement(factor_scores)
    assert bool(agreement.iloc[0]["same_top_ranked"]) is True


def test_parse_evaluator_response_and_mock_scoring():
    raw = json.dumps({
        "candidate_id": "Candidate-01",
        "factors": [
            {
                "factor_id": "Factor-01",
                "coherence": 4,
                "distinctiveness": 5,
                "prevention_relevance": 4,
                "justification": "ok",
            }
        ],
    })
    parsed = semantic.parse_evaluator_response(raw, expected_factor_ids=["Factor-01"])
    assert parsed[0]["coherence"] == 4.0

    packages = [
        {
            "configuration_id": "A0_cfg_001",
            "factors": [
                {"cluster_label": 0, "topic_id": "A0_000", "n_units": 3, "samples": [{"text": "chute"}]},
            ],
        },
        {
            "configuration_id": "A0_cfg_002",
            "factors": [
                {"cluster_label": 0, "topic_id": "A0_000", "n_units": 2, "samples": [{"text": "glissade"}]},
            ],
        },
    ]

    def fake_chat(*, model: str, prompt: str) -> str:
        score = 5 if "chute" in prompt else 2
        return json.dumps({
            "candidate_id": "Candidate-XX",
            "factors": [
                {
                    "factor_id": "Factor-01",
                    "coherence": score,
                    "distinctiveness": score,
                    "prevention_relevance": score,
                    "justification": model,
                }
            ],
        })

    scores = semantic.score_packages_with_evaluators(
        packages,
        role="A0",
        config={"validation": {"semantic_evaluation": {
            "random_state": 0,
            "evaluator_1": {"model": "gpt-5.4"},
            "evaluator_2": {"model": "gpt-5"},
        }}},
        chat_completion=fake_chat,
    )
    assert set(scores["evaluator_id"]) == {"evaluator_1", "evaluator_2"}
    aggregated = semantic.aggregate_semantic_scores(scores)
    assert aggregated.sort_values("semantic_score", ascending=False).iloc[0]["configuration_id"] == "A0_cfg_001"


def test_sample_membership_stratified_units_respects_budget():
    frame = pd.DataFrame({
        "sentence": [f"u{i}" for i in range(30)],
        "membership_strength": np.linspace(0.1, 1.0, 30),
        "accident_id": [str(i // 2) for i in range(30)],
    })
    samples = semantic.sample_membership_stratified_units(
        frame,
        units_per_factor=9,
        rng=np.random.default_rng(0),
    )
    assert len(samples) == 9
    assert {item["membership_stratum"] for item in samples} <= {"high", "mid", "low"}


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


def test_materialize_and_load_selected_configurations():
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
