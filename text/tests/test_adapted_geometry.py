from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_adapted_geometry_run import aligned_embeddings, geometry
from prepare_adapted_geometry_campaign import build_recipe


def test_recipe_has_all_adapted_conditions_and_keeps_ce_source_embeddings():
    recipe = build_recipe()
    assert len(recipe["models"]) == 18
    assert recipe["storage"]["keep_embeddings"] is True
    assert recipe["storage"]["cleanup_after_completion"] is True
    for model_id, spec in recipe["models"].items():
        assert spec["source_corpus"] in {"btp", "metallurgie"}
        if model_id.endswith("cross_entropy"):
            assert spec["overrides"]["exports"]["save_btp_embeddings"] is True


def test_reduced_recipe_is_isolated_and_has_only_three_methods():
    recipe = build_recipe("qwen3_btp")
    assert recipe["output_root"] == "output/adapted_geometry_pilot/runs"
    assert set(recipe["models"]) == {
        "qwen3_btp_cross_entropy", "qwen3_btp_supcon", "qwen3_btp_softtriple"}
    assert recipe["training"]["n_seeds"] == 5


def test_role_margin_sign_and_neighbour_entropy():
    source = np.array([[1, 0], [.9, .1], [0, 1], [.1, .9], [-1, 0], [-.9, .1],
                       [0, -1], [.1, -.9], [.8, .2], [.2, .8]], dtype=np.float32)
    roles = np.array(["A0", "A0", "A1", "A1", "B", "B", "C", "C", "A0", "A1"])
    target = np.array([[1, 0], [1, 0]], dtype=np.float32)
    answer = geometry(source, roles, target, np.array(["A0", "A1"]), k=3, block=1)
    assert answer.role_margin.iloc[0] > 0
    assert answer.role_margin.iloc[1] < 0
    assert answer.source_label_entropy_k10.iloc[0] == pytest.approx(0)
    assert answer.source_distance_k10.iloc[0] == pytest.approx(answer.source_distance_k10.iloc[1])


def test_embedding_alignment_rejects_wrong_unit(tmp_path: Path):
    run = tmp_path / "run"
    folder = run / "embeddings"
    folder.mkdir(parents=True)
    np.save(folder / "projected_btp.npy", np.ones((2, 3), dtype=np.float32))
    pd.DataFrame({"accident_id": ["a", "b"], "doc_id": ["x", "wrong"],
                  "pred_label": ["A0", "A1"], "sentence": ["one", "two"]}).to_csv(
                      folder / "projected_btp_metadata.csv", index=False)
    expected = pd.DataFrame({"accident_id": ["a", "b"], "fact_id": ["x", "y"],
                             "pred_label": ["A0", "A1"], "sentence": ["one", "two"]}).set_index(
                                 ["accident_id", "fact_id"])
    with pytest.raises(ValueError, match="keys differ"):
        aligned_embeddings(run, "btp", expected)


def test_embedding_alignment_reorders_by_accident_and_fact(tmp_path: Path):
    run = tmp_path / "run"
    folder = run / "embeddings"
    folder.mkdir(parents=True)
    np.save(folder / "projected_btp.npy", np.array([[2., 0.], [1., 0.]], dtype=np.float32))
    pd.DataFrame({"accident_id": ["b", "a"], "doc_id": ["y", "x"],
                  "pred_label": ["A1", "A0"], "sentence": ["two", "one"]}).to_csv(
                      folder / "projected_btp_metadata.csv", index=False)
    expected = pd.DataFrame({"accident_id": ["a", "b"], "fact_id": ["x", "y"],
                             "pred_label": ["A0", "A1"], "sentence": ["one", "two"]}).set_index(
                                 ["accident_id", "fact_id"])
    result = aligned_embeddings(run, "btp", expected)
    np.testing.assert_array_equal(result, [[1., 0.], [2., 0.]])
