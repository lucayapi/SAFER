"""Tests légers des réplications et de leur bootstrap, sans modèle HF."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from replication.bootstrap import analyze_replications
from replication.runner import load_replication_config, run_dir_for, task_matrix, training_seeds


def test_versioned_replication_recipe_generates_same_seed_list_for_all_models():
    config = load_replication_config(TEXT_ROOT / "output/replication_recipes/replication_config.yaml")
    seeds = training_seeds(config)
    assert len(seeds) == config["training"]["n_seeds"] == 5
    assert len(set(seeds)) == len(seeds)
    tasks = task_matrix(config)
    assert len(tasks) == 3 * len(seeds)
    assert [task["training_seed"] for task in tasks[:5]] == seeds
    assert [task["training_seed"] for task in tasks[5:10]] == seeds


def test_contrastive_split_seed_is_independent_from_training_seed():
    from contrastive_methods.config import ContrastiveConfig
    from contrastive_methods.data import get_group_kfold_splits
    from safer_core.kfold_eval import group_kfold_splits

    class Dataset:
        def get_groups(self):
            return ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]

    cfg = ContrastiveConfig(
        method_name="supcon", dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        seed=999, split_seed=42, n_folds=3,
    )
    actual = get_group_kfold_splits(Dataset(), cfg)
    expected = group_kfold_splits(Dataset().get_groups(), 3, 42)
    assert [(train.tolist(), val.tolist()) for train, val in actual] == [
        (train.tolist(), val.tolist()) for train, val in expected
    ]


def test_replication_results_notebook_is_structurally_valid():
    import nbformat

    from scripts.build_notebook_10_replication_uncertainty_results import build_notebook

    notebook = nbformat.from_dict(build_notebook())
    nbformat.validate(notebook)
    sources = "\n".join("".join(cell["source"]) for cell in notebook.cells)
    assert "seed_scores.csv" in sources
    assert "paired_bootstrap_differences.csv" in sources
    assert "accident_id" in sources


def test_bootstrap_is_paired_by_accident_and_averages_seeds(tmp_path: Path):
    config = {
        "output_root": str(tmp_path / "replications"),
        "training": {
            "n_seeds": 2,
            "seed_generator": 9,
            "split_seed": 42,
            "n_folds": 3,
            "test_corpora": ["target"],
        },
        "bootstrap": {
            "n_resamples": 100,
            "seed": 17,
            "confidence_level": 0.95,
            "metrics": ["balanced_accuracy", "macro_f1"],
        },
        "models": {"good": {"runner": "contrastive"}, "bad": {"runner": "contrastive"}},
    }
    truth = ["A0", "A1", "B", "C", "A0", "A1", "B", "C"]
    accidents = ["a", "a", "b", "b", "c", "c", "d", "d"]
    for model in config["models"]:
        for seed in training_seeds(config):
            frame = pd.DataFrame({
                "accident_id": accidents,
                "fact_id": list(range(8)),
                "true_macro": truth,
                "pred_macro": truth if model == "good" else ["C"] * len(truth),
            })
            folder = run_dir_for(config, model, seed) / "predictions"
            folder.mkdir(parents=True)
            frame.to_csv(folder / "predictions_target.csv", index=False)

    paths = analyze_replications(config, destination=tmp_path / "analysis")
    intervals = pd.read_csv(paths["bootstrap_ci"])
    differences = pd.read_csv(paths["paired_differences"])
    assert set(intervals["resampling_unit"]) == {"accident_id"}
    assert set(intervals["n_seeds"]) == {2}
    assert (differences["difference_a_minus_b"] > 0).all()
    assert differences["paired_on"].str.contains("same accident_id").all()
