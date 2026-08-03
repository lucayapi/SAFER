"""Tests du tuning contrastif par architecture, sans entraînement HF."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.architecture_tuning import (
    _apply_full_overrides,
    _build_lr_summary,
    _run_logistic_group_cv,
    _merge_partial_summary,
    architecture_name,
    expand_grid,
    parse_variants,
)
from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.encoder_model import ContrastiveEncoder, EncoderConfig
from contrastive_methods.post_eval import fit_classifier_on_embeddings
from safer_core.io import load_yaml


def test_macro_ft_architecture_grids_have_exactly_eight_variants():
    for method in ("supcon", "softtriple", "batch_triplet"):
        spec = load_yaml(TEXT_ROOT / f"configs/tuning/{method}_macro_ft_grid.yaml")
        combos = expand_grid(spec["architecture_grid"])
        names = {
            architecture_name(combo["model.train_last_n_layers"], combo["model.use_projector"])
            for combo in combos
        }
        assert len(combos) == 8
        assert names == {
            f"{scope}_{projector}"
            for scope in ("last_1", "last_2", "last_3", "full")
            for projector in ("yes", "no")
        }
        assert spec["logistic_grid"]["C"] == [0.01, 0.1, 1.0, 10.0]


def test_full_overrides_only_apply_to_full_encoder():
    full = _apply_full_overrides(
        {"model.train_last_n_layers": None},
        {"training.epochs": 3, "training.use_amp": False, "training.learning_rate": 2e-6},
    )
    last = _apply_full_overrides(
        {"model.train_last_n_layers": 2},
        {"training.epochs": 3, "training.use_amp": False, "training.learning_rate": 2e-6},
    )
    assert full["training.epochs"] == 3
    assert full["training.use_amp"] is False
    assert full["training.learning_rate"] == 2e-6
    assert "training.epochs" not in last


def test_variant_selection_and_partial_summary_replacement(tmp_path):
    assert parse_variants(["full_yes", "last1_no"]) == {"full_yes", "last_1_no"}
    summary_path = tmp_path / "grid_summary.csv"
    pd.DataFrame([
        {"variant": "full_yes", "cv_ba_mean": 0.1},
        {"variant": "full_no", "cv_ba_mean": 0.2},
    ]).to_csv(summary_path, index=False)
    merged = _merge_partial_summary(summary_path, [{"variant": "full_yes", "cv_ba_mean": 0.9}])
    assert {row["variant"] for row in merged} == {"full_yes", "full_no"}
    assert next(row for row in merged if row["variant"] == "full_yes")["cv_ba_mean"] == 0.9


def test_logistic_grid_selects_best_balanced_accuracy_without_leakage():
    folds = [
        {
            "lr_0_val_balanced_accuracy": 0.50,
            "lr_1_val_balanced_accuracy": 0.80,
            "lr_0_val_accuracy": 0.50,
            "lr_1_val_accuracy": 0.80,
            "lr_0_val_macro_f1": 0.50,
            "lr_1_val_macro_f1": 0.80,
        },
        {
            "lr_0_val_balanced_accuracy": 0.60,
            "lr_1_val_balanced_accuracy": 0.70,
            "lr_0_val_accuracy": 0.60,
            "lr_1_val_accuracy": 0.70,
            "lr_0_val_macro_f1": 0.60,
            "lr_1_val_macro_f1": 0.70,
        },
    ]
    rows, best_index, best = _build_lr_summary(
        folds,
        [{"C": 0.01, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
         {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"}],
    )
    assert best_index == 1
    assert best["C"] == 1.0
    assert rows[1]["mean_val_balanced_accuracy"] == 0.75


def test_projector_dimensions_and_trainable_layer_scopes():
    projected = ContrastiveEncoder(EncoderConfig(
        backbone_name="__test_dummy__",
        backbone_trainable=True,
        train_last_n_layers=2,
        use_projector=True,
        projection="mlp_sklearn",
        hiddim=128,
    ))
    raw = ContrastiveEncoder(EncoderConfig(
        backbone_name="__test_dummy__",
        backbone_trainable=True,
        train_last_n_layers=None,
        use_projector=False,
    ))
    assert projected.embedding_dim == 128
    assert raw.embedding_dim == 32
    assert projected.backbone.count_trainable_transformer_layers() == 2
    assert raw.backbone.count_trainable_transformer_layers() == 4


def test_lr_overrides_reach_sklearn_classifier():
    import numpy as np

    rng = np.random.RandomState(3)
    X = np.vstack([rng.randn(12, 4) + i for i in range(4)])
    y = np.repeat(np.arange(4), 12)
    cfg = ContrastiveConfig(method_name="supcon", dataset_path=TEXT_ROOT / "dataset/data_btp.csv")
    pipe = fit_classifier_on_embeddings(
        X,
        y,
        cfg,
        classifier_overrides={"C": 0.01, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
    )
    assert pipe.named_steps["clf"].C == 0.01
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_n_folds_only_splits_precomputed_full_btp_embeddings():
    import numpy as np

    rng = np.random.RandomState(7)
    rows = []
    embeddings = []
    labels = ["A0", "A1", "B", "C"]
    for label_id, label in enumerate(labels):
        for group_id in range(8):
            for item_id in range(2):
                vector = rng.randn(6) * 0.05
                vector[label_id] += 3.0
                embeddings.append(vector)
                rows.append({
                    "label_id": label_id,
                    "pred_label": label,
                    "accident_id": f"{label}_{group_id}",
                    "sentence": f"{label}-{group_id}-{item_id}",
                })
    cfg = ContrastiveConfig(
        method_name="supcon",
        dataset_path=TEXT_ROOT / "dataset/data_btp.csv",
        label_col="pred_label",
        group_col="accident_id",
    )
    fold_rows = _run_logistic_group_cv(
        cfg,
        np.asarray(embeddings),
        pd.DataFrame(rows),
        [{"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"}],
        n_folds=4,
    )
    assert len(fold_rows) == 4
    assert all("lr_0_val_balanced_accuracy" in row for row in fold_rows)
