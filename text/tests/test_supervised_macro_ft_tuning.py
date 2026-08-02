"""Tests tuning / variantes supervised_macro_ft (grille 8 combos)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from contrastive_methods.config import merge_config_dict
from supervised_macro_ft.tuning import (
    _combo_id,
    _merge_overrides,
    apply_full_encoder_training_overrides,
    build_variants_results_summary,
    encoder_scope_label,
    expand_grid,
    filter_macro_ft_tuning_combos,
    is_valid_macro_ft_tuning_model_cfg,
    projector_label,
    validate_macro_ft_grid_keys,
)


def test_validate_macro_ft_grid_keys_accepts_model_training():
    validate_macro_ft_grid_keys(
        {
            "model.projection": ["mlp_sklearn", None],
            "model.train_last_n_layers": [1, 2, 3, None],
        }
    )


def test_validate_macro_ft_grid_keys_rejects_unknown():
    try:
        validate_macro_ft_grid_keys({"data.batch_size": [8]})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "data.batch_size" in str(exc)


def test_expand_grid_cartesian():
    grid = {
        "model.projection": ["mlp_sklearn", None],
        "model.train_last_n_layers": [1, 3],
    }
    combos = expand_grid(grid)
    assert len(combos) == 4


def test_merge_overrides_dotted_keys():
    base = {"model": {"projection": "mlp_sklearn", "hiddim": 128}, "training": {"lr_head": 1e-3}}
    merged = _merge_overrides(base, {"model.projection": None, "model.train_last_n_layers": 2})
    assert merged["model"]["projection"] is None
    assert merged["model"]["train_last_n_layers"] == 2
    assert merged["training"]["lr_head"] == 1e-3


def test_full_encoder_training_overrides_apply_only_to_full_encoder():
    base = {
        "model": {"backbone_trainable": True},
        "training": {"epochs": 30, "use_amp": True, "lr_backbone": 2e-5},
    }
    stability_overrides = {
        "training.epochs": 3,
        "training.use_amp": False,
        "training.lr_backbone": 2e-6,
    }

    full = apply_full_encoder_training_overrides(
        {"model.train_last_n_layers": None}, base, stability_overrides
    )
    partial = apply_full_encoder_training_overrides(
        {"model.train_last_n_layers": 3}, base, stability_overrides
    )

    assert _merge_overrides(base, full)["training"] == {
        "epochs": 3,
        "use_amp": False,
        "lr_backbone": 2e-6,
    }
    assert _merge_overrides(base, partial)["training"] == base["training"]


def test_combo_id_readable():
    cid = _combo_id({"model.projection": "mlp_sklearn", "model.train_last_n_layers": 3})
    assert "projection" in cid
    assert "mlp_sklearn" in cid
    assert len(cid.split("_")[-1]) == 8


def test_encoder_projector_labels():
    assert encoder_scope_label(1) == "Last 1 layer"
    assert encoder_scope_label(3) == "Last 3 layers"
    assert encoder_scope_label(None) == "Full encoder"
    assert projector_label("mlp_sklearn") == "Yes"
    assert projector_label(None) == "No"
    assert projector_label("null") == "No"


@patch("supervised_macro_ft.tuning.run_supervised_macro_ft_cv")
def test_run_combo_cv_selection_score_from_mean_balanced_accuracy(mock_cv):
    from supervised_macro_ft.tuning import _run_combo_cv

    mock_cv.return_value = {
        "mean_balanced_accuracy": 0.72,
        "std_balanced_accuracy": 0.03,
        "n_folds": 3,
    }
    base_cfg = merge_config_dict(
        {
            "model": {
                "projection": "mlp_sklearn",
                "hiddim": 128,
                "backbone_trainable": True,
                "class_weight": "balanced",
                "oversampling": False,
            },
            "training": {"seed": 42},
        },
        {},
    )
    row = _run_combo_cv(
        base_cfg,
        {"model.train_last_n_layers": 2},
        combo_output_dir=Path("/tmp/combo"),
        backbone_hidden=None,
        shared_cache_dir=Path("/tmp/cache"),
        n_folds=3,
        seed=42,
        selection_metric="balanced_accuracy",
    )
    assert row["selection_score"] == 0.72
    assert row["mean_balanced_accuracy"] == 0.72


def test_grid_yaml_loads():
    from safer_core.io import load_yaml

    spec = load_yaml(TEXT_ROOT / "configs/tuning/supervised_macro_ft_grid.yaml")
    base = load_yaml(TEXT_ROOT / spec["base_config"])
    assert spec["selection_metric"] == "balanced_accuracy"
    assert spec["n_folds"] == 3
    assert "mlp_sklearn" in spec["grid"]["model.projection"]
    assert None in spec["grid"]["model.projection"]
    assert spec["grid"]["model.backbone_trainable"] == [True]
    assert spec["full_encoder_training_overrides"] == {
        "training.epochs": 3,
        "training.use_amp": False,
        "training.lr_backbone": 2e-6,
    }
    raw = expand_grid(spec["grid"])
    combos = filter_macro_ft_tuning_combos(raw, base)
    assert len(raw) == 8
    assert len(combos) == 8
    assert "training.lr_head" not in spec["grid"]


def test_filter_rejects_frozen_backbone():
    assert not is_valid_macro_ft_tuning_model_cfg(
        {
            "projection": "mlp_sklearn",
            "hiddim": 128,
            "backbone_trainable": False,
            "train_last_n_layers": 3,
            "class_weight": "balanced",
            "oversampling": False,
        }
    )


def test_filter_rejects_linear_projection():
    assert not is_valid_macro_ft_tuning_model_cfg(
        {
            "projection": "linear",
            "backbone_trainable": True,
            "train_last_n_layers": 3,
            "class_weight": "balanced",
            "oversampling": False,
        }
    )


def test_filter_rejects_mlp_sklearn_with_wrong_hiddim():
    assert not is_valid_macro_ft_tuning_model_cfg(
        {
            "projection": "mlp_sklearn",
            "hiddim": 512,
            "backbone_trainable": True,
            "train_last_n_layers": 3,
            "class_weight": "balanced",
            "oversampling": False,
        }
    )


def test_filter_accepts_article_combo():
    assert is_valid_macro_ft_tuning_model_cfg(
        {
            "projection": "mlp_sklearn",
            "hiddim": 128,
            "backbone_trainable": True,
            "train_last_n_layers": 3,
            "class_weight": "balanced",
            "oversampling": False,
            "cache_backbone_embeddings": False,
        }
    )
    assert is_valid_macro_ft_tuning_model_cfg(
        {
            "projection": None,
            "backbone_trainable": True,
            "train_last_n_layers": None,
            "class_weight": "balanced",
            "oversampling": False,
            "cache_backbone_embeddings": False,
        }
    )


def test_build_variants_results_summary_orders_rows(tmp_path: Path):
    rows = [
        {
            "encoder_scope": "Full encoder",
            "projector": "No",
            "combo_id": "full_no",
            "mean_balanced_accuracy": 0.6,
            "std_balanced_accuracy": 0.01,
            "combo_output_dir": str(tmp_path / "a"),
            "ba_ood_metallurgie": 0.5,
            "ba_ood_caou": 0.4,
            "ba_ood_nicollin": 0.45,
        },
        {
            "encoder_scope": "Last 1 layer",
            "projector": "Yes",
            "combo_id": "l1_yes",
            "mean_balanced_accuracy": 0.7,
            "std_balanced_accuracy": 0.02,
            "combo_output_dir": str(tmp_path / "b"),
            "ba_ood_metallurgie": 0.55,
            "ba_ood_caou": 0.52,
            "ba_ood_nicollin": 0.5,
        },
    ]
    summary = build_variants_results_summary(
        rows, test_corpora=["metallurgie", "caou", "nicollin"]
    )
    assert list(summary["encoder_scope"]) == ["Last 1 layer", "Full encoder"]
    assert summary.iloc[0]["ba_ood_avg"] == pytest.approx( (0.55 + 0.52 + 0.5) / 3 )
    assert summary.iloc[0]["ba_ood_worst"] == 0.5


def test_run_group_kfold_cv_save_fold_checkpoints_default():
    import inspect

    from supervised_macro_ft.cv import run_group_kfold_cv

    param = inspect.signature(run_group_kfold_cv).parameters["save_fold_checkpoints"]
    assert param.default is False
