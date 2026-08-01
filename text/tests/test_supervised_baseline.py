"""Tests légers pour macro_transfer.supervised_baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_transfer.constants import MACRO_NAMES
from safer_core.classification_metrics import build_gating_from_predictions
from macro_transfer.supervised_baseline import (
    aggregate_cv_metrics,
    build_classifier_pipeline,
    build_predictions_dataframe,
    load_ood_balanced_accuracy_by_corpus,
    merge_model_registry,
    run_model_group_kfold_cv,
    select_best_model,
    summarize_all_models_test_metrics,
    summarize_cross_domain_generalization,
    _fit_pipeline,
    _resample_train_for_class_weight,
)
from safer_core.kfold_eval import group_kfold_splits

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_cv_metrics_three_folds():
    rows = []
    for fold_id in range(3):
        rows.append(
            {
                "model": "logistic_regression",
                "fold_id": fold_id,
                "accuracy": 0.6 + 0.05 * fold_id,
                "macro_f1": 0.5 + 0.04 * fold_id,
                "balanced_accuracy": 0.55 + 0.03 * fold_id,
            }
        )
    summary = aggregate_cv_metrics(rows)
    assert len(summary) == 1
    assert summary.loc[0, "model"] == "logistic_regression"
    assert summary.loc[0, "n_folds"] == 3
    assert abs(summary.loc[0, "mean_accuracy"] - 0.65) < 1e-9
    assert summary.loc[0, "std_macro_f1"] > 0


def test_group_kfold_no_leakage():
    groups = np.array(["a1", "a1", "a2", "a2", "a3", "a3", "a4", "a4"])
    splits = group_kfold_splits(groups, n_splits=2, seed=42)
    for tr_idx, va_idx in splits:
        tr_groups = set(groups[tr_idx])
        va_groups = set(groups[va_idx])
        assert tr_groups.isdisjoint(va_groups)


def test_build_gating_from_classifier_probs():
    preds = pd.DataFrame(
        {
            "pred_macro": ["A0", "B"],
            "confidence": [0.9, 0.8],
            "prob_A0": [0.9, 0.1],
            "prob_A1": [0.05, 0.1],
            "prob_B": [0.03, 0.7],
            "prob_C": [0.02, 0.1],
        }
    )
    gating = build_gating_from_predictions(preds, MACRO_NAMES)
    assert list(gating["m_hat"]) == ["A0", "B"]
    assert float(gating.loc[0, "prob_A0"]) == pytest.approx(0.9)
    assert float(gating.loc[1, "prob_B"]) == pytest.approx(0.7)


def test_select_best_model_by_macro_f1():
    summary = pd.DataFrame(
        [
            {"model": "a", "mean_macro_f1": 0.4},
            {"model": "b", "mean_macro_f1": 0.55},
        ]
    )
    assert select_best_model(summary, selection_metric="macro_f1") == "b"


def test_select_best_model_accepts_mean_prefix():
    summary = pd.DataFrame(
        [
            {"model": "a", "mean_balanced_accuracy": 0.4},
            {"model": "b", "mean_balanced_accuracy": 0.55},
        ]
    )
    assert select_best_model(summary, selection_metric="mean_balanced_accuracy") == "b"


def test_build_classifier_pipeline_logistic():
    pipe = build_classifier_pipeline("logistic_regression", seed=0)
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_build_mlp_pipeline_class_weight_via_oversampling():
    pipe = build_classifier_pipeline("mlp", seed=0, params={"max_iter": 50, "hidden_layer_sizes": (8,)})
    assert getattr(pipe, "_mlp_class_weight", None) == "balanced"
    X = np.random.RandomState(0).randn(24, 6)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3] * 3, dtype=np.int64)
    X_fit, y_fit = _resample_train_for_class_weight(X, y, "balanced", seed=0)
    assert len(y_fit) == 4 * 12  # 4 classes × max count (12)
    assert len(np.unique(y_fit)) == 4
    _fit_pipeline(pipe, X, y, seed=0)
    assert hasattr(pipe.named_steps["clf"], "classes_")


def test_build_predictions_dataframe_columns():
    meta = pd.DataFrame(
        {
            "sentence": ["s1", "s2"],
            "pred_label": ["A0", "B"],
            "accident_id": [1, 2],
        }
    )
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.2]])
    preds = build_predictions_dataframe(
        meta,
        ["A0", "B"],
        probs,
        [0.7, 0.6],
        [0.6, 0.5],
        [0.5, 0.4],
        macros=MACRO_NAMES,
        method_name="test",
        text_col="sentence",
        group_col="accident_id",
        label_col="pred_label",
    )
    assert "prob_A0" in preds.columns
    assert "true_macro" in preds.columns
    assert len(preds) == 2


def test_run_model_group_kfold_cv_tiny_synthetic():
    rng = np.random.RandomState(0)
    n = 40
    X = rng.randn(n, 8)
    macros = list(MACRO_NAMES)
    y = np.array([macros[i % 4] for i in range(n)], dtype=object)
    groups = np.array([f"g{i // 4}" for i in range(n)], dtype=object)
    rows = run_model_group_kfold_cv(
        "logistic_regression",
        X,
        y,
        groups,
        macros=macros,
        n_folds=2,
        seed=0,
        params={"max_iter": 200},
    )
    assert len(rows) == 2
    assert "macro_f1" in rows[0]


def test_merge_model_registry_overrides():
    reg = merge_model_registry({"logistic_regression": {"params": {"C": 0.5}}})
    assert reg["logistic_regression"]["params"]["C"] == 0.5
    assert reg["logistic_regression"]["params"]["max_iter"] == 2000


def test_supervised_cache_roundtrip(tmp_path):
    from macro_transfer.supervised_baseline import (
        export_cv_results,
        export_test_results,
        load_cached_cv_results,
        load_cached_fold_rows_for_model,
        load_cached_test_results,
        load_supervised_run_manifest,
        save_supervised_run_manifest,
        supervised_ml_artifacts_exist,
    )

    fold_rows = [
        {
            "model": "logistic_regression",
            "fold_id": 0,
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "balanced_accuracy": 0.45,
        }
    ]
    summary = aggregate_cv_metrics(fold_rows)
    export_cv_results(tmp_path, fold_rows, summary)

    preds = pd.DataFrame(
        {
            "pred_macro": ["A0", "B"],
            "confidence": [0.9, 0.8],
            "prob_A0": [0.9, 0.1],
            "prob_A1": [0.05, 0.1],
            "prob_B": [0.03, 0.7],
            "prob_C": [0.02, 0.1],
            "sentence": ["a", "b"],
        }
    )
    metrics = {
        "macro_f1": 0.5,
        "_confusion_matrix": np.zeros((4, 4), dtype=np.int64),
        "_classification_report": {"A0": {"precision": 1.0}},
    }
    export_test_results(tmp_path, preds, metrics, macros=MACRO_NAMES)
    save_supervised_run_manifest(
        tmp_path,
        best_model="logistic_regression",
        selection_metric="macro_f1",
        seed=0,
        n_folds=1,
        test_corpus="metallurgie",
    )
    assert supervised_ml_artifacts_exist(tmp_path)
    loaded_rows, loaded_summary = load_cached_cv_results(tmp_path)
    assert len(loaded_rows) == 1
    assert load_cached_fold_rows_for_model(tmp_path, "logistic_regression")[0]["model"] == "logistic_regression"
    loaded_preds, loaded_metrics = load_cached_test_results(tmp_path, macros=MACRO_NAMES)
    assert len(loaded_preds) == 2
    assert "_confusion_matrix" in loaded_metrics
    assert load_supervised_run_manifest(tmp_path)["best_model"] == "logistic_regression"


def test_test_corpus_merge_requires_doc_id(tmp_path):
    """CSV test sans doc_id : create_doc_id_if_missing avant merge."""
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    from scgm_text.utils_io import create_doc_id_if_missing

    meta_csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "accident_id": ["g1", "g2"],
            "sentence": ["a", "b"],
            "pred_label": ["A0", "B"],
        }
    ).to_csv(meta_csv, index=False)
    emb_csv = tmp_path / "emb.csv"
    pd.DataFrame(
        {
            "doc_id": [1, 2],
            "dim_0": [1.0, 0.0],
            "dim_1": [0.0, 1.0],
        }
    ).to_csv(emb_csv, index=False)

    meta = pd.read_csv(meta_csv)
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    with pytest.raises(KeyError):
        merge_metadata_with_embeddings(slim, str(emb_csv))

    meta = create_doc_id_if_missing(meta)
    slim = meta.drop(columns=[c for c in meta.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_cols = merge_metadata_with_embeddings(slim, str(emb_csv))
    assert len(merged) == 2
    assert dim_cols == ["dim_0", "dim_1"]


def test_merge_metadata_with_embeddings_non_strict(tmp_path, capsys):
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings

    meta_csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "pred_label": ["A0", "A1", "B"],
        }
    ).to_csv(meta_csv, index=False)
    emb_csv = tmp_path / "emb.csv"
    pd.DataFrame(
        {
            "doc_id": [1, 2],
            "dim_0": [1.0, 0.0],
            "dim_1": [0.0, 1.0],
        }
    ).to_csv(emb_csv, index=False)

    meta = pd.read_csv(meta_csv)
    merged, dim_cols = merge_metadata_with_embeddings(meta, str(emb_csv), strict=False)
    assert len(merged) == 2
    assert dim_cols == ["dim_0", "dim_1"]
    out = capsys.readouterr().out
    assert "Attention" in out


def test_merge_metadata_with_embeddings_corpus_hint_from_filename(tmp_path, capsys):
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings

    meta = pd.DataFrame({"doc_id": [1, 2], "pred_label": ["A0", "A1"]})
    emb_csv = tmp_path / "Qwen3-Embedding-0.6B_metallurgie.csv"
    pd.DataFrame({"doc_id": [1], "dim_0": [1.0]}).to_csv(emb_csv, index=False)

    merged, _ = merge_metadata_with_embeddings(meta, str(emb_csv), strict=False)
    assert len(merged) == 1
    out = capsys.readouterr().out
    assert "--corpus metallurgie" in out
    assert "CORPUS=metallurgie" in out


def test_summarize_all_models_test_metrics():
    summary = summarize_all_models_test_metrics(
        {
            "a": {"accuracy": 0.5, "macro_f1": 0.4, "balanced_accuracy": 0.45},
            "b": {"accuracy": 0.6, "macro_f1": 0.55, "balanced_accuracy": 0.5},
        }
    )
    assert list(summary["model"]) == ["a", "b"]
    assert summary.loc[1, "macro_f1"] == pytest.approx(0.55)


def test_load_supervised_datasets_target_paths_differ_by_corpus():
    """Chaque corpus test doit charger son propre data_csv / emb_csv (pas un TARGET_CFG figé)."""
    from macro_transfer.supervised_baseline import load_supervised_datasets
    from safer_core.test_corpus import resolve_test_corpus

    anchor = Path(__file__).resolve().parents[1]
    source = {
        "dataset_path": "dataset/data_btp.csv",
        "emb_csv": "embeddings/Qwen3-Embedding-0.6B_btp.csv",
        "text_col": "sentence",
        "label_col": "pred_label",
        "group_col": "accident_id",
        "pred_ok_col": "pred_ok",
    }
    col_cfg = {
        "text_col": "sentence",
        "label_col": "pred_label",
        "group_col": "accident_id",
        "pred_ok_col": "pred_ok",
    }
    sizes: dict[str, int] = {}
    for corpus_id in ("metallurgie", "caou"):
        spec = resolve_test_corpus(
            corpus_id, anchor=anchor, require_files=True, require_emb_csv=True
        )
        cfg = {
            "corpus": corpus_id,
            "source": source,
            "target": {
                **col_cfg,
                "dataset_path": str(spec.data_csv.relative_to(anchor)).replace("\\", "/"),
                "emb_csv": str(spec.emb_csv.relative_to(anchor)).replace("\\", "/"),
            },
        }
        data = load_supervised_datasets(cfg, anchor=anchor)
        assert data["corpus_id"] == corpus_id
        sizes[corpus_id] = len(data["X_test"])
    assert sizes["metallurgie"] != sizes["caou"]


def test_summarize_cross_domain_generalization():
    cv_summary = pd.DataFrame(
        [
            {
                "model": "a",
                "mean_balanced_accuracy": 0.80,
                "std_balanced_accuracy": 0.04,
            },
            {
                "model": "b",
                "mean_balanced_accuracy": 0.70,
                "std_balanced_accuracy": 0.02,
            },
        ]
    )
    ood_ba_by_corpus = {
        "metallurgie": {"a": 0.60, "b": 0.50},
        "caou": {"a": 0.40, "b": 0.30},
    }
    summary = summarize_cross_domain_generalization(
        cv_summary, ood_ba_by_corpus, model_keys=["a", "b"]
    )
    assert list(summary["model"]) == ["a", "b"]
    assert summary.loc[0, "cv_ba_mean"] == pytest.approx(0.80)
    assert summary.loc[0, "cv_ba_std"] == pytest.approx(0.04)
    assert summary.loc[0, "ba_ood_avg"] == pytest.approx(0.50)
    assert summary.loc[0, "ba_ood_worst"] == pytest.approx(0.40)
    assert summary.loc[1, "ba_ood_avg"] == pytest.approx(0.40)
    assert summary.loc[1, "ba_ood_worst"] == pytest.approx(0.30)


def test_load_ood_balanced_accuracy_by_corpus(tmp_path):
    for corpus_id, ba_a, ba_b in [
        ("metallurgie", 0.55, 0.45),
        ("caou", 0.35, 0.25),
    ]:
        transfer = (
            tmp_path
            / "output_test"
            / corpus_id
            / "supervised_baseline"
            / "transfer"
        )
        transfer.mkdir(parents=True)
        pd.DataFrame(
            [
                {"model": "a", "balanced_accuracy": ba_a},
                {"model": "b", "balanced_accuracy": ba_b},
            ]
        ).to_csv(transfer / "all_models_test_metrics.csv", index=False)

    loaded = load_ood_balanced_accuracy_by_corpus(
        ["metallurgie", "caou"], ["a", "b"], anchor=tmp_path
    )
    assert loaded["metallurgie"]["a"] == pytest.approx(0.55)
    assert loaded["caou"]["b"] == pytest.approx(0.25)

    partial = load_ood_balanced_accuracy_by_corpus(
        ["metallurgie", "nicollin"],
        ["a", "b"],
        anchor=tmp_path,
        skip_missing=True,
    )
    assert set(partial) == {"metallurgie"}
    with pytest.raises(FileNotFoundError, match="nicollin"):
        load_ood_balanced_accuracy_by_corpus(
            ["metallurgie", "nicollin"], ["a", "b"], anchor=tmp_path
        )
