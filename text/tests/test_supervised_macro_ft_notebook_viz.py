"""Tests légers pour supervised_macro_ft.notebook_viz."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from supervised_macro_ft.notebook_viz import (
    build_prediction_df,
    discover_projected_corpora,
    export_metrics_latex_table,
    get_misclassification_sample,
    load_macro_ft_artifacts,
    load_raw_backbone_embeddings,
    plot_confusion_matrix_brand,
    plot_cv_metrics_bars,
    plot_raw_embeddings_pca_tsne,
    plot_raw_vs_projected_tsne_pair,
    plot_supervised_macro_ft_train_history,
    plot_tsne_true_vs_pred_brand,
    style_metrics_table,
    validate_results_dir,
)


def _minimal_run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "cv").mkdir(parents=True)
    (root / "metrics").mkdir(parents=True)
    (root / "embeddings").mkdir(parents=True)
    (root / "checkpoints" / "best_model").mkdir(parents=True)

    pd.DataFrame(
        {
            "fold": [0, 1],
            "balanced_accuracy": [0.72, 0.68],
            "macro_f1": [0.70, 0.66],
            "accuracy": [0.74, 0.69],
        }
    ).to_csv(root / "cv" / "cv_per_fold.csv", index=False)
    pd.DataFrame(
        {
            "model": ["supervised_macro_ft"],
            "mean_balanced_accuracy": [0.70],
            "std_balanced_accuracy": [0.02],
            "mean_macro_f1": [0.68],
            "std_macro_f1": [0.02],
        }
    ).to_csv(root / "cv" / "cv_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": ["supervised_macro_ft"],
            "cv_ba_mean": [0.70],
            "cv_ba_std": [0.02],
            "ba_ood_avg": [0.55],
            "ba_ood_worst": [0.50],
            "balanced_accuracy_metallurgie": [0.55],
            "balanced_accuracy_caou": [0.50],
        }
    ).to_csv(root / "metrics" / "cross_domain_generalization.csv", index=False)

    n = 40
    np.save(root / "embeddings" / "projected_btp.npy", np.random.randn(n, 8).astype(np.float64))
    pd.DataFrame(
        {
            "sentence": [f"s{i}" for i in range(n)],
            "pred_label": ["A0", "A1", "B", "C"] * (n // 4),
        }
    ).to_csv(root / "embeddings" / "projected_btp_metadata.csv", index=False)

    with open(root / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"method": "supervised_macro_ft", "n_folds": 2}, f)
    with open(root / "checkpoints" / "best_model" / "config.json", "w", encoding="utf-8") as f:
        json.dump({"backbone_name": "test", "n_classes": 4}, f)

    return root


def test_validate_and_load_macro_ft_artifacts(tmp_path: Path):
    root = _minimal_run_dir(tmp_path)
    validate_results_dir(root)
    art = load_macro_ft_artifacts(root)
    assert art.cv_per_fold is not None
    assert art.cv_summary is not None
    assert art.cross_domain is not None
    assert "btp" in art.projected_corpora
    assert art.checkpoint_dir is not None


def test_discover_projected_corpora(tmp_path: Path):
    root = _minimal_run_dir(tmp_path)
    stems = discover_projected_corpora(root)
    assert stems == ["btp"]


def test_style_metrics_table_and_latex():
    df = pd.DataFrame(
        {
            "phase": ["cv_val", "lr_eval"],
            "corpus": ["btp", "metallurgie"],
            "balanced_accuracy": ["0.800 ± 0.040", "0.550"],
            "macro_f1": ["0.750 ± 0.030", "0.480"],
        }
    )
    styler = style_metrics_table(df)
    assert styler is not None
    html = styler.to_html()
    assert "0.800 ± 0.040" in html
    latex = export_metrics_latex_table(df)
    assert "\\textbf" in latex
    assert "0.550" in latex
    assert "cv_val" in latex


def test_plot_cv_metrics_bars(tmp_path: Path):
    root = _minimal_run_dir(tmp_path)
    cv = pd.read_csv(root / "cv" / "cv_per_fold.csv")
    out = plot_cv_metrics_bars(cv, fig_dir=tmp_path / "fig", show=False)
    assert out is not None and out.is_file()


def test_plot_supervised_macro_ft_train_history(tmp_path: Path):
    hist = pd.DataFrame(
        {
            "phase": ["cv", "cv", "cv", "cv", "final", "final"],
            "fold": [0, 0, 1, 1, -1, -1],
            "epoch": [1, 2, 1, 2, 1, 2],
            "train_loss": [1.2, 0.9, 1.1, 0.8, 0.7, 0.5],
            "val_macro_f1": [0.4, 0.5, 0.45, 0.55, float("nan"), float("nan")],
            "val_balanced_accuracy": [0.42, 0.52, 0.44, 0.54, float("nan"), float("nan")],
        }
    )
    out = plot_supervised_macro_ft_train_history(
        hist,
        fig_dir=tmp_path / "fig",
        filename="train_history.png",
        show=False,
    )
    assert out is not None and out.is_file()


def test_plot_confusion_and_tsne_true_vs_pred(tmp_path: Path):
    n = 80
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, 4))
    df = pd.DataFrame(
        {
            "true_macro": ["A0", "A1", "B", "C"] * (n // 4),
            "pred_macro": ["A0", "A1", "B", "C"] * (n // 4),
        }
    )
    df.loc[0, "pred_macro"] = "A1"
    out_cm = plot_confusion_matrix_brand(
        df["true_macro"],
        df["pred_macro"],
        fig_dir=tmp_path / "fig",
        filename="cm.png",
        show=False,
    )
    assert out_cm is not None and out_cm.is_file()
    out_tsne = plot_tsne_true_vs_pred_brand(
        emb,
        df,
        "true_macro",
        "pred_macro",
        fig_dir=tmp_path / "fig",
        filename="tsne.png",
        max_points=60,
        show=False,
    )
    assert out_tsne is not None and out_tsne.is_file()


def test_plot_raw_embeddings_and_pair(tmp_path: Path):
    n = 60
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((n, 6))
    proj = rng.standard_normal((n, 4))
    meta = pd.DataFrame({"pred_label": ["A0", "A1", "B", "C"] * (n // 4)})
    out_raw = plot_raw_embeddings_pca_tsne(
        raw,
        meta,
        "pred_label",
        corpus_name="btp",
        fig_dir=tmp_path / "fig",
        filename="raw.png",
        max_points=40,
        show=False,
    )
    assert out_raw is not None and out_raw.is_file()
    out_pair = plot_raw_vs_projected_tsne_pair(
        raw,
        proj,
        meta,
        "pred_label",
        fig_dir=tmp_path / "fig",
        filename="pair.png",
        max_points=40,
        show=False,
    )
    assert out_pair is not None and out_pair.is_file()


def test_get_misclassification_sample():
    df = pd.DataFrame(
        {
            "sentence": ["a", "b", "c"],
            "true_macro": ["A0", "A1", "B"],
            "pred_macro": ["A0", "C", "B"],
            "margin": [0.9, 0.1, 0.8],
        }
    )
    err = get_misclassification_sample(df, n=5)
    assert len(err) == 1
    assert err.iloc[0]["true_macro"] == "A1"


def test_build_prediction_df_mocked(tmp_path: Path):
    meta = pd.DataFrame(
        {
            "sentence": ["phrase un", "phrase deux"],
            "pred_label": ["A0", "B"],
        }
    )
    mock_model = MagicMock()
    mock_model.tokenizer = MagicMock()
    pred = np.array(["A0", "B"], dtype=object)
    probs = np.array([[0.9, 0.05, 0.03, 0.02], [0.1, 0.1, 0.7, 0.1]])
    conf = np.array([0.9, 0.7])
    margin = np.array([0.85, 0.6])
    entropy = np.array([0.2, 0.5])

    with (
        patch("supervised_macro_ft.checkpoint_io.load_checkpoint", return_value=mock_model),
        patch("supervised_macro_ft.checkpoint_io.read_checkpoint_config", return_value={
            "max_seq_length": 64,
            "backbone_name": "test-model",
            "backbone_trainable": True,
        }),
        patch("supervised_macro_ft.notebook_viz._load_macro_ft_tokenizer", return_value=MagicMock()),
        patch(
            "supervised_macro_ft.inference.predict_corpus",
            return_value=(pred, probs, conf, margin, entropy),
        ),
    ):
        out = build_prediction_df(
            tmp_path / "ckpt",
            meta,
            meta["sentence"].tolist(),
            label_col="pred_label",
            device="cpu",
        )
    assert list(out["pred_macro"]) == ["A0", "B"]
    assert "confidence" in out.columns


def test_build_prediction_df_falls_back_when_emb_csv_stale(tmp_path: Path, monkeypatch):
    """Si le CSV Qwen est désaligné, predict_corpus est utilisé (pas d'erreur)."""
    meta = pd.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "sentence": ["a", "b", "c"],
            "pred_label": ["A0", "A1", "B"],
        }
    )
    emb_csv = tmp_path / "stale_emb.csv"
    pd.DataFrame({"doc_id": ["d1", "d2"], "dim_0": [0.1, 0.2], "dim_1": [0.3, 0.4]}).to_csv(
        emb_csv, index=False
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "backbone_name": "test-model",
                "backbone_trainable": False,
                "max_seq_length": 32,
                "n_classes": 4,
            }
        ),
        encoding="utf-8",
    )

    mock_model = MagicMock()
    pred = np.array(["A0", "A1", "B"], dtype=object)
    probs = np.eye(3, 4)
    conf = np.array([0.9, 0.8, 0.7])
    margin = np.array([0.8, 0.7, 0.6])
    entropy = np.array([0.2, 0.3, 0.4])

    monkeypatch.setattr(
        "supervised_macro_ft.notebook_viz._load_macro_ft_tokenizer",
        lambda _name: MagicMock(),
    )

    with (
        patch("supervised_macro_ft.checkpoint_io.load_checkpoint", return_value=mock_model),
        patch("supervised_macro_ft.checkpoint_io.read_checkpoint_config") as read_cfg,
        patch(
            "supervised_macro_ft.inference.predict_corpus",
            return_value=(pred, probs, conf, margin, entropy),
        ) as predict_corpus,
    ):
        read_cfg.return_value = {
            "backbone_name": "test-model",
            "backbone_trainable": False,
            "max_seq_length": 32,
        }
        out = build_prediction_df(
            ckpt,
            meta,
            meta["sentence"].tolist(),
            label_col="pred_label",
            device="cpu",
            results_dir=tmp_path / "run",
            corpus_id="btp",
            backbone_emb_csv=emb_csv,
        )
    predict_corpus.assert_called_once()
    assert list(out["pred_macro"]) == ["A0", "A1", "B"]


def test_load_raw_backbone_from_cache(tmp_path: Path):
    root = _minimal_run_dir(tmp_path)
    (root / "cache").mkdir(exist_ok=True)
    n = 20
    np.save(root / "cache" / "backbone_hidden.npy", np.random.randn(n, 16).astype(np.float32))
    hidden, meta, missing = load_raw_backbone_embeddings(root, "btp", anchor=tmp_path)
    assert hidden is not None
    assert meta is not None
    assert hidden.shape[0] == n
    assert not missing


def test_validate_results_dir_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_results_dir(tmp_path / "absent")

    bare = tmp_path / "empty"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_results_dir(bare)
