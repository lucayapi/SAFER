"""Tests légers notebook_viz (K-fold + export projections)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from scgm_text.notebook_viz import (
    macro_centroids_2d,
    plot_kfold_metrics_bars,
    plot_kfold_summary_errorbars,
    plot_kfold_val_curves,
    plot_projection_matplotlib,
    plot_topics_distribution_by_macro,
    plot_topics_n_units_by_z,
)


def test_plot_kfold_metrics_bars(tmp_path):
    df = pd.DataFrame(
        {
            "fold_id": [0, 1, 2, 3, 4],
            "delta_macro_pct": [10.0, 12.0, 11.0, 9.5, 10.5],
            "val_eta2_macro_balanced": [0.5, 0.52, 0.48, 0.51, 0.49],
            "rankme_global": [8.0, 8.2, 7.9, 8.1, 8.0],
            "c1_global": [0.3, 0.31, 0.29, 0.3, 0.3],
            "c10_global": [0.6, 0.61, 0.59, 0.6, 0.6],
        }
    )
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_kfold_metrics_bars(df, save_fig=_save)
    assert out is not None
    assert out.is_file()


def test_plot_kfold_val_curves(tmp_path):
    folds = tmp_path / "folds"
    for fold_id in range(2):
        mdir = folds / f"fold_{fold_id}" / "metrics"
        mdir.mkdir(parents=True)
        log = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "val_delta_macro_pct": [5.0 + fold_id, 6.0, 7.0],
                "val_eta2_macro_balanced": [0.4, 0.45, 0.5],
            }
        )
        log.to_csv(mdir / "train_log.csv", index=False)

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_kfold_val_curves(tmp_path, save_fig=_save)
    assert out is not None
    assert (fig_dir / "kfold_val_curves.png").is_file()


def test_plot_kfold_summary_errorbars(tmp_path):
    summary = pd.DataFrame(
        [{"mean_delta_macro_pct": 10.0, "std_delta_macro_pct": 1.0, "mean_rankme_global": 8.0, "std_rankme_global": 0.2}]
    )
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_kfold_summary_errorbars(summary, save_fig=_save)
    assert out is not None


def test_macro_centroids_2d():
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((40, 2))
    labels = np.array(["A0"] * 10 + ["A1"] * 10 + ["B"] * 10 + ["C"] * 10)
    cx, cy, names, colors = macro_centroids_2d(coords, labels)
    assert len(names) == 4
    assert len(cx) == 4
    assert "A0" in colors


def test_plot_projection_with_centroids(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    meta = pd.DataFrame(
        {
            "pred_label": rng.choice(["A0", "A1", "B", "C"], size=n),
            "z_hat": rng.integers(0, 5, size=n),
        }
    )
    pca_xy = rng.standard_normal((n, 2))
    tsne_xy = rng.standard_normal((n, 2))
    themes = pd.DataFrame(
        {"z_id": range(5), "dominant_macro": ["A0", "A1", "B", "C", "A0"], "n_units": [10] * 5}
    )
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_projection_matplotlib(
        pca_xy,
        tsne_xy,
        meta,
        "pred_label",
        save_fig=_save,
        png_name="proj_centroids.png",
        show_macro_centroids=True,
        show_z_centroids=True,
        themes_z=themes,
    )
    assert out.is_file()


def test_plot_topics_bars(tmp_path):
    themes = pd.DataFrame(
        {
            "z_id": [0, 1, 2],
            "dominant_macro": ["A0", "A0", "B"],
            "n_units": [100, 50, 80],
        }
    )
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    plot_topics_distribution_by_macro(themes, save_fig=_save)
    plot_topics_n_units_by_z(themes, save_fig=_save)
    assert (fig_dir / "topics_by_macro.png").is_file()
    assert (fig_dir / "topics_n_units_by_z.png").is_file()


@patch("scgm_text.eval_corpus.project_embedding_corpus")
@patch("scgm_text.eval_corpus.TextEmbeddingDataset")
def test_save_scgm_projected_corpus(mock_dataset_cls, mock_project, tmp_path):
    from scgm_text.eval_corpus import save_scgm_projected_corpus

    mock_project.return_value = (np.ones((3, 4), dtype=np.float32), np.array(["A0", "A1", "B"]))
    meta = pd.DataFrame(
        {
            "doc_id": ["d0", "d1", "d2"],
            "accident_id": ["a0", "a1", "a2"],
            "pred_label": ["A0", "A1", "B"],
        }
    )
    ds = mock_dataset_cls.return_value
    ds.get_metadata_df.return_value = meta

    emb_dir = tmp_path / "embeddings"
    paths = save_scgm_projected_corpus(
        "fake.pt",
        "data.csv",
        "emb.csv",
        emb_dir,
        stem="test",
    )
    assert paths["projections"].name == "projected_embeddings_test.npy"
    assert paths["metadata"].name == "test_metadata.csv"
    assert np.load(paths["projections"]).shape == (3, 4)
    assert paths["metadata"].is_file()
