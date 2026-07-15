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
    macro_counts_per_z,
    macro_centroids_2d,
    plot_corpus_projections,
    plot_embeddings_csv_pca_tsne,
    plot_embeddings_csv_tsne_per_macro,
    plot_kfold_metrics_bars,
    plot_kfold_summary_errorbars,
    plot_kfold_val_curves,
    plot_projection_matplotlib,
    plot_topics_distribution_by_macro,
    plot_topics_n_units_by_z,
    plot_tsne_per_macro_grid,
)


def test_plot_kfold_metrics_bars(tmp_path):
    df = pd.DataFrame(
        {
            "fold_id": [0, 1, 2, 3, 4],
            "eta2_macro_balanced_perc": [10.0, 12.0, 11.0, 9.5, 10.5],
            "val_eta2_macro_balanced": [0.5, 0.52, 0.48, 0.51, 0.49],
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
                "val_eta2_macro_balanced_perc": [5.0 + fold_id, 6.0, 7.0],
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
        [{"mean_eta2_macro_balanced_perc": 10.0, "std_eta2_macro_balanced_perc": 1.0}]
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
            "n_A0": [60, 50, 0],
            "n_A1": [20, 0, 0],
            "n_B": [10, 0, 80],
            "n_C": [10, 0, 0],
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


def test_macro_counts_per_z():
    meta = pd.DataFrame(
        {
            "z_hat": [0, 0, 0, 1, 1, 2],
            "pred_label": ["A0", "A0", "B", "A1", "A1", "C"],
        }
    )
    out = macro_counts_per_z(meta, z_col="z_hat", label_col="pred_label")
    row0 = out[out["z_id"] == 0].iloc[0]
    assert int(row0["A0"]) == 2 and int(row0["B"]) == 1
    assert int(row0["n_total"]) == 3
    row1 = out[out["z_id"] == 1].iloc[0]
    assert int(row1["A1"]) == 2 and int(row1["n_total"]) == 2
    row2 = out[out["z_id"] == 2].iloc[0]
    assert int(row2["C"]) == 1 and int(row2["n_total"]) == 1


def test_plot_topics_n_units_by_z_with_metadata_merge(tmp_path):
    themes = pd.DataFrame(
        {
            "z_id": [0, 1],
            "dominant_macro": ["A0", "B"],
            "n_units": [4, 2],
        }
    )
    meta = pd.DataFrame(
        {
            "z_hat": [0, 0, 0, 0, 1, 1],
            "pred_label": ["A0", "A1", "A0", "B", "B", "C"],
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

    plot_topics_n_units_by_z(themes, metadata_df=meta, save_fig=_save)
    assert (fig_dir / "topics_n_units_by_z.png").is_file()


@patch("scgm_text.checkpoint_io.load_scgm_checkpoint")
@patch("scgm_text.eval_corpus.project_embedding_corpus")
@patch("scgm_text.eval_corpus.TextRawDataset")
def test_save_scgm_projected_corpus(mock_dataset_cls, mock_project, mock_load_ckpt, tmp_path):
    from scgm_text.eval_corpus import save_scgm_projected_corpus

    mock_load_ckpt.return_value = (None, {"input_mode": "precomputed_embeddings"}, None)
    meta = pd.DataFrame(
        {
            "doc_id": ["d0", "d1", "d2"],
            "accident_id": ["a0", "a1", "a2"],
            "pred_label": ["A0", "A1", "B"],
            "sentence": ["s0", "s1", "s2"],
        }
    )
    mock_project.return_value = (np.ones((3, 4), dtype=np.float32), meta)

    emb_dir = tmp_path / "embeddings"
    paths = save_scgm_projected_corpus(
        "fake.pt",
        "data.csv",
        emb_dir,
        stem="test",
    )
    assert paths["projections"].name == "projected_test.npy"
    assert paths["metadata"].name == "projected_test_metadata.csv"
    assert np.load(paths["projections"]).shape == (3, 4)
    assert paths["metadata"].is_file()


def test_plot_embeddings_csv_pca_tsne(tmp_path):
    n = 40
    labels = ["A0", "A1", "B", "C"] * (n // 4)
    meta = pd.DataFrame(
        {
            "doc_id": list(range(n)),
            "accident_id": [f"a{i // 4}" for i in range(n)],
            "pred_label": labels,
            "pred_ok": [True] * n,
        }
    )
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)

    emb = meta[["doc_id"]].copy()
    rng = np.random.default_rng(0)
    for j in range(8):
        emb[f"dim_{j}"] = rng.standard_normal(n)
    emb_path = tmp_path / "emb.csv"
    emb.to_csv(emb_path, index=False)

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_embeddings_csv_pca_tsne(
        emb_path,
        meta_path,
        "pred_label",
        corpus_name="test",
        save_fig=_save,
        png_name="emb_pca_tsne.png",
        max_points=n,
        seed=0,
    )
    assert out is not None
    assert out.is_file()
    assert plot_embeddings_csv_pca_tsne(tmp_path / "missing.csv", meta_path) is None


def test_plot_tsne_per_macro_grid(tmp_path):
    n = 40
    labels = np.array(["A0", "A1", "B", "C"] * (n // 4))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 8))
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_tsne_per_macro_grid(
        x,
        labels,
        corpus_name="test",
        save_fig=_save,
        png_name="tsne_pm.png",
        seed=0,
    )
    assert out is not None
    assert out.is_file()


def test_plot_tsne_per_macro_grid_sparse_macro(tmp_path):
    labels = np.array(["A0"] * 5 + ["A1"] * 40)
    x = np.random.default_rng(1).standard_normal((45, 4))
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_tsne_per_macro_grid(
        x,
        labels,
        save_fig=_save,
        png_name="sparse.png",
        min_points=10,
    )
    assert out is not None


def test_plot_embeddings_csv_tsne_per_macro(tmp_path):
    n = 40
    labels = ["A0", "A1", "B", "C"] * (n // 4)
    meta = pd.DataFrame(
        {
            "doc_id": list(range(n)),
            "accident_id": [f"a{i // 4}" for i in range(n)],
            "pred_label": labels,
            "pred_ok": [True] * n,
        }
    )
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)

    emb = meta[["doc_id"]].copy()
    rng = np.random.default_rng(0)
    for j in range(8):
        emb[f"dim_{j}"] = rng.standard_normal(n)
    emb_path = tmp_path / "emb.csv"
    emb.to_csv(emb_path, index=False)

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_embeddings_csv_tsne_per_macro(
        emb_path,
        meta_path,
        "pred_label",
        save_fig=_save,
        png_name="emb_tsne_pm.png",
        max_points=n,
        seed=0,
    )
    assert out is not None
    assert out.is_file()


def test_softtriple_centers_summary_and_projection(tmp_path):
    from scgm_text.notebook_viz import (
        plot_embeddings_csv_pca_tsne_with_softtriple_centers,
        softtriple_centers_summary_table,
    )

    rng = np.random.default_rng(0)
    d = 8
    n = 40
    labels = ["A0", "A1", "B", "C"] * (n // 4)
    meta = pd.DataFrame(
        {
            "doc_id": range(n),
            "accident_id": [f"a{i // 4}" for i in range(n)],
            "pred_label": labels,
            "pred_ok": [True] * n,
        }
    )
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)

    emb = meta.copy()
    emb[[f"dim_{i:04d}" for i in range(d)]] = rng.standard_normal((n, d))
    emb_path = tmp_path / "emb.csv"
    emb.to_csv(emb_path, index=False)

    centers_dir = tmp_path / "results" / "centers"
    centers_dir.mkdir(parents=True)
    center_rows = []
    for macro, cid in enumerate(["A0", "A1", "B", "C"]):
        vec = rng.standard_normal(d)
        row = {"class_id": cid, "class_name": macro, "effective_center_id": 0, "group_size": 3}
        for i, val in enumerate(vec):
            row[f"dim_{i:04d}"] = val
        center_rows.append(row)
    pd.DataFrame(center_rows).to_csv(centers_dir / "softtriple_effective_centers.csv", index=False)

    summary = softtriple_centers_summary_table(centers_dir / "softtriple_effective_centers.csv")
    assert len(summary) == 4
    assert "l2_norm" in summary.columns

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    out = plot_embeddings_csv_pca_tsne_with_softtriple_centers(
        emb_path,
        meta_path,
        "pred_label",
        results_dir=tmp_path / "results",
        save_fig=_save,
        png_name="st_centers.png",
        max_points=n,
        seed=0,
    )
    assert out is not None
    assert out.is_file()


def test_load_projected_embeddings_pair(tmp_path):
    from scgm_text.notebook_viz import load_projected_embeddings_pair

    npy_path = tmp_path / "projected_btp.npy"
    meta_path = tmp_path / "projected_btp_metadata.csv"
    np.save(npy_path, np.random.randn(4, 8))
    pd.DataFrame({"pred_label": ["A0", "A1", "B", "C"]}).to_csv(meta_path, index=False)
    pair = load_projected_embeddings_pair(npy_path, meta_path)
    assert pair is not None
    x, meta = pair
    assert x.shape == (4, 8)
    assert len(meta) == 4
    assert load_projected_embeddings_pair(tmp_path / "missing.npy", meta_path) is None


def test_plot_corpus_projections_without_plotly(tmp_path):
    n = 40
    rng = np.random.default_rng(0)
    projected = rng.standard_normal((n, 16))
    labels = (["A0", "A1", "B", "C"] * (n // 4 + 1))[:n]
    meta = pd.DataFrame({"pred_label": labels, "doc_id": [f"d{i}" for i in range(n)]})
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    def _save(name: str) -> Path:
        p = fig_dir / name
        import matplotlib.pyplot as plt

        plt.savefig(p, dpi=80)
        plt.close("all")
        return p

    paths = plot_corpus_projections(
        projected,
        meta,
        "pred_label",
        corpus_name="Test",
        save_fig=_save,
        figures_dir=fig_dir,
        png_name="proj.png",
        max_points=n,
        seed=0,
        include_plotly=False,
    )
    assert len(paths) == 1
    assert paths[0].suffix == ".png"
    assert not any(p.suffix == ".html" for p in paths)
    assert not list(fig_dir.glob("*.html"))
