"""Tests pour metrics.geometry_comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metrics.geometry_comparison import (
    build_geometry_comparison_table,
    compute_geometry_for_method_corpus,
    plot_geometry_comparison_bars,
)


def _orthogonal_dirs(n_dims: int = 8) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    return q


def _separated_embeddings(n_per_macro: int = 20, n_dims: int = 8) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    basis = _orthogonal_dirs(n_dims)
    rows, labels = [], []
    for mid, name in enumerate(["A0", "A1", "B", "C"]):
        for _ in range(n_per_macro):
            rows.append(basis[mid] * 5.0 + rng.normal(scale=0.05, size=n_dims))
            labels.append(name)
    return np.asarray(rows, dtype=np.float64), labels


def _write_projected_run(
    tmp_path: Path,
    corpus_id: str,
    x: np.ndarray,
    labels: list[str],
) -> Path:
    run_dir = tmp_path / "run"
    emb_dir = run_dir / "embeddings"
    emb_dir.mkdir(parents=True)
    np.save(emb_dir / f"projected_{corpus_id}.npy", x)
    meta = pd.DataFrame({"pred_label": labels, "sentence": [f"s{i}" for i in range(len(labels))]})
    meta.to_csv(emb_dir / f"projected_{corpus_id}_metadata.csv", index=False)
    return run_dir


def test_build_geometry_comparison_table_high_eta2(tmp_path, monkeypatch):
    x, labels = _separated_embeddings()

    def fake_raw(corpus_id, *, label_col="pred_label", anchor=None):
        assert corpus_id == "btp"
        return x, pd.DataFrame({label_col: labels})

    def fake_projected(results_dir, corpus_id, *, method_key=None, label_col="pred_label", anchor=None):
        assert corpus_id == "btp"
        return x, pd.DataFrame({label_col: labels})

    monkeypatch.setattr(
        "metrics.geometry_comparison.load_raw_qwen_embeddings",
        fake_raw,
    )
    monkeypatch.setattr(
        "metrics.geometry_comparison.load_projected_embeddings",
        fake_projected,
    )

    run_dir = _write_projected_run(tmp_path, "btp", x, labels)
    specs = [
        {"name": "Qwen brut", "kind": "raw"},
        {"name": "TestProj", "kind": "projected", "results_dir": run_dir},
    ]
    df = build_geometry_comparison_table(specs, ["btp"], skip_errors=False)
    assert len(df) == 2
    assert "eta2_macro_balanced_perc" in df.columns
    assert float(df["eta2_macro_balanced_perc"].min()) > 50.0


def test_compute_geometry_raw_kind(monkeypatch):
    x, labels = _separated_embeddings(n_per_macro=10)

    def fake_raw(corpus_id, *, label_col="pred_label", anchor=None):
        return x, pd.DataFrame({label_col: labels})

    monkeypatch.setattr(
        "metrics.geometry_comparison.load_raw_qwen_embeddings",
        fake_raw,
    )
    row = compute_geometry_for_method_corpus(
        {"name": "Qwen brut", "kind": "raw"},
        "metallurgie",
    )
    assert row["method"] == "Qwen brut"
    assert row["corpus"] == "metallurgie"
    assert row["eta2_macro_balanced_perc"] > 50.0


def test_compute_geometry_projected_kind(tmp_path):
    x, labels = _separated_embeddings(n_per_macro=10)
    run_dir = _write_projected_run(tmp_path, "caou", x, labels)
    row = compute_geometry_for_method_corpus(
        {"name": "SCGM", "kind": "projected", "results_dir": run_dir},
        "caou",
        anchor=tmp_path,
    )
    assert row["method"] == "SCGM"
    assert row["corpus"] == "caou"
    assert row["embedding_dim"] == x.shape[1]


def test_plot_geometry_comparison_bars_agg_backend(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    summary = pd.DataFrame(
        [
            {"method": "A", "corpus": "btp", "eta2_macro_balanced_perc": 80.0},
            {"method": "B", "corpus": "btp", "eta2_macro_balanced_perc": 60.0},
            {"method": "A", "corpus": "metallurgie", "eta2_macro_balanced_perc": 70.0},
        ]
    )
    out = plot_geometry_comparison_bars(
        summary,
        "btp",
        fig_dir=tmp_path,
        show=False,
    )
    assert out is not None
    assert out.is_file()


def test_skip_errors_omits_missing(monkeypatch, tmp_path):
    x, labels = _separated_embeddings(n_per_macro=5)

    def fake_raw(corpus_id, *, label_col="pred_label", anchor=None):
        return x, pd.DataFrame({label_col: labels})

    monkeypatch.setattr(
        "metrics.geometry_comparison.load_raw_qwen_embeddings",
        fake_raw,
    )

    specs = [
        {"name": "Qwen brut", "kind": "raw"},
        {
            "name": "Missing",
            "kind": "projected",
            "results_dir": tmp_path / "nonexistent",
        },
    ]
    with pytest.warns(UserWarning):
        df = build_geometry_comparison_table(specs, ["btp"], skip_errors=True)
    assert len(df) == 1
    assert df.iloc[0]["method"] == "Qwen brut"
