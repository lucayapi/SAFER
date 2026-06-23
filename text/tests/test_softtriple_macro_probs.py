"""Tests assignation macro via centres SoftTriple natifs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from macro_transfer.softtriple_macro import (
    assign_macros_from_softtriple_centers,
    export_softtriple_source_centers,
    load_softtriple_centers,
    load_softtriple_hyperparams,
    summarize_center_weights,
)


def test_assign_macros_from_softtriple_centers_shapes_and_probs():
    n, c, k, d = 8, 4, 3, 6
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, d))
    centers = rng.normal(size=(c, k, d))
    macros = ["A0", "A1", "B", "C"]

    out = assign_macros_from_softtriple_centers(
        z,
        centers,
        macros,
        gamma=0.1,
        temperature=0.07,
        distance_metric="cosine",
    )

    probs = out["probs"]
    assert probs.shape == (n, c)
    assert np.allclose(probs.sum(axis=1), np.ones(n), atol=1e-5)
    assert len(out["pred_macro"]) == n
    assert out["gamma_jmk"].shape == (n, c, k)
    assert out["relaxed_scores"].shape == (n, c)
    assert out["distances"].shape == (n, c)
    assert np.allclose(out["distances"], -out["relaxed_scores"], atol=1e-6)

    top = probs.argmax(axis=1)
    assert (out["pred_macro"] == np.array(macros)[top]).all()


def test_export_softtriple_source_centers_k_rows_per_macro():
    centers = np.array(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ],
        dtype=np.float64,
    )
    macros = ["A0", "A1", "B", "C"]
    df = export_softtriple_source_centers(centers, macros)
    assert len(df) == 8
    assert set(df["macro"]) == set(macros)
    assert "center_k" in df.columns
    assert "dim_0000" in df.columns


def test_summarize_center_weights():
    gamma_jmk = np.ones((5, 4, 2), dtype=np.float64) * 0.5
    summary = summarize_center_weights(gamma_jmk, ["A0", "A1", "B", "C"])
    assert len(summary) == 8
    assert summary.loc[0, "mean_weight"] == pytest.approx(0.5)


def test_load_softtriple_checkpoint_helpers(tmp_path: Path):
    centers = torch.randn(4, 2, 8)
    ckpt = {
        "loss_state": {"centers": centers},
        "config": {"gamma": 0.2, "distance_metric": "euclidean", "centers_per_class": 2},
    }
    ckpt_dir = tmp_path / "best_model"
    ckpt_dir.mkdir()
    torch.save(ckpt, ckpt_dir / "softtriple_state.pt")
    (ckpt_dir / "effective_centers.json").write_text(
        '{"centers": [[[1.0, 0.0]]]}', encoding="utf-8"
    )

    loaded = load_softtriple_centers(ckpt_dir, prefer_raw_centers=True)
    assert loaded.shape == (4, 2, 8)

    hparams = load_softtriple_hyperparams(ckpt_dir)
    assert hparams["gamma"] == pytest.approx(0.2)
    assert hparams["centers_per_class"] == 2
