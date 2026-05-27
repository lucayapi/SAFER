"""Reprise phase 1 TPN : chargement source/target_projected.npy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macro_transfer.tpn_pipeline import try_load_cached_projected_embeddings


def test_try_load_missing_files(tmp_path: Path) -> None:
    h_s, h_t, skipped = try_load_cached_projected_embeddings(
        tmp_path, n_source=10, n_target=5
    )
    assert h_s is None and h_t is None and skipped is False


def test_try_load_valid_cache(tmp_path: Path) -> None:
    emb = tmp_path / "embeddings"
    emb.mkdir()
    np.save(emb / "source_projected.npy", np.zeros((10, 4), dtype=np.float32))
    np.save(emb / "target_projected.npy", np.ones((5, 4), dtype=np.float32))
    h_s, h_t, skipped = try_load_cached_projected_embeddings(
        emb, n_source=10, n_target=5
    )
    assert skipped is True
    assert h_s is not None and h_t is not None
    assert h_s.shape == (10, 4) and h_t.shape == (5, 4)


def test_try_load_shape_mismatch(tmp_path: Path) -> None:
    emb = tmp_path
    np.save(emb / "source_projected.npy", np.zeros((9, 4)))
    np.save(emb / "target_projected.npy", np.zeros((5, 4)))
    h_s, h_t, skipped = try_load_cached_projected_embeddings(
        emb, n_source=10, n_target=5
    )
    assert skipped is False


def test_force_reencode_ignores_cache(tmp_path: Path) -> None:
    np.save(tmp_path / "source_projected.npy", np.zeros((10, 4)))
    np.save(tmp_path / "target_projected.npy", np.zeros((5, 4)))
    h_s, h_t, skipped = try_load_cached_projected_embeddings(
        tmp_path,
        n_source=10,
        n_target=5,
        force_reencode=True,
    )
    assert skipped is False
