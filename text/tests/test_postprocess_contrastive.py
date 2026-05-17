"""Tests postprocess_contrastive_results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.postprocess_contrastive_results import postprocess_contrastive_method


def test_postprocess_writes_metrics_geometry(tmp_path, monkeypatch):
    import safer_core.paths as paths_mod

    text_root = tmp_path / "text"
    (text_root / "configs" / "methods").mkdir(parents=True)
    (text_root / "dataset").mkdir(parents=True)
    (text_root / "legacy" / "contrastive_method_v0" / "batchTripplet").mkdir(parents=True)

    data_csv = text_root / "dataset" / "data.csv"
    n = 40
    rng = np.random.default_rng(0)
    meta = pd.DataFrame(
        {
            "doc_id": np.arange(1, n + 1),
            "accident_id": rng.integers(0, 10, size=n),
            "pred_label": rng.choice(["A0", "A1", "B", "C"], size=n),
            "pred_ok": True,
            "sentence": [f"s{i}" for i in range(n)],
        }
    )
    meta.to_csv(data_csv, index=False)

    dim = 4
    emb = pd.DataFrame({"doc_id": np.arange(1, n + 1)})
    for i in range(dim):
        emb[f"dim_{i}"] = rng.normal(size=n)
    grid_dir = (
        text_root
        / "legacy"
        / "contrastive_method_v0"
        / "batchTripplet"
        / "fnembeddings_grid"
    )
    grid_dir.mkdir(parents=True)
    emb.to_csv(grid_dir / "embeddings__pred_label__lr1e-05_bs16_ep2.csv", index=False)

    cfg = {
        "method_name": "batch_triplet",
        "dataset_path": "dataset/data.csv",
        "label_col": "pred_label",
        "output_dir": "resultats/batch_triplet",
    }
    (text_root / "configs" / "methods" / "batch_triplet.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8"
    )

    results = text_root / "resultats" / "batch_triplet"
    results.mkdir(parents=True)
    pd.DataFrame([{"combo_id": "lr1e-05_bs16_ep2", "selection_score_mean_test_delta_ratio": 0.5}]).to_csv(
        results / "grid_search_summary.csv", index=False
    )

    monkeypatch.setattr(paths_mod, "TEXT_ROOT", text_root)
    monkeypatch.setattr(paths_mod, "RESULTS_ROOT", text_root / "resultats")
    paths_mod.METHOD_RESULTS_DIRS["batch_triplet"] = results

    dest = postprocess_contrastive_method("batch_triplet")
    assert dest.is_file()
    metrics_csv = results / "metrics" / "metrics_geometry.csv"
    assert metrics_csv.is_file()
    mdf = pd.read_csv(metrics_csv)
    assert "eta2_macro_balanced" in mdf.columns
    assert mdf.iloc[0]["method"] == "Batch Triplet"
    assert (results / "configs" / "config_resolved.yaml").is_file()
