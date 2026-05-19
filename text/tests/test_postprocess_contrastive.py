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
    for i in range(1, dim + 1):
        emb[f"dim_{i:04d}"] = rng.normal(size=n)

    cfg = {
        "method_name": "batch_triplet",
        "output_dir": "resultats/batch_triplet",
        "data": {
            "dataset_path": "dataset/data.csv",
            "label_col": "pred_label",
        },
    }
    (text_root / "configs" / "methods" / "batch_triplet.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8"
    )

    results = text_root / "resultats" / "batch_triplet"
    emb_dir = results / "embeddings"
    emb_dir.mkdir(parents=True)
    emb.to_csv(emb_dir / "final_embeddings.csv", index=False)

    monkeypatch.setattr(paths_mod, "TEXT_ROOT", text_root)
    monkeypatch.setattr(paths_mod, "RESULTS_ROOT", text_root / "resultats")
    paths_mod.METHOD_RESULTS_DIRS["batch_triplet"] = results

    dest = postprocess_contrastive_method("batch_triplet")
    assert dest.is_file()
    metrics_csv = results / "metrics" / "metrics_geometry.csv"
    assert metrics_csv.is_file()
    mdf = pd.read_csv(metrics_csv)
    assert "eta2_macro_balanced" in mdf.columns
    assert "eta2_macro_balanced_perc" in mdf.columns
    assert mdf.iloc[0]["method"] == "Batch Triplet"
    assert (results / "configs" / "config_resolved.yaml").is_file()
