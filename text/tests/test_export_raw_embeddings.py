"""Tests export_raw_embeddings (method_name + output_dir)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))


def _make_mini_emb_csv(path: Path, n: int = 30, dim: int = 8) -> None:
    rng = np.random.default_rng(0)
    rows = {
        "doc_id": np.arange(1, n + 1),
        "accident_id": rng.integers(0, 5, size=n),
        "pred_label": rng.choice(["A0", "A1", "B", "C"], size=n),
        "pred_ok": [True] * n,
        "sentence": [f"s{i}" for i in range(n)],
    }
    for j in range(dim):
        rows[f"dim_{j}"] = rng.standard_normal(n)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_export_raw_embeddings_method_name(tmp_path, monkeypatch):
    data_csv = tmp_path / "data.csv"
    emb_csv = tmp_path / "emb.csv"
    _make_mini_emb_csv(emb_csv)
    meta = pd.read_csv(emb_csv)[["doc_id", "accident_id", "pred_label", "pred_ok", "sentence"]]
    meta.to_csv(data_csv, index=False)

    out_dir = tmp_path / "raw_test"
    monkeypatch.chdir(tmp_path)
    sys.argv = [
        "export_raw_embeddings",
        "--data_csv",
        str(data_csv),
        "--emb_csv",
        str(emb_csv),
        "--output_dir",
        str(out_dir),
        "--method_name",
        "Embedding brut (test métallurgie)",
        "--skip_npy",
    ]
    from scripts import export_raw_embeddings

    export_raw_embeddings.main()

    metrics_csv = out_dir / "metrics" / "metrics_geometry.csv"
    assert metrics_csv.is_file()
    df = pd.read_csv(metrics_csv)
    assert df.iloc[0]["method"] == "Embedding brut (test métallurgie)"
    assert "eta2_macro_balanced" in df.columns


def test_postprocess_script_mentions_raw_exports():
    sh = (TEXT_ROOT / "jobs" / "postprocess_scgm_text.sh").read_text(encoding="utf-8")
    assert "export_raw_embeddings.py" in sh
    assert "raw_embedding_test" in sh
    assert "eval_scgm_test_metrics.py" in sh
    assert "embeddings/test/" in sh
