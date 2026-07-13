"""Tests jobs exist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

TEXT_ROOT = Path(__file__).resolve().parents[1]
METHODS_CONFIG = TEXT_ROOT / "configs" / "methods"
JOBS_DIR = TEXT_ROOT / "jobs"


def test_paths_use_resultats_default():
    cfg = yaml.safe_load((METHODS_CONFIG / "scgm_text.yaml").read_text(encoding="utf-8"))
    flat = {**cfg.get("model", {}), **cfg}
    assert "output/" in str(flat.get("output_dir", ""))
    assert not str(flat.get("output_dir", "")).startswith("runs/")


def test_no_prompt_default_in_method_configs():
    for path in METHODS_CONFIG.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        merged = {}
        for block in (data, data.get("model", {}), data.get("training", {})):
            if isinstance(block, dict):
                merged.update(block)
        assert merged.get("use_prompt") is False, path.name


def test_eta2_metrics_range():
    from metrics.inertia import compute_eta2_inertia_metrics
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16))
    y = rng.choice(["A0", "A1", "B", "C"], size=200)
    m = compute_eta2_inertia_metrics(x, y)
    assert 0.0 <= m["eta2_macro_balanced"] <= 1.0
    assert 0.0 <= m["eta2_weighted"] <= 1.0


def test_method_result_dirs():
    from safer_core.paths import ensure_method_dirs, get_method_dir
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        import safer_core.paths as p

        old = p.RESULTS_ROOT
        p.RESULTS_ROOT = Path(tmp) / "output"
        p.METHOD_RESULTS_DIRS["scgm_text"] = p.RESULTS_ROOT / "scgm_text"
        try:
            root = ensure_method_dirs("scgm_text")
            for sub in ("configs", "checkpoints", "embeddings", "metrics", "logs"):
                assert (root / sub).is_dir()
        finally:
            p.RESULTS_ROOT = old


def test_no_deprecated_duplicate_code_dirs():
    deprecated = (
        TEXT_ROOT / "constrastive method",
        TEXT_ROOT / "learn embeddings",
    )
    for path in deprecated:
        assert not path.exists(), f"Dossier dupliqué à supprimer : {path.name}"


def test_native_contrastive_modules_present():
    pkg = TEXT_ROOT / "contrastive_methods"
    for module in (
        "config.py",
        "data.py",
        "export.py",
        "eval_corpus.py",
        "train.py",
        "training_triplet.py",
        "training_supcon.py",
        "training_softtriple.py",
        "tuning.py",
        "losses/supcon.py",
        "losses/softtriple.py",
    ):
        assert (pkg / module).is_file(), module


def test_jobs_exist():
    for name in (
        "train_scgm_text.sh",
        "export_raw_embeddings_eval.sh",
        "export_corpus_embeddings.sh",
        "export_test_embeddings.sh",
        "train_batch_triplet.sh",
        "train_softtriple.sh",
        "train_supcon.sh",
    ):
        assert (JOBS_DIR / name).is_file(), name


def test_notebooks_no_training():
    forbidden = (
        "run_training(",
        "trainer.fit",
        "spec_from_file_location(train_module_name",
        "scgm_train_text.run_training",
    )
    for name in (
        "02_scgm_text_results.ipynb",
        "05_view_batch_triplet_results.ipynb",
    ):
        nb = TEXT_ROOT / "notebooks" / name
        if not nb.is_file():
            pytest.skip(f"notebook not generated: {name}")
        data = json.loads(nb.read_text(encoding="utf-8"))
        src = "".join("".join(c.get("source", [])) for c in data["cells"])
        for token in forbidden:
            assert token not in src, f"{name} must not reference {token!r}"
