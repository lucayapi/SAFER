"""Tests légers pour macro_transfer.notebook_viz (sans fit BERTopic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_transfer.notebook_viz import (
    FSPRunArtifacts,
    RawTestEmbeddingVizResult,
    build_topics_display_dataframe,
    compute_fsp_confidence_calibration,
    export_fsp_metrics_latex_table,
    get_fsp_top_confident_errors,
    load_fsp_metrics_comparison_table,
    load_fsp_run_artifacts,
    load_softtriple_native_vs_legacy_metrics,
    merge_assignments,
    pick_accident_id_for_colored_text,
    plot_softtriple_center_similarity_heatmap,
    render_colored_accident_html,
    resolve_test_label_col,
    _theme_label_map,
)
from safer_core.test_corpus import macro_transfer_output_dir

TEXT_ROOT = Path(__file__).resolve().parents[1]


def test_import_notebook_viz():
    import macro_transfer.notebook_viz as nv

    assert hasattr(nv, "plot_test_corpus_raw_embeddings")
    assert hasattr(nv, "load_fsp_run_artifacts")


def test_macro_transfer_output_dir_convention():
    p = macro_transfer_output_dir("frozen_source_prototypes/scgm_text", "metallurgie", anchor=TEXT_ROOT)
    assert p.name == "scgm_text"
    assert p.parent.name == "frozen_source_prototypes"
    assert "metallurgie" in str(p)


def test_merge_assignments_and_theme_map():
    meta = pd.DataFrame(
        {
            "doc_idx": [0, 1, 2],
            "m_hat": ["A0", "A0", "B"],
            "q_conf": [0.9, 0.8, 0.95],
            "sentence": ["a", "b", "c"],
        }
    )
    assignments = pd.DataFrame(
        {
            "doc_idx": [0, 1, 2],
            "macro": ["A0", "A0", "B"],
            "topic_id": [0, 1, 0],
            "prob": [0.9, 0.85, 0.7],
        }
    )
    merged = merge_assignments(meta, assignments, confidence_threshold=0.5)
    assert "topic_id" in merged.columns
    themes = pd.DataFrame(
        {
            "macro": ["A0", "A0"],
            "topic_id": [0, 1],
            "theme_label": ["Label long A0-0", "Label long A0-1"],
            "top_words": ["w1", "w2"],
        }
    )
    tmap = _theme_label_map(themes, max_chars=20)
    assert ("A0", 0) in tmap


def _minimal_fsp_run(tmp_path: Path) -> Path:
    root = tmp_path / "fsp_run"
    transfer = root / "transfer"
    transfer.mkdir(parents=True)
    topics = root / "topics_bertopic"
    topics.mkdir()
    n = 6
    meta = pd.DataFrame(
        {
            "sentence": [f"phrase {i} sur l'accident." for i in range(n)],
            "accident_id": [1, 1, 1, 2, 2, 2],
            "pred_macro": ["A0", "A0", "A1", "B", "B", "C"],
            "confidence": [0.95, 0.88, 0.91, 0.85, 0.80, 0.99],
            "doc_idx": list(range(n)),
        }
    )
    meta.to_csv(transfer / "target_macro_predictions.csv", index=False)
    pd.DataFrame(
        {
            "doc_idx": [0, 1, 2, 3, 4, 5],
            "macro": ["A0", "A0", "A1", "B", "B", "C"],
            "topic_id": [0, 1, 0, 0, 1, 0],
            "prob": [0.9, 0.8, 0.85, 0.7, 0.75, 0.92],
        }
    ).to_csv(topics / "assignments.csv", index=False)
    pd.DataFrame(
        {
            "macro": ["A0", "A0", "A1", "B", "B", "C"],
            "topic_id": [0, 1, 0, 0, 1, 0],
            "theme_label": ["Thème A0-0", "Thème A0-1", "Thème A1-0", "Thème B-0", "Thème B-1", "Thème C-0"],
            "n_units": [1, 1, 1, 1, 1, 1],
        }
    ).to_csv(topics / "themes_by_macro.csv", index=False)
    return root


def test_build_topics_display_dataframe(tmp_path: Path):
    root = _minimal_fsp_run(tmp_path)
    meta = pd.read_csv(root / "transfer" / "target_macro_predictions.csv")
    df = build_topics_display_dataframe(root, meta, confidence_threshold=0.0)
    assert "theme_label" in df.columns
    assert "theme_label_short" in df.columns
    assert "Thème A0-0" in df.loc[df["doc_idx"] == 0, "theme_label_short"].iloc[0]
    assert len(df) == 6


def test_pick_accident_id_for_colored_text():
    df = pd.DataFrame(
        {
            "accident_id": [1, 1, 1, 2, 2],
            "sentence": ["a", "b", "c", "d", "e"],
        }
    )
    assert pick_accident_id_for_colored_text(df, min_units=3) == 1
    assert pick_accident_id_for_colored_text(df, min_units=3, prefer_id=2) == 2


def test_render_colored_accident_html(tmp_path: Path):
    root = _minimal_fsp_run(tmp_path)
    meta = pd.read_csv(root / "transfer" / "target_macro_predictions.csv")
    df = build_topics_display_dataframe(root, meta, confidence_threshold=0.0)
    html_topic = render_colored_accident_html(df, 1, color_by="topic")
    assert "phrase 0" in html_topic


def test_load_fsp_run_artifacts_minimal(tmp_path: Path):
    root = tmp_path / "fsp"
    transfer = root / "transfer"
    transfer.mkdir(parents=True)
    preds = pd.DataFrame(
        {
            "pred_macro": ["A0", "B"],
            "confidence": [0.8, 0.7],
            "prob_A0": [0.8, 0.1],
            "prob_A1": [0.1, 0.1],
            "prob_B": [0.05, 0.7],
            "prob_C": [0.05, 0.1],
        }
    )
    preds.to_csv(transfer / "target_macro_predictions.csv", index=False)
    pd.DataFrame({"macro": ["A0"], "n_source": [10]}).to_csv(
        transfer / "source_prototypes.csv", index=False
    )
    with open(transfer / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"balanced_accuracy": 0.5}, f)

    art = load_fsp_run_artifacts(root)
    assert isinstance(art, FSPRunArtifacts)
    assert len(art.predictions) == 2


def test_load_fsp_run_artifacts_legacy_scgm_sibling(tmp_path: Path):
    corpus_root = tmp_path / "output_test" / "metallurgie" / "macro_transfer" / "frozen_source_prototypes"
    legacy_transfer = corpus_root / "scgm" / "transfer"
    legacy_transfer.mkdir(parents=True)
    pd.DataFrame({"pred_macro": ["A0"], "confidence": [0.8]}).to_csv(
        legacy_transfer / "target_macro_predictions.csv", index=False
    )
    pd.DataFrame({"macro": ["A0"], "n_source": [1]}).to_csv(
        legacy_transfer / "source_prototypes.csv", index=False
    )

    canonical = corpus_root / "scgm_text"
    art = load_fsp_run_artifacts(canonical)
    assert art.out_dir.name == "scgm"
    assert len(art.predictions) == 1


def test_fsp_confidence_calibration_and_errors():
    pred = pd.DataFrame(
        {
            "true_macro": ["A0", "A1", "A0", "B"],
            "pred_macro": ["A0", "A0", "B", "B"],
            "confidence": [0.9, 0.85, 0.6, 0.95],
        }
    )
    calib = compute_fsp_confidence_calibration(pred)
    assert calib is not None
    err = get_fsp_top_confident_errors(pred, top_k=2)
    assert len(err) >= 1


def test_resolve_test_label_col():
    meta = pd.DataFrame({"pred_label": ["A0"], "sentence": ["x"]})
    assert resolve_test_label_col(meta) == "pred_label"


def test_raw_test_embedding_viz_result_dataclass():
    r = RawTestEmbeddingVizResult(n_points=10, missing=[])
    assert r.n_points == 10


def test_load_fsp_metrics_comparison_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    anchor = tmp_path
    corpus = "metallurgie"

    def _fake_out(method: str) -> Path:
        root = anchor / "output_test" / corpus / "macro_transfer" / f"frozen_source_prototypes/{method}"
        transfer = root / "transfer"
        transfer.mkdir(parents=True, exist_ok=True)
        metrics = {"balanced_accuracy": 0.4, "macro_f1": 0.3}
        if method == "scgm_text":
            metrics["method"] = "SCGM custom"
        with open(transfer / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        return root

    monkeypatch.setattr(
        "macro_transfer.fsp_config.resolve_fsp_output_dir",
        lambda corpus_id, base_method, *, anchor, output_dir=None: _fake_out(base_method),
    )

    df = load_fsp_metrics_comparison_table(corpus, methods=["scgm_text", "raw_embedding"], anchor=anchor)
    assert len(df) == 2
    assert df.loc[df["method_key"] == "scgm_text", "Méthode"].iloc[0] == "SCGM custom"
    assert df["metrics_available"].all()
    latex = export_fsp_metrics_latex_table(df[["Méthode", "Bal. Acc.", "F1 (étapes)", "Confiance moy.", "Entropie moy."]])
    assert "\\toprule" in latex


def test_load_softtriple_native_vs_legacy_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    anchor = tmp_path
    corpus = "metallurgie"

    def _fake_out(method: str) -> Path:
        root = anchor / "output_test" / corpus / "macro_transfer" / f"frozen_source_prototypes/{method}"
        transfer = root / "transfer"
        transfer.mkdir(parents=True, exist_ok=True)
        metrics = {
            "balanced_accuracy": 0.5 if method == "softtriple_native" else 0.4,
            "macro_f1": 0.35,
            "assignment_mode": "softtriple_native_centers" if method == "softtriple_native" else None,
        }
        with open(transfer / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        return root

    monkeypatch.setattr(
        "macro_transfer.fsp_config.resolve_fsp_output_dir",
        lambda corpus_id, base_method, *, anchor, output_dir=None: _fake_out(base_method),
    )

    df = load_softtriple_native_vs_legacy_metrics(corpus, anchor=anchor)
    assert len(df) == 3
    assert df.loc[df["method_key"] == "delta_native_minus_legacy", "Bal. Acc."].iloc[0] == pytest.approx(0.1)


def test_plot_softtriple_center_similarity_heatmap_runs(tmp_path: Path):
    pytest.importorskip("seaborn")
    protos = pd.DataFrame(
        {
            "macro": ["A0", "A0", "A1"],
            "center_k": [0, 1, 0],
            "dim_0000": [1.0, 0.0, 0.0],
            "dim_0001": [0.0, 1.0, 1.0],
        }
    )
    plot_softtriple_center_similarity_heatmap(protos, fig_dir=tmp_path)
    assert (tmp_path / "softtriple_center_similarity_heatmap.png").is_file()


def test_plot_tsne_true_vs_pred_runs(tmp_path: Path):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    h = rng.normal(size=(80, 16))
    df = pd.DataFrame(
        {
            "true_macro": np.array(["A0", "A1", "B", "C"] * 20),
            "pred_macro": np.array(["A0", "A1", "B", "C"] * 20),
        }
    )
    from macro_transfer.notebook_viz import plot_tsne_true_vs_pred

    out = plot_tsne_true_vs_pred(
        h,
        df,
        "true_macro",
        "pred_macro",
        title="test",
        fig_dir=tmp_path,
        filename="tsne_true_vs_pred.png",
        max_points=80,
    )
    assert out is not None and out.is_file()
