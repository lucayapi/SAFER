"""Tests builders notebooks FSP (04/06/08)."""

from __future__ import annotations

from pathlib import Path


TEXT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builder_06_references_fsp_and_not_tpn_compare():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_06_macro_transfer_topics.py")
    assert "frozen_source_prototypes" in src
    assert "FSP_BASE_METHOD" in src
    assert "target_macro_predictions.csv" in src
    assert "source_prototypes.csv" in src
    assert "bertopic_input_all.csv" in src
    assert "TPN-SCGM" not in src
    assert "TPN-SoftTriple" not in src
    assert "metadata_with_tpn_macro_probs" not in src
    assert "transfer_metrics_adapted" not in src


def test_builder_08_references_fsp_and_not_tpn_training():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_08_fsp_macro_transfer.py")
    assert "frozen_source_prototypes" in src
    assert "FSP_BASE_METHOD" in src
    assert "FSP_ENCODER_METHODS" in src
    assert "raw_embedding" in src
    assert "load_fsp_run_artifacts" in src
    assert "table_transfer_direct_all_methods" in src
    assert "load_fsp_metrics_comparison_table" in src
    assert "export_fsp_metrics_latex_table" in src
    assert "classification_report.csv" in src
    assert "confusion_matrix.csv" in src
    assert "training_log.csv" not in src
    assert "metadata_with_initial_macro_probs.csv" not in src
    assert "transfer_metrics_adapted" not in src


def test_builder_04_defaults_to_fsp_scgm_text():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_04_bn_macro_transfer.py")
    assert "frozen_source_prototypes/scgm_text" in src or "FSP_BASE_METHOD" in src
    assert "run_frozen_source_prototypes.sh" in src
    assert "bn_network.png" in src
    assert "bn_network.html" in src
    assert "bn_path_scenarios.csv" in src
    assert "SCENARIO_MIN_MACROS" in src
    assert "bn_network_slide.png" in src
    assert "bn_scenario_slide_" in src
    assert "SCENARIO_SLIDE_TOP_N" in src
    assert "plot_bn_scenario_slide" in src
    assert "BN_DISALLOW_A0_TO_B" in src
    assert "BN_ENSURE_MACRO_CHAIN" in src
    assert "extract_bn_path_scenarios" in src
    assert "extract_subgraph_for_slide" in src
    assert "SCENARIO_MODE" not in src
    assert "run_bn_queries" not in src


def test_view_builders_include_raw_test_embedding_viz():
    builders = [
        "build_notebook_02_scgm_results.py",
        "build_notebook_04_bn_macro_transfer.py",
        "build_notebook_06_macro_transfer_topics.py",
        "build_notebook_08_fsp_macro_transfer.py",
        "build_contrastive_view_notebooks.py",
    ]
    for name in builders:
        src = _read(TEXT_ROOT / "scripts" / name)
        assert (
            "plot_test_corpus_raw_embeddings" in src
            or "notebook_raw_test_embedding_source" in src
        ), name


def test_no_legacy_notebook_builders_07_09():
    scripts = TEXT_ROOT / "scripts"
    assert not (scripts / "build_notebook_07_macro_transfer_interactive.py").exists()
    assert not (scripts / "build_notebook_09_bn_tpn_macro_transfer.py").exists()
    assert not (scripts / "build_notebook_08_tpn_macro_transfer.py").exists()


def test_single_fsp_job_exists():
    assert (TEXT_ROOT / "jobs" / "run_frozen_source_prototypes.sh").is_file()
    job = _read(TEXT_ROOT / "jobs" / "run_frozen_source_prototypes.sh")
    assert "BASE_METHOD" in job
    assert "SKIP_BERTOPIC" not in job


def test_softtriple_native_fsp_job_exists():
    assert (TEXT_ROOT / "jobs" / "run_softtriple_native_fsp.sh").is_file()
    job = _read(TEXT_ROOT / "jobs" / "run_softtriple_native_fsp.sh")
    assert "run_softtriple_native_fsp.py" in job
    assert "frozen_source_prototypes_softtriple_native.yaml" in job


def test_builder_09_softtriple_native_diagnostics():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_09_softtriple_native_fsp.py")
    assert "softtriple_native" in src
    assert "load_softtriple_native_vs_legacy_metrics" in src
    assert "plot_softtriple_center_similarity_heatmap" in src
    assert "run_softtriple_native_fsp.sh" in src
    assert "softtriple_center_weights_summary" in src
