from __future__ import annotations

from pathlib import Path


TEXT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builder_06_references_fsp_and_not_tpn_compare():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_06_macro_transfer_topics.py")
    assert "frozen_source_prototypes" in src
    assert "target_macro_predictions.csv" in src
    assert "source_prototypes.csv" in src
    assert "bertopic_input_all.csv" in src
    assert "TPN-SCGM" not in src
    assert "TPN-SoftTriple" not in src
    assert "metadata_with_tpn_macro_probs" not in src
    assert "transfer_metrics_adapted" not in src


def test_builder_08_references_fsp_and_not_tpn_training():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_08_tpn_macro_transfer.py")
    assert "frozen_source_prototypes" in src
    assert "target_macro_predictions.csv" in src
    assert "raw\" / \"transfer\" / \"metrics.json" in src
    assert "scgm\" / \"transfer\" / \"metrics.json" in src
    assert "table_transfer_direct.csv" in src
    assert "table_transfer_direct.tex" in src
    assert "\\\\toprule" in src
    assert "classification_report.csv" in src
    assert "confusion_matrix.csv" in src
    assert "training_log.csv" not in src
    assert "metadata_with_initial_macro_probs.csv" not in src
    assert "transfer_metrics_adapted" not in src
