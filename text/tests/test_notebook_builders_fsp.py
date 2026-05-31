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


def test_builder_04_defaults_to_fsp_scgm():
    src = _read(TEXT_ROOT / "scripts" / "build_notebook_04_bn_macro_transfer.py")
    assert "frozen_source_prototypes/scgm" in src
    assert "run_frozen_proto_scgm.sh" in src
    assert "bn_network.png" in src
    assert "bn_network.html" in src
    assert "recurring_scenarios.csv" in src
    assert "enrich_scenarios_table" in src
    assert "run_bn_queries" not in src
    assert "export_node_cards_png" not in src
    assert "try_pyvis_bn_graph" not in src
    assert "write_bn_report" not in src
    assert "LEARN_UNCONSTRAINED_TOPIC" not in src
    assert "Inférence (VariableElimination)" not in src
    assert "Configurations typiques de co-présence" not in src
