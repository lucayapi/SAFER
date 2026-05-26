"""Tableaux récapitulatifs macro_transfer (notebooks + exports CSV)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.topics_export import build_macro_topic_test_table


_ENCODER_LABELS: Dict[str, str] = {
    "scgm_text": "SCGM",
    "softtriple": "SoftTriple",
    "supcon": "SupCon",
    "batch_triplet": "BatchTriplet",
}


def encoder_display_name(base_encoder: str) -> str:
    key = str(base_encoder).strip().lower()
    return _ENCODER_LABELS.get(key, str(base_encoder))


def build_transfer_metrics_comparison(
    metrics_initial: Dict[str, Any],
    metrics_adapted: Dict[str, Any],
    base_encoder: str,
) -> pd.DataFrame:
    """Tableau initial vs adapté pour summary/transfer_metrics_comparison.csv."""
    enc = encoder_display_name(base_encoder)
    rows = []
    for phase, m, suffix in (
        ("initial", metrics_initial, "initial"),
        ("adapted", metrics_adapted, "adapté"),
    ):
        rows.append(
            {
                "phase": phase,
                "modele": f"{enc} {suffix}",
                "balanced_accuracy": m.get("balanced_accuracy"),
                "macro_f1": m.get("macro_f1"),
                "accuracy": m.get("accuracy"),
                "mean_q_conf": m.get("mean_q_conf"),
                "mean_entropy": m.get("mean_entropy"),
                "mean_margin": m.get("mean_margin"),
                "n_eval": m.get("n_eval"),
            }
        )
    return pd.DataFrame(rows)


def format_transfer_metrics_table(
    metrics_initial: Dict[str, Any],
    metrics_adapted: Dict[str, Any],
    base_encoder: str,
) -> pd.DataFrame:
    """Affichage notebook : Modèle, Bal. Acc., Macro-F1, Confiance moy., Entropie moy."""
    raw = build_transfer_metrics_comparison(metrics_initial, metrics_adapted, base_encoder)
    if raw.empty:
        return raw
    out = raw[["modele", "balanced_accuracy", "macro_f1", "mean_q_conf", "mean_entropy"]].copy()
    out.columns = [
        "Modèle",
        "Bal. Acc.",
        "Macro-F1",
        "Confiance moy.",
        "Entropie moy.",
    ]
    for col in ("Bal. Acc.", "Macro-F1", "Confiance moy.", "Entropie moy."):
        if col in out.columns:
            out[col] = out[col].astype(float).round(2)
    return out


def load_transfer_metrics_pair(out_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Charge transfer_metrics_{initial,adapted}.json."""
    root = Path(out_dir)
    transfer = root / "transfer"

    def _load(prefix: str) -> Dict[str, Any]:
        p = transfer / f"transfer_metrics_{prefix}.json"
        if not p.is_file():
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    return _load("initial"), _load("adapted")


def load_macro_topic_stats(out_dir: Path) -> pd.DataFrame:
    """
    Lit summary/macro_topic_stats.csv ou reconstruit depuis manifest + assignments.
    """
    root = Path(out_dir)
    stats_path = root / "summary" / "macro_topic_stats.csv"
    if stats_path.is_file():
        return pd.read_csv(stats_path)

    manifest_path = root / "run_manifest.json"
    macro_counts: Dict[str, Any] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        bert = manifest.get("bertopic_summary") or {}
        macro_counts = bert.get("macro_topic_counts") or {}

    topics_dir = root / "topics_bertopic"
    assign_path = topics_dir / "assignments.csv"
    themes_path = topics_dir / "themes_by_macro.csv"
    assignments = pd.read_csv(assign_path) if assign_path.is_file() else pd.DataFrame()
    themes = pd.read_csv(themes_path) if themes_path.is_file() else pd.DataFrame()

    if not macro_counts and assignments.empty:
        return pd.DataFrame(
            columns=[
                "macro",
                "n_units",
                "n_topics",
                "bruit_pct",
                "plus_gros_topic",
                "plus_gros_topic_pct",
            ]
        )
    return build_macro_topic_test_table(macro_counts, assignments, themes)


def embedding_paths_manifest(out_dir: Path) -> Dict[str, str]:
    """Chemins relatifs des 4 fichiers d'embeddings pour run_manifest."""
    emb = Path(out_dir) / "embeddings"
    names = (
        "source_projected",
        "target_projected",
        "source_adapted",
        "target_adapted",
    )
    out: Dict[str, str] = {}
    for name in names:
        p = emb / f"{name}.npy"
        if p.is_file():
            out[name] = str(p.relative_to(out_dir))
    return out
