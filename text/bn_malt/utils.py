"""Chargement des exports MALT pour le pipeline BN (chemins relatifs à la racine du dépôt)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from topic_eval.paths import find_repo_root, resolve_repo_path

REQUIRED_BN_FILES = (
    "metadata_with_malt_predictions.csv",
    "pt_z_target.npy",
    "pt_y_target.npy",
    "pt_y_given_z.npy",
    "z_assignments_target.csv",
)

MACRO_NAMES = ("A0", "A1", "B", "C")
SEVERITY_ORDER = {"NONE": 0, "none": 0, "HOSPITALIZED": 1, "hospitalized": 1, "FATAL": 2, "fatal": 2}
SEVERITY_LABELS = ("NONE", "HOSPITALIZED", "FATAL")

METADATA_BASE_COLS = (
    "accident_id",
    "fact_id",
    "doc_id",
    "sentence",
    "accident_summary",
    "pred_label",
    "pred_subtype",
    "pred_severity",
    "z_hat",
    "z_confidence",
    "z_dominant_macro",
    "p0_macro_name",
    "pt_macro_name",
    "p_A0_given_z",
    "p_A1_given_z",
    "p_B_given_z",
    "p_C_given_z",
)


def require_bn_malt_files(exports_dir: str | Path, *, repo_root: Path | None = None) -> Path:
    d = resolve_repo_path(exports_dir, repo_root)
    missing = [f for f in REQUIRED_BN_FILES if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Fichiers MALT manquants pour le BN dans "
            f"{d.resolve()} :\n" + "\n".join(f"  - {m}" for m in missing)
        )
    return d


def _argmax_macro_row(row: np.ndarray, id2label: dict[int, str]) -> str:
    return id2label[int(np.argmax(row))]


def enrich_metadata_for_bn(meta: pd.DataFrame, exports_dir: Path) -> pd.DataFrame:
    """Complète z_hat, z_confidence, z_dominant_macro, p_y|z si absents (alignement lignes = CSV)."""
    out = meta.copy()
    n = len(out)
    id2 = {0: "A0", 1: "A1", 2: "B", 3: "C"}

    pz = np.load(exports_dir / "pt_z_target.npy")
    if pz.shape[0] != n:
        raise ValueError(f"pt_z_target.npy ({pz.shape[0]} lignes) ≠ metadata ({n}).")
    if "z_hat" not in out.columns:
        out["z_hat"] = np.argmax(pz, axis=1)
    if "z_confidence" not in out.columns:
        out["z_confidence"] = np.max(pz, axis=1)

    prob_y_z = np.load(exports_dir / "pt_y_given_z.npy")
    z_hat_arr = out["z_hat"].to_numpy(dtype=int)
    if "z_dominant_macro" not in out.columns:
        out["z_dominant_macro"] = [_argmax_macro_row(prob_y_z[int(z)], id2) for z in z_hat_arr]

    for i, name in enumerate(MACRO_NAMES):
        col = f"p_{name}_given_z"
        if col not in out.columns:
            out[col] = prob_y_z[z_hat_arr, i]

    pt = np.load(exports_dir / "pt_y_target.npy")
    if pt.shape[0] != n:
        raise ValueError(f"pt_y_target.npy ({pt.shape[0]} lignes) ≠ metadata ({n}).")
    for i, name in enumerate(MACRO_NAMES):
        c = f"pt_{name}"
        if c not in out.columns:
            out[c] = pt[:, i]
    if "pt_macro_name" not in out.columns:
        out["pt_macro_name"] = [_argmax_macro_row(pt[j], id2) for j in range(n)]

    return out


def load_metadata_for_bn(exports_dir: str | Path, *, repo_root: Path | None = None) -> tuple[pd.DataFrame, Path]:
    d = require_bn_malt_files(exports_dir, repo_root=repo_root)
    usecols = [c for c in METADATA_BASE_COLS if c != "accident_summary"]
    path = d / "metadata_with_malt_predictions.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in usecols if c in header]
    if "accident_summary" in header and "accident_summary" not in cols:
        cols.append("accident_summary")
    meta = pd.read_csv(path, usecols=cols, low_memory=False)
    if "accident_id" not in meta.columns:
        raise KeyError(
            "La colonne accident_id est obligatoire pour l'agrégation au niveau accident. "
            "Vérifiez metadata_with_malt_predictions.csv."
        )
    meta = enrich_metadata_for_bn(meta, d)
    return meta, d


def severity_to_rank(val) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    s = str(val).strip()
    return int(SEVERITY_ORDER.get(s, SEVERITY_ORDER.get(s.upper(), 0)))


def rank_to_severity_label(rank: int) -> str:
    rank = int(max(0, min(rank, len(SEVERITY_LABELS) - 1)))
    return SEVERITY_LABELS[rank]


def aggregate_severity_by_accident(series: Iterable) -> tuple[int, str]:
    ranks = [severity_to_rank(x) for x in series]
    mx = max(ranks) if ranks else 0
    return mx, rank_to_severity_label(mx)


def ensure_output_dirs(root: Path) -> None:
    """Crée l’arborescence standard des sorties BN (BTP / MALT)."""
    for sub in (
        "staging/malt_like_exports",
        "tables",
        "tables/cpds_macro",
        "tables/cpds_topic",
        "models",
        "figures/static",
        "figures/interactive",
        "figures/nodes",
        "reports",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)
