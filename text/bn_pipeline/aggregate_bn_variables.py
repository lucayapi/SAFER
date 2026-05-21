"""Agrégation des exports SCGM au niveau accident pour variables binaires de motifs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import MACRO_NAMES, aggregate_severity_by_accident, rank_to_severity_label, severity_to_rank


def _topic_column_name(z_id: int, macro: str) -> str:
    return f"Z_{int(z_id):02d}_{macro}"


def dominant_macro_for_topic(z_id: int, prob_y_z: np.ndarray) -> str:
    id2 = {0: "A0", 1: "A1", 2: "B", 3: "C"}
    row = prob_y_z[int(z_id)]
    return id2[int(np.argmax(row))]


def create_accident_topic_matrix(
    metadata_df: pd.DataFrame,
    accident_id_col: str,
    z_col: str,
    z_conf_col: str,
    z_macro_col: str,
    confidence_threshold: float,
    min_topic_accident_support: int,
    max_topics_per_macro: int,
    prob_y_z: np.ndarray,
    include_macro_aggregate_nodes: bool = True,
    include_severity: bool = True,
    severity_col: str = "pred_severity",
    warn_max_binary_nodes: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construit la matrice accident × topics (binaire) + métadonnées de sélection.

    Retourne
    --------
    accident_topic_df
        Une ligne par accident, colonnes accident_id, Z_* , M_* optionnel,
        Severity_ord, Severity_high, etc.
    selected_variables_df
        description des variables Z sélectionnées (nom, z_id, macro, support, …)
    topic_macro_mapping_df
        z_id, dominant_macro_topic, column_name
    """
    df = metadata_df.copy()
    if accident_id_col not in df.columns:
        raise KeyError(f"Colonne manquante : {accident_id_col}")

    if z_conf_col in df.columns:
        high_conf = df[z_conf_col].astype(float) >= float(confidence_threshold)
    else:
        high_conf = pd.Series(True, index=df.index)

    df["_use_row"] = high_conf.to_numpy()
    df["_z"] = df[z_col].astype(int)

    all_z = sorted(int(z) for z in df.loc[df["_use_row"], "_z"].unique())
    accident_ids = df[accident_id_col].astype(str)

    def accident_has_topic(aid: str, z_id: int) -> bool:
        m = (accident_ids == aid) & df["_use_row"] & (df["_z"] == z_id)
        return bool(m.any())

    support: Dict[int, int] = {}
    for z_id in all_z:
        acc_with = {str(a) for a in df.loc[df["_use_row"] & (df["_z"] == z_id), accident_id_col].unique()}
        support[z_id] = len(acc_with)

    candidates = [z for z in all_z if support[z] >= min_topic_accident_support]

    macro_of_z: Dict[int, str] = {}
    for z_id in candidates:
        if z_macro_col in df.columns:
            sub = df.loc[df["_z"] == z_id, z_macro_col].astype(str)
            if len(sub):
                macro_of_z[z_id] = str(sub.mode().iloc[0])
            else:
                macro_of_z[z_id] = dominant_macro_for_topic(z_id, prob_y_z)
        else:
            macro_of_z[z_id] = dominant_macro_for_topic(z_id, prob_y_z)

    selected: List[int] = []
    for macro in MACRO_NAMES:
        zs = [z for z in candidates if macro_of_z.get(z) == macro]
        zs.sort(key=lambda z: -support[z])
        selected.extend(zs[: max(0, int(max_topics_per_macro))])

    selected = sorted(set(selected))
    if len(selected) > warn_max_binary_nodes:
        import warnings

        warnings.warn(
            f"{len(selected)} variables de topics sélectionnées (> {warn_max_binary_nodes}). "
            "Risque de BN instable ou illisible.",
            UserWarning,
            stacklevel=2,
        )

    n_acc = df[accident_id_col].nunique()
    rows: List[dict] = []
    for aid in sorted(df[accident_id_col].astype(str).unique()):
        row: dict = {accident_id_col: aid}
        sub = df[accident_ids == aid]
        for z_id in selected:
            col = _topic_column_name(z_id, macro_of_z[z_id])
            row[col] = int(accident_has_topic(aid, z_id))

        if include_macro_aggregate_nodes:
            for macro in MACRO_NAMES:
                cols_m = [
                    _topic_column_name(z, macro_of_z[z])
                    for z in selected
                    if macro_of_z.get(z) == macro
                ]
                if cols_m:
                    row[f"M_{macro}"] = int(any(row[c] == 1 for c in cols_m))
                else:
                    row[f"M_{macro}"] = 0

        if include_severity and severity_col in df.columns:
            rnk, lab = aggregate_severity_by_accident(sub[severity_col])
            row["Severity_ord"] = int(rnk)
            row["Severity_label"] = lab
            row["Severity_high"] = int(rnk >= 1)
        rows.append(pd.Series(row))

    accident_topic_df = pd.DataFrame(rows)

    mapping_rows = []
    for z_id in selected:
        mapping_rows.append(
            {
                "z_id": z_id,
                "dominant_macro_topic": macro_of_z[z_id],
                "column_name": _topic_column_name(z_id, macro_of_z[z_id]),
                "n_accidents_support": support[z_id],
            }
        )
    topic_macro_mapping_df = pd.DataFrame(mapping_rows)

    sel_rows = []
    for z_id in selected:
        sel_rows.append(
            {
                "variable": _topic_column_name(z_id, macro_of_z[z_id]),
                "z_id": z_id,
                "macro": macro_of_z[z_id],
                "n_accidents_with_topic": support[z_id],
                "share_accidents": support[z_id] / max(1, n_acc),
            }
        )
    selected_variables_df = pd.DataFrame(sel_rows)

    return accident_topic_df, selected_variables_df, topic_macro_mapping_df


def _macro_topic_column_name(macro: str, topic_id: int) -> str:
    return f"macro_topic_{macro}_{int(topic_id):02d}"


def create_accident_matrix_from_macro_transfer(
    metadata_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    *,
    accident_id_col: str = "accident_id",
    doc_idx_col: str = "doc_idx",
    macro_col: str = "macro",
    topic_id_col: str = "topic_id",
    gamma_col: str = "gamma",
    macro_conf_col: str = "q_conf",
    macro_conf_threshold: float = 0.5,
    topic_gamma_threshold: float = 0.5,
    min_topic_accident_support: int = 20,
    max_topics_per_macro: int = 6,
    include_macro_aggregate_nodes: bool = True,
    warn_max_binary_nodes: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Matrice accident × topics **intra-macro** (colonnes ``macro_topic_<macro>_<id>``).

    ``assignments_df`` : sortie ``topics_bertopic/assignments.csv``.
    """
    meta = metadata_df.copy()
    if accident_id_col not in meta.columns:
        raise KeyError(f"Colonne manquante : {accident_id_col}")
    assign = assignments_df.copy()
    for col in (doc_idx_col, macro_col, topic_id_col):
        if col not in assign.columns:
            raise KeyError(f"assignments : colonne {col!r} manquante")

    n_rows = len(meta)
    doc_max = int(assign[doc_idx_col].max()) if len(assign) else -1
    if doc_max >= n_rows or int(assign[doc_idx_col].min()) < 0:
        raise ValueError("doc_idx hors bornes par rapport à metadata")

    meta["_doc_idx"] = np.arange(n_rows, dtype=np.int64)
    if macro_conf_col in meta.columns:
        meta["_macro_ok"] = meta[macro_conf_col].astype(float) >= float(macro_conf_threshold)
    else:
        meta["_macro_ok"] = True

    g_thresh = float(topic_gamma_threshold)
    if gamma_col in assign.columns:
        assign["_topic_ok"] = assign[gamma_col].astype(float) >= g_thresh
    else:
        assign["_topic_ok"] = True

    merged = assign.merge(
        meta[[accident_id_col, "_doc_idx", "_macro_ok"]],
        left_on=doc_idx_col,
        right_on="_doc_idx",
        how="inner",
    )
    merged = merged.loc[merged["_macro_ok"] & merged["_topic_ok"]]
    unique_pairs = sorted(
        {
            (str(m), int(t))
            for m, t in zip(
                merged[macro_col].astype(str),
                merged[topic_id_col].astype(int),
            )
        }
    )

    support: Dict[tuple[str, int], int] = {}
    for macro, topic_id in unique_pairs:
        sub = merged[
            (merged[macro_col].astype(str) == macro)
            & (merged[topic_id_col].astype(int) == int(topic_id))
        ]
        support[(macro, topic_id)] = int(sub[accident_id_col].astype(str).nunique())

    candidates = [
        (m, t) for (m, t) in unique_pairs if support.get((m, t), 0) >= min_topic_accident_support
    ]
    selected: List[tuple[str, int]] = []
    for macro in MACRO_NAMES:
        zs = [(m, t) for (m, t) in candidates if m == macro]
        zs.sort(key=lambda p: -support.get(p, 0))
        selected.extend(zs[: max(0, int(max_topics_per_macro))])

    selected = sorted(set(selected))
    if len(selected) > warn_max_binary_nodes:
        import warnings

        warnings.warn(
            f"{len(selected)} variables macro_topic (> {warn_max_binary_nodes}).",
            UserWarning,
            stacklevel=2,
        )

    n_acc = meta[accident_id_col].nunique()
    rows: List[dict] = []
    for aid in sorted(meta[accident_id_col].astype(str).unique()):
        row: dict = {accident_id_col: aid}
        sub_a = merged[merged[accident_id_col].astype(str) == aid]
        for macro, topic_id in selected:
            col = _macro_topic_column_name(macro, topic_id)
            hit = sub_a[
                (sub_a[macro_col].astype(str) == macro)
                & (sub_a[topic_id_col].astype(int) == int(topic_id))
            ]
            row[col] = int(len(hit) > 0)
        if include_macro_aggregate_nodes:
            for macro in MACRO_NAMES:
                cols_m = [
                    _macro_topic_column_name(m, t)
                    for (m, t) in selected
                    if m == macro
                ]
                row[f"M_{macro}"] = int(any(row.get(c, 0) == 1 for c in cols_m)) if cols_m else 0
        rows.append(row)

    accident_topic_df = pd.DataFrame(rows)
    mapping_rows = [
        {
            "macro": macro,
            "topic_id": topic_id,
            "column_name": _macro_topic_column_name(macro, topic_id),
            "n_accidents_support": support.get((macro, topic_id), 0),
        }
        for macro, topic_id in selected
    ]
    topic_macro_mapping_df = pd.DataFrame(mapping_rows)
    sel_rows = [
        {
            "variable": _macro_topic_column_name(macro, topic_id),
            "macro": macro,
            "topic_id": topic_id,
            "n_accidents_with_topic": support.get((macro, topic_id), 0),
            "share_accidents": support.get((macro, topic_id), 0) / max(1, n_acc),
        }
        for macro, topic_id in selected
    ]
    selected_variables_df = pd.DataFrame(sel_rows)
    return accident_topic_df, selected_variables_df, topic_macro_mapping_df


def export_aggregate_outputs(
    accident_topic_df: pd.DataFrame,
    selected_variables_df: pd.DataFrame,
    topic_macro_mapping_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accident_topic_df.to_csv(output_dir / "accident_topic_matrix.csv", index=False)
    selected_variables_df.to_csv(output_dir / "selected_bn_variables.csv", index=False)
    topic_macro_mapping_df.to_csv(output_dir / "topic_macro_mapping.csv", index=False)
