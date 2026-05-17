"""Agrégation des sorties MALT au niveau accident pour variables binaires de motifs."""

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
