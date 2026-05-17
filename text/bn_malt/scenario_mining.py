"""Extraction de scénarios typiques à partir de la matrice accident × topics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _config_tuple(row: pd.Series, topic_cols: Sequence[str]) -> tuple:
    return tuple(sorted(c for c in topic_cols if int(row.get(c, 0)) == 1))


def extract_typical_scenarios(
    accident_topic_matrix: pd.DataFrame,
    bn_model: Optional[Any],
    topic_cols: Sequence[str],
    accident_id_col: str,
    severity_high_col: Optional[str] = None,
    min_support: int = 5,
    top_n: int = 30,
    metadata_unit: Optional[pd.DataFrame] = None,
    text_col: str = "sentence",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrège les configurations binaires de topics par accident ; fréquence et, si demandé, lien gravité.

    Si ``severity_high_col`` est ``None`` ou absent du tableau, aucune métrique de gravité n'est produite
    et ``high_df`` est un DataFrame vide.
    """
    df = accident_topic_matrix.copy()
    topic_cols = [c for c in topic_cols if c in df.columns]
    use_severity = bool(
        severity_high_col and severity_high_col in df.columns
    )
    configs: List[tuple] = []
    sev: List[int] = []
    aids = []
    for _, row in df.iterrows():
        cfg = _config_tuple(row, topic_cols)
        configs.append(cfg)
        aids.append(row[accident_id_col])
        if use_severity:
            sev.append(int(row[severity_high_col]))

    ctr = Counter(configs)
    rows_f = []
    rows_h = []
    for i, (cfg, sup) in enumerate(ctr.most_common(top_n * 3)):
        if sup < min_support:
            continue
        mask = np.array([configs[j] == cfg for j in range(len(configs))])
        rep_acc = [aids[j] for j in np.flatnonzero(mask)[:5].tolist()]
        rep_sent = _representative_sentences(rep_acc, metadata_unit, text_col)
        macro_path = " -> ".join(_macro_path_from_topics(cfg)) if cfg else ""
        sid = len(rows_f)
        row_common: Dict[str, Any] = {
            "scenario_id": sid,
            "macro_path": macro_path,
            "topics_present": " + ".join(cfg) if cfg else "",
            "support": sup,
            "representative_accidents": " | ".join(map(str, rep_acc)),
            "representative_sentences": " || ".join(rep_sent[:5]),
        }
        p_sev = 0.0
        if use_severity:
            p_sev = float(np.mean(np.array(sev)[mask])) if mask.any() else 0.0
            row_common["p_severity_high"] = p_sev
            row_common["lift_severity"] = float("nan")
        rows_f.append(row_common.copy())
        if use_severity and p_sev > 0:
            rh = row_common.copy()
            rh["scenario_id"] = len(rows_h)
            rows_h.append(rh)

    freq_df = pd.DataFrame(rows_f).head(top_n)
    if use_severity and rows_h:
        high_df = (
            pd.DataFrame(rows_h)
            .sort_values("p_severity_high", ascending=False)
            .head(top_n)
        )
    else:
        high_df = pd.DataFrame()
    return freq_df, high_df


def _macro_path_from_topics(cfg: tuple) -> List[str]:
    out = []
    for name in cfg:
        parts = str(name).split("_")
        if len(parts) >= 3:
            out.append(parts[2])
    return out


def _representative_sentences(
    accident_ids: List,
    metadata_unit: Optional[pd.DataFrame],
    text_col: str,
) -> List[str]:
    if metadata_unit is None or text_col not in metadata_unit.columns:
        return []
    out = []
    for aid in accident_ids:
        sub = metadata_unit.loc[metadata_unit["accident_id"].astype(str) == str(aid), text_col]
        for s in sub.head(2):
            out.append(str(s)[:200])
    return out


def export_scenarios(freq: pd.DataFrame, high: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out_dir / "frequent_scenarios.csv", index=False)
    high.to_csv(out_dir / "high_risk_scenarios.csv", index=False)
