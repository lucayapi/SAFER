"""Extraction de scénarios typiques à partir de la matrice accident × topics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from bn_pipeline.utils import MACRO_NAMES

MACRO_SCENARIO_ORDER: tuple[str, ...] = MACRO_NAMES


def _macro_from_topic_column(name: str) -> Optional[str]:
    n = str(name)
    if not n.startswith("macro_topic_"):
        return None
    parts = n.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


def _macros_in_config(cfg: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for name in cfg:
        macro = _macro_from_topic_column(name)
        if macro:
            out.add(macro)
    return out


def sort_topics_by_macro_order(cfg: Sequence[str]) -> tuple[str, ...]:
    """Ordre causal A0 → A1 → B → C (puis nom de colonne)."""

    def _key(name: str) -> tuple[int, str]:
        macro = _macro_from_topic_column(name) or "Z"
        idx = MACRO_SCENARIO_ORDER.index(macro) if macro in MACRO_SCENARIO_ORDER else 99
        return (idx, str(name))

    return tuple(sorted(cfg, key=_key))


def macro_path_from_topics(cfg: Sequence[str]) -> str:
    """Chemin macro unique, ordonné A0 → A1 → B → C."""
    seen: List[str] = []
    for name in sort_topics_by_macro_order(cfg):
        macro = _macro_from_topic_column(name)
        if macro and macro not in seen:
            seen.append(macro)
    return " -> ".join(seen)


def _config_tuple(row: pd.Series, topic_cols: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for c in topic_cols:
        val = pd.to_numeric(row.get(c, 0), errors="coerce")
        if pd.notna(val) and int(val) == 1:
            out.append(str(c))
    return tuple(out)


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
    *,
    exclude_empty: bool = True,
    require_full_macro_path: bool = False,
    required_macros: Sequence[str] = MACRO_SCENARIO_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrège les configurations binaires de topics par accident ; fréquence et, si demandé, lien gravité.

    Si ``require_full_macro_path`` est True, ne retient que les accidents où au moins un topic
    est présent pour chaque macro de ``required_macros`` (par défaut A0, A1, B, C).
    """
    del bn_model  # réservé extensions BN conditionnelles
    df = accident_topic_matrix.copy()
    topic_cols = [c for c in topic_cols if c in df.columns]
    required = tuple(required_macros)
    use_severity = bool(severity_high_col and severity_high_col in df.columns)
    configs: List[tuple[str, ...]] = []
    sev: List[int] = []
    aids = []
    for _, row in df.iterrows():
        cfg = _config_tuple(row, topic_cols)
        if exclude_empty and not cfg:
            continue
        if require_full_macro_path:
            present = _macros_in_config(cfg)
            if not all(m in present for m in required):
                continue
        configs.append(cfg)
        aids.append(row[accident_id_col])
        if use_severity:
            sev.append(int(row[severity_high_col]))

    ctr = Counter(configs)
    rows_f = []
    rows_h = []
    for cfg, sup in ctr.most_common(top_n * 3):
        if sup < min_support:
            continue
        ordered_cfg = sort_topics_by_macro_order(cfg)
        mask = np.array([configs[j] == cfg for j in range(len(configs))])
        rep_acc = [aids[j] for j in np.flatnonzero(mask)[:5].tolist()]
        rep_sent = _representative_sentences(rep_acc, metadata_unit, text_col)
        path = macro_path_from_topics(ordered_cfg)
        sid = len(rows_f)
        row_common: Dict[str, Any] = {
            "scenario_id": sid,
            "macro_path": path,
            "topics_present": " + ".join(ordered_cfg) if ordered_cfg else "",
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
