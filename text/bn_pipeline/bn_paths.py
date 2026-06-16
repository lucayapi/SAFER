"""Chemins causaux dans le DAG appris et scénarios par support accident."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import networkx as nx
import pandas as pd

from bn_pipeline.bn_structure import macro_label_of_node, macro_rank
from bn_pipeline.scenario_mining import (
    _config_tuple,
    _macros_in_config,
    _representative_sentences,
    macro_path_from_topics,
    sort_topics_by_macro_order,
)


def _path_macros(path: Sequence[str], variable_macro_map: Dict[str, str]) -> list[str]:
    seen: list[str] = []
    for node in path:
        macro = macro_label_of_node(str(node), variable_macro_map)
        if macro and macro not in seen:
            seen.append(macro)
    return seen


def _path_macro_order_valid(path: Sequence[str], variable_macro_map: Dict[str, str]) -> bool:
    ranks = [macro_rank(str(node), variable_macro_map) for node in path]
    return all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))


def _path_has_min_macros(
    path: Sequence[str],
    variable_macro_map: Dict[str, str],
    min_macros: int,
) -> bool:
    return len(_path_macros(path, variable_macro_map)) >= int(min_macros)


def enumerate_macro_paths(
    model: Any,
    variable_macro_map: Dict[str, str],
    *,
    min_macros: int = 2,
    max_path_len: int = 8,
    max_paths: int = 500,
) -> list[tuple[str, ...]]:
    """
    Énumère les chemins simples du DAG avec au moins ``min_macros`` macros distinctes
    et un ordre macro non décroissant (A0 → … → C).
    """
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())
    nodes = [str(n) for n in graph.nodes()]
    if len(nodes) < 2:
        return []

    min_macros = max(2, int(min_macros))
    results: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for src in nodes:
        src_rank = macro_rank(src, variable_macro_map)
        for tgt in nodes:
            if src == tgt:
                continue
            if macro_rank(tgt, variable_macro_map) < src_rank:
                continue
            try:
                for path in nx.all_simple_paths(graph, src, tgt, cutoff=max_path_len):
                    if len(path) < 2:
                        continue
                    path_tuple = tuple(str(node) for node in path)
                    if path_tuple in seen:
                        continue
                    if not _path_macro_order_valid(path_tuple, variable_macro_map):
                        continue
                    if not _path_has_min_macros(path_tuple, variable_macro_map, min_macros):
                        continue
                    seen.add(path_tuple)
                    results.append(path_tuple)
                    if len(results) >= max_paths:
                        return results
            except nx.NetworkXNoPath:
                continue

    return results


def _accidents_with_min_macros(
    configs: Sequence[tuple[str, ...]],
    min_macros: int,
) -> int:
    return sum(1 for cfg in configs if len(_macros_in_config(cfg)) >= min_macros)


def _support_for_path(path: Sequence[str], configs: Sequence[tuple[str, ...]]) -> int:
    path_set = set(path)
    return sum(1 for cfg in configs if path_set.issubset(set(cfg)))


def extract_bn_path_scenarios(
    accident_topic_matrix: pd.DataFrame,
    bn_model: Any,
    topic_cols: Sequence[str],
    variable_macro_map: Dict[str, str],
    accident_id_col: str = "accident_id",
    min_support: int = 3,
    top_n: int = 30,
    metadata_unit: Optional[pd.DataFrame] = None,
    text_col: str = "sentence",
    *,
    min_macros: int = 2,
    max_path_len: int = 8,
    max_paths: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scénarios = chemins du DAG (≥ ``min_macros`` macros) + support accident."""
    df = accident_topic_matrix.copy()
    topic_cols = [c for c in topic_cols if c in df.columns]
    min_macros = max(2, int(min_macros))

    configs: list[tuple[str, ...]] = []
    accident_ids: list[Any] = []
    for _, row in df.iterrows():
        cfg = _config_tuple(row, topic_cols)
        if not cfg:
            continue
        configs.append(cfg)
        accident_ids.append(row[accident_id_col])

    paths = enumerate_macro_paths(
        bn_model,
        variable_macro_map,
        min_macros=min_macros,
        max_path_len=max_path_len,
        max_paths=max_paths,
    )

    diag: dict[str, Any] = {
        "n_paths_dag": len(paths),
        "min_macros": min_macros,
        "n_accidents_with_topics": len(configs),
        "n_accidents_min_macros_cooc": _accidents_with_min_macros(configs, min_macros),
        "n_accidents_full_macro_cooc": _accidents_with_min_macros(configs, 4),
        "min_support_applied": min_support,
        "support_fallback": False,
    }

    rows: list[dict[str, Any]] = []
    for path in paths:
        support = _support_for_path(path, configs)
        ordered = sort_topics_by_macro_order(path)
        matching = [i for i, cfg in enumerate(configs) if set(path).issubset(set(cfg))]
        rep_acc = [accident_ids[i] for i in matching[:5]]
        rep_sent = _representative_sentences(rep_acc, metadata_unit, text_col)
        rows.append(
            {
                "scenario_id": 0,
                "path_nodes": " -> ".join(path),
                "macro_path": macro_path_from_topics(ordered),
                "topics_present": " + ".join(ordered),
                "support": support,
                "representative_accidents": " | ".join(map(str, rep_acc)),
                "representative_sentences": " || ".join(rep_sent[:5]),
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r["support"]),
            len(str(r["macro_path"]).split(" -> ")),
            len(str(r["path_nodes"]).split(" -> ")),
        )
    )
    diag["n_paths_support_ge_1"] = sum(1 for r in rows if int(r["support"]) >= 1)

    filtered = [r for r in rows if int(r["support"]) >= min_support]
    if not filtered and rows:
        diag["support_fallback"] = True
        diag["min_support_applied"] = 1
        filtered = [r for r in rows if int(r["support"]) >= 1]
    if not filtered and rows:
        diag["support_fallback"] = True
        diag["min_support_applied"] = 0
        filtered = rows

    for idx, row in enumerate(filtered[:top_n]):
        row["scenario_id"] = idx

    return pd.DataFrame(filtered[:top_n]), diag
