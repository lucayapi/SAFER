"""Empirical recurrent scenario mining on the accident-factor matrix.

Selection is driven by empirical recurrence (accident count) and closed-pattern
redundancy reduction. Confidence, lift and BN path support are descriptive only.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scenario_pipeline import ROLES, StructuralEMResult, _edge_conditional_contrast_signed, _theme_label_map

RECURRENCE_SORT_COLS = [
    "scenario_accident_count",
    "scenario_support",
    "lift",
    "confidence",
    "scenario_id",
]
RECURRENCE_SORT_ASCENDING = [False, False, False, False, True]

ARTICLE_COLUMNS = [
    "scenario_id",
    "upstream_factor_ids",
    "upstream_labels",
    "upstream_roles",
    "B_factor_id",
    "B_label",
    "C_factor_id",
    "C_label",
    "scenario_accident_count",
    "scenario_support",
    "confidence",
    "lift",
    "upstream_to_B_confidence",
    "upstream_to_B_lift",
    "positive_bn_path_support",
    "stable_positive_bn_path_support",
    "path_bootstrap_frequencies",
]


def _scenario_config(config: Mapping[str, Any]) -> dict[str, Any]:
    bn = dict(config.get("bayesian_networks", {}))
    mining = dict(config.get("scenario_mining", {}))
    stable = mining.get("bn_display_bootstrap_threshold", bn.get("bn_display_bootstrap_threshold", 0.60))
    n_article = int(mining.get("n_article_scenarios", mining.get("article_max_scenarios", 6)))
    return {
        "min_accident_count": int(mining.get("scenario_min_accident_count", 5)),
        "max_upstream": int(mining.get("max_upstream_factors_per_scenario", 2)),
        "stable_threshold": float(stable),
        "n_article_scenarios": n_article,
        "threshold_sensitivity": [int(value) for value in mining.get("threshold_sensitivity_counts", [3, 5, 8, 10])],
        "rare_high_lift_max_count": int(mining.get("rare_high_lift_max_count", 5)),
        "rare_high_lift_min_lift": float(mining.get("rare_high_lift_min_lift", 5.0)),
    }


def _upstream_combinations(upstream_nodes: Sequence[str], max_upstream: int) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for size in range(1, max_upstream + 1):
        combos.extend(itertools.combinations(upstream_nodes, size))
    return combos


def _pattern_key(upstream: Sequence[str], b_factor: str, c_factor: str) -> tuple[tuple[str, ...], str, str]:
    return (tuple(sorted(upstream)), b_factor, c_factor)


def _count_all_present(matrix_values: np.ndarray) -> int:
    return int(matrix_values.all(axis=1).sum())


def _format_edge_list(edges: Sequence[tuple[str, str, float]]) -> str:
    return " | ".join(f"{parent}->{child}:{frequency:.2f}" for parent, child, frequency in edges)


def _path_bootstrap_frequencies(
    upstream: Sequence[str],
    b_factor: str,
    c_factor: str,
    bootstrap_freq: Mapping[tuple[str, str], float],
) -> str:
    parts = [
        f"{u}->{b_factor}:{bootstrap_freq.get((u, b_factor), float('nan')):.2f}"
        for u in upstream
    ]
    parts.append(f"{b_factor}->{c_factor}:{bootstrap_freq.get((b_factor, c_factor), float('nan')):.2f}")
    return " | ".join(parts)


def _positive_bn_path_support(
    upstream: Sequence[str],
    b_factor: str,
    c_factor: str,
    positive_edges: set[tuple[str, str]],
    stable_positive_edges: set[tuple[str, str]],
    *,
    stable: bool,
) -> str:
    edge_set = stable_positive_edges if stable else positive_edges
    upstream_links = [(u, b_factor) for u in upstream if (u, b_factor) in edge_set]
    b_c_ok = (b_factor, c_factor) in edge_set
    if upstream_links and b_c_ok:
        return "COMPLETE"
    if upstream_links or b_c_ok:
        return "PARTIAL"
    return "NONE"


def _sort_by_recurrence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(RECURRENCE_SORT_COLS, ascending=RECURRENCE_SORT_ASCENDING).reset_index(drop=True)


def _select_article_scenarios(recurrent_closed: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Highest-recurrence closed scenarios for the main-text table."""
    if recurrent_closed.empty:
        return recurrent_closed.copy()
    ordered = _sort_by_recurrence(recurrent_closed)
    return ordered.head(int(cfg["n_article_scenarios"])).copy()


def _article_export_table(article: pd.DataFrame) -> pd.DataFrame:
    if article.empty:
        return article.copy()
    columns = [column for column in ARTICLE_COLUMNS if column in article.columns]
    return article[columns].copy()


def mine_recurrent_scenarios(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    result: StructuralEMResult,
    bootstrap: pd.DataFrame | None,
    theme_dictionary: pd.DataFrame | None,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    cfg = _scenario_config(config)
    label_map = _theme_label_map(theme_dictionary)
    nodes = [column for column in matrix.columns if column != "accident_id"]
    data = matrix[nodes].to_numpy(dtype=np.int8)
    node_index = {node: index for index, node in enumerate(nodes)}
    n_accidents = len(matrix)

    upstream_nodes = sorted(node for node in nodes if roles[node] in {"A0", "A1"})
    b_nodes = sorted(node for node in nodes if roles[node] == "B")
    c_nodes = sorted(node for node in nodes if roles[node] == "C")
    n_admissible = len(_upstream_combinations(upstream_nodes, cfg["max_upstream"])) * len(b_nodes) * len(c_nodes)

    learned_edges = set(result.edges)
    bootstrap_freq: dict[tuple[str, str], float] = {}
    if bootstrap is not None and not bootstrap.empty:
        bootstrap_freq = {
            (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
            for _, row in bootstrap.iterrows()
        }

    signed_lookup = {
        (parent, child): _edge_conditional_contrast_signed(result, parent, child)
        for parent, child in result.edges
    }
    positive_edges = {edge for edge, signed in signed_lookup.items() if signed > 0}
    stable_positive_edges = {
        edge for edge in positive_edges
        if bootstrap_freq.get(edge, 0.0) >= cfg["stable_threshold"]
    }

    prevalences = {node: float(data[:, node_index[node]].mean()) for node in nodes}
    rows: list[dict[str, Any]] = []
    scenario_counter = 0

    for upstream in _upstream_combinations(upstream_nodes, cfg["max_upstream"]):
        upstream_indices = [node_index[node] for node in upstream]
        upstream_matrix = data[:, upstream_indices] if upstream_indices else data[:, :0]
        upstream_count_base = _count_all_present(upstream_matrix) if len(upstream) else n_accidents
        for b_factor in b_nodes:
            b_index = node_index[b_factor]
            p_b = prevalences[b_factor]
            for c_factor in c_nodes:
                c_index = node_index[c_factor]
                p_c = prevalences[c_factor]
                scenario_indices = upstream_indices + [b_index, c_index]
                scenario_matrix = data[:, scenario_indices]
                scenario_count = _count_all_present(scenario_matrix)
                antecedent_count = (
                    _count_all_present(data[:, upstream_indices + [b_index]])
                    if upstream_indices else int(data[:, b_index].sum())
                )
                support = scenario_count / n_accidents if n_accidents else 0.0
                antecedent_support = antecedent_count / n_accidents if n_accidents else 0.0
                confidence = scenario_count / antecedent_count if antecedent_count > 0 else 0.0
                lift = confidence / p_c if p_c > 0 else np.nan
                leverage = support - antecedent_support * p_c
                upstream_only_count = upstream_count_base
                ub_confidence = antecedent_count / upstream_only_count if upstream_only_count > 0 else 0.0
                ub_lift = ub_confidence / p_b if p_b > 0 else np.nan

                scenario_counter += 1
                rows.append({
                    "scenario_id": f"SC_{scenario_counter:05d}",
                    "upstream_factor_ids": " | ".join(upstream),
                    "upstream_labels": " | ".join(label_map.get(node, node) for node in upstream),
                    "upstream_roles": " | ".join(roles[node] for node in upstream),
                    "n_upstream_factors": len(upstream),
                    "B_factor_id": b_factor,
                    "B_label": label_map.get(b_factor, b_factor),
                    "C_factor_id": c_factor,
                    "C_label": label_map.get(c_factor, c_factor),
                    "scenario_accident_count": scenario_count,
                    "scenario_support": support,
                    "antecedent_count": antecedent_count,
                    "antecedent_support": antecedent_support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "upstream_count": upstream_only_count,
                    "upstream_to_B_confidence": ub_confidence,
                    "upstream_to_B_lift": ub_lift,
                    "_pattern_key": _pattern_key(upstream, b_factor, c_factor),
                })

    candidates = pd.DataFrame(rows)
    empty = pd.DataFrame()
    if candidates.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_dir / "scenario_candidates_all.csv", index=False)
        empty.to_csv(output_dir / "recurrent_scenarios_all.csv", index=False)
        empty.to_csv(output_dir / "scenarios_article.csv", index=False)
        return {
            "candidates_all": empty,
            "recurrent_all": empty,
            "article": empty,
            "article_full": empty,
            "prevention": empty,
            "threshold_summary": empty,
            "n_admissible": n_admissible,
            "n_observed": 0,
        }

    candidates = candidates.drop_duplicates(subset=["upstream_factor_ids", "B_factor_id", "C_factor_id"]).reset_index(drop=True)

    count_lookup = candidates.set_index("_pattern_key")["scenario_accident_count"].to_dict()
    closed_flags = []
    for _, row in candidates.iterrows():
        upstream = tuple(part for part in str(row["upstream_factor_ids"]).split(" | ") if part)
        b_factor = str(row["B_factor_id"])
        c_factor = str(row["C_factor_id"])
        count = int(row["scenario_accident_count"])
        is_closed = True
        for extra in upstream_nodes:
            if extra in upstream:
                continue
            if len(upstream) >= cfg["max_upstream"]:
                continue
            superset = tuple(sorted((*upstream, extra)))
            sup_count = count_lookup.get(_pattern_key(superset, b_factor, c_factor))
            if sup_count is not None and int(sup_count) == count:
                is_closed = False
                break
        closed_flags.append(is_closed)
    candidates["is_closed_pattern"] = closed_flags
    candidates["rare_high_lift_flag"] = (
        candidates["scenario_accident_count"].le(cfg["rare_high_lift_max_count"])
        & candidates["lift"].ge(cfg["rare_high_lift_min_lift"])
    )

    bn_rows = []
    for _, row in candidates.iterrows():
        upstream = tuple(part for part in str(row["upstream_factor_ids"]).split(" | ") if part)
        b_factor = str(row["B_factor_id"])
        c_factor = str(row["C_factor_id"])
        positive = _positive_bn_path_support(
            upstream, b_factor, c_factor, positive_edges, stable_positive_edges, stable=False,
        )
        stable_positive = _positive_bn_path_support(
            upstream, b_factor, c_factor, positive_edges, stable_positive_edges, stable=True,
        )
        bn_rows.append({
            "positive_bn_path_support": positive,
            "stable_positive_bn_path_support": stable_positive,
            "path_bootstrap_frequencies": _path_bootstrap_frequencies(upstream, b_factor, c_factor, bootstrap_freq),
            "upstream_B_edges": " | ".join(
                f"{u}->{b_factor}" for u in upstream if (u, b_factor) in learned_edges
            ),
            "upstream_B_bootstrap_frequencies": _format_edge_list([
                (u, b_factor, bootstrap_freq.get((u, b_factor), np.nan))
                for u in upstream if (u, b_factor) in learned_edges
            ]),
            "B_C_direct_edge": (b_factor, c_factor) in learned_edges,
            "B_C_bootstrap_frequency": bootstrap_freq.get((b_factor, c_factor), np.nan),
        })
    candidates = pd.concat([candidates.reset_index(drop=True), pd.DataFrame(bn_rows)], axis=1)
    candidates = candidates.drop(columns=["_pattern_key"], errors="ignore")

    n_observed = int((candidates["scenario_accident_count"] > 0).sum())
    threshold_rows = []
    for threshold in cfg["threshold_sensitivity"]:
        subset = candidates[candidates["scenario_accident_count"] >= threshold]
        closed_subset = subset[subset["is_closed_pattern"]]
        retained = (len(subset) / n_observed) if n_observed else 0.0
        threshold_rows.append({
            "threshold": threshold,
            "n_candidates": int(len(subset)),
            "n_closed_patterns": int(len(closed_subset)),
            "percentage_of_observed_candidates_retained": retained,
        })
    threshold_summary = pd.DataFrame(threshold_rows)

    recurrent_all = _sort_by_recurrence(
        candidates[
            (candidates["scenario_accident_count"] >= cfg["min_accident_count"])
            & candidates["is_closed_pattern"]
        ].copy()
    )
    article_full = _select_article_scenarios(recurrent_all, cfg)
    article = _article_export_table(article_full)
    prevention = article_full[[
        "scenario_id", "upstream_labels", "B_label", "C_label",
        "scenario_accident_count", "scenario_support", "confidence", "lift",
        "positive_bn_path_support", "stable_positive_bn_path_support",
    ]].copy() if not article_full.empty else empty.copy()

    assert "is_pareto" not in candidates.columns
    assert "is_pareto" not in recurrent_all.columns
    assert "is_pareto" not in article.columns

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "scenario_candidates_all.csv", index=False)
    recurrent_all.to_csv(output_dir / "recurrent_scenarios_all.csv", index=False)
    article.to_csv(output_dir / "scenarios_article.csv", index=False)
    article_full.to_csv(output_dir / "scenarios_article_extended.csv", index=False)
    prevention.to_csv(output_dir / "scenario_prevention_summary.csv", index=False)
    threshold_summary.to_csv(output_dir / "scenario_threshold_summary.csv", index=False)

    # Drop legacy Pareto artifact if a previous run left it behind.
    legacy_pareto = output_dir / "scenario_pareto.csv"
    if legacy_pareto.is_file():
        legacy_pareto.unlink()

    return {
        "candidates_all": candidates,
        "recurrent_all": recurrent_all,
        "article": article,
        "article_full": article_full,
        "prevention": prevention,
        "threshold_summary": threshold_summary,
        "n_admissible": n_admissible,
        "n_observed": n_observed,
    }


def build_scenario_article_table(article: pd.DataFrame) -> pd.DataFrame:
    if article.empty:
        return article
    rows = []
    for index, row in article.iterrows():
        stable = str(row.get("stable_positive_bn_path_support", ""))
        positive = str(row.get("positive_bn_path_support", ""))
        bn_bits = [f"Stable positive: {stable}", f"Positive: {positive}"]
        if "path_bootstrap_frequencies" in row:
            bn_bits.append(str(row["path_bootstrap_frequencies"]))
        rows.append({
            "Scenario": f"S{index + 1}",
            "Upstream": row["upstream_labels"],
            "B": row["B_label"],
            "C": row["C_label"],
            "Accidents_n": int(row["scenario_accident_count"]),
            "Support": f"{100.0 * float(row['scenario_support']):.2f}%",
            "Confidence": f"{100.0 * float(row['confidence']):.1f}%",
            "Lift": f"{float(row['lift']):.1f}x",
            "BN_support": " | ".join(bn_bits),
        })
    return pd.DataFrame(rows)


def write_scenarios_latex_csv(
    article: pd.DataFrame,
    result: StructuralEMResult,
    bootstrap: pd.DataFrame | None,
    output_dir: Path,
    *,
    stable_threshold: float = 0.60,
) -> pd.DataFrame:
    signed_lookup = {
        (parent, child): _edge_conditional_contrast_signed(result, parent, child)
        for parent, child in result.edges
    }
    bootstrap_freq: dict[tuple[str, str], float] = {}
    if bootstrap is not None and not bootstrap.empty:
        bootstrap_freq = {
            (str(row["parent"]), str(row["child"])): float(row["selection_frequency"])
            for _, row in bootstrap.iterrows()
        }
    stable_positive = {
        edge for edge, signed in signed_lookup.items()
        if signed > 0 and bootstrap_freq.get(edge, 0.0) >= stable_threshold
    }
    rows = []
    for _, row in article.iterrows():
        upstream = [part for part in str(row["upstream_factor_ids"]).split(" | ") if part]
        b_factor = str(row["B_factor_id"])
        c_factor = str(row["C_factor_id"])
        sequence_parts = []
        if len(upstream) == 1:
            sequence_parts.append(upstream[0])
        elif len(upstream) > 1:
            sequence_parts.append("[" + " + ".join(upstream) + "]")
        sequence_parts.extend([b_factor, c_factor])
        display_sequence = " > ".join(sequence_parts)
        link_parts = []
        ordered_nodes = upstream + [b_factor, c_factor]
        for left, right in zip(ordered_nodes, ordered_nodes[1:]):
            link_type = "SOLID" if (left, right) in stable_positive else "DASHED"
            freq = bootstrap_freq.get((left, right), 0.0)
            link_parts.append(f"{left}->{right}:{link_type}:{freq:.2f}")
        rows.append({
            "scenario_id": row["scenario_id"],
            "upstream_factor_ids": row["upstream_factor_ids"],
            "upstream_labels": row["upstream_labels"],
            "B_factor_id": b_factor,
            "B_label": row["B_label"],
            "C_factor_id": c_factor,
            "C_label": row["C_label"],
            "scenario_accident_count": int(row["scenario_accident_count"]),
            "scenario_support": float(row["scenario_support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "display_sequence": display_sequence,
            "display_links": " | ".join(link_parts),
        })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "scenarios_latex.csv", index=False)
    return frame


def assert_scenario_mining_outputs(
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    result: StructuralEMResult,
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    recurrent_all: pd.DataFrame | None = None,
    article: pd.DataFrame | None = None,
) -> None:
    cfg = _scenario_config(config)
    nodes = [column for column in matrix.columns if column != "accident_id"]
    n_accidents = len(matrix)
    role_counts = {role: sum(1 for node in nodes if roles[node] == role) for role in ROLES}
    assert result.n_states == 1
    assert "Z" not in result.nodes
    assert "is_pareto" not in candidates.columns

    expected = config.get("expected_inventory", {})
    dataset_id = str(config.get("data", {}).get("dataset_id", config.get("dataset_id", "")))
    dataset_expected = expected.get(dataset_id, {})
    if dataset_expected:
        assert n_accidents == int(dataset_expected["n_accidents"])
        assert len(nodes) == int(dataset_expected["n_factors"])
        for role, count in dataset_expected.get("role_counts", {}).items():
            assert role_counts[role] == int(count)

    if not candidates.empty:
        for _, row in candidates.iterrows():
            assert 1 <= int(row["n_upstream_factors"]) <= cfg["max_upstream"]
            assert 0.0 <= float(row["scenario_support"]) <= 1.0
            assert 0.0 <= float(row["confidence"]) <= 1.0
            assert float(row["lift"]) >= 0.0 or np.isnan(float(row["lift"]))
            expected_support = float(row["scenario_accident_count"]) / n_accidents
            assert abs(float(row["scenario_support"]) - expected_support) < 1e-9

    if recurrent_all is not None and not recurrent_all.empty:
        assert "is_pareto" not in recurrent_all.columns
        assert (recurrent_all["scenario_accident_count"] >= cfg["min_accident_count"]).all()
        assert recurrent_all["is_closed_pattern"].all()
        ordered = _sort_by_recurrence(recurrent_all)
        assert list(recurrent_all["scenario_id"]) == list(ordered["scenario_id"])
        counts = recurrent_all["scenario_accident_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    if article is not None and not article.empty:
        assert "is_pareto" not in article.columns
        assert (article["scenario_accident_count"] >= cfg["min_accident_count"]).all()
        if recurrent_all is not None and not recurrent_all.empty:
            assert set(article["scenario_id"]).issubset(set(recurrent_all["scenario_id"]))
        article_counts = article["scenario_accident_count"].tolist()
        assert article_counts == sorted(article_counts, reverse=True)
        for i in range(len(article_counts) - 1):
            assert article_counts[i] >= article_counts[i + 1]

    signed_lookup = {
        (parent, child): _edge_conditional_contrast_signed(result, parent, child)
        for parent, child in result.edges
    }
    positive_edges = {edge for edge, signed in signed_lookup.items() if signed > 0}
    for _, row in candidates.iterrows():
        upstream = tuple(part for part in str(row["upstream_factor_ids"]).split(" | ") if part)
        b_factor = str(row["B_factor_id"])
        c_factor = str(row["C_factor_id"])
        expected_positive = _positive_bn_path_support(
            upstream, b_factor, c_factor, positive_edges, positive_edges, stable=False,
        )
        assert row["positive_bn_path_support"] == expected_positive

    _ = role_counts
