"""LaTeX-ready CSV exports for recurrent accident scenarios (Abdat-style tables)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scenario_pipeline import ROLES, StructuralEMResult


def _split_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text == "—":
        return []
    for separator in (" | ", ";", "|"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _join_cell(values: Sequence[str]) -> str:
    return " | ".join(values)


def _scenario_heading(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for column in ("A0_labels", "A1_labels", "B_labels", "C_labels"):
        labels = _split_cell(row.get(column, ""))
        if labels:
            parts.append(labels[0])
    return " – ".join(parts)


def _positive_factors_by_role(row: Mapping[str, Any], roles: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {role: [] for role in ROLES}
    for role in ROLES:
        for column in (f"{role}_factor_ids", f"{role}_factors"):
            if column in row and str(row.get(column, "")).strip():
                grouped[role] = _split_cell(row.get(column, ""))
                break
    return grouped


def _flat_display_sequence(grouped: Mapping[str, list[str]]) -> list[str]:
    sequence: list[str] = []
    for role in ROLES:
        sequence.extend(grouped[role])
    return sequence


def _format_display_sequence(grouped: Mapping[str, list[str]]) -> str:
    blocks: list[str] = []
    for role in ROLES:
        factors = grouped[role]
        if not factors:
            continue
        if len(factors) == 1:
            blocks.append(factors[0])
        else:
            blocks.append("[" + " + ".join(factors) + "]")
    return " > ".join(blocks)


def _link_type(source: str, target: str, learned_edges: set[tuple[str, str]]) -> str:
    return "SOLID" if (source, target) in learned_edges else "DASHED"


def _format_display_links(sequence: Sequence[str], learned_edges: set[tuple[str, str]]) -> str:
    if len(sequence) < 2:
        return ""
    links = []
    for index in range(len(sequence) - 1):
        source, target = sequence[index], sequence[index + 1]
        links.append(f"{source}->{target}:{_link_type(source, target, learned_edges)}")
    return " | ".join(links)


def _hard_assignment_count(responsibilities: np.ndarray, state: int) -> int:
    return int((np.argmax(responsibilities, axis=1) == state).sum())


def _coobservation_stats(
    matrix: pd.DataFrame,
    responsibilities: np.ndarray,
    state: int,
    source: str,
    target: str,
) -> tuple[bool, int, float]:
    hard_mask = np.argmax(responsibilities, axis=1) == state
    co_mask = (matrix[source].to_numpy(dtype=int) == 1) & (matrix[target].to_numpy(dtype=int) == 1)
    hard_co = hard_mask & co_mask
    family_mass = float(responsibilities[:, state].sum())
    weighted = float((responsibilities[:, state] * co_mask).sum() / max(family_mass, 1e-12))
    return bool(hard_co.any()), int(hard_co.sum()), weighted


def write_recurrent_scenarios_latex(
    scenarios: pd.DataFrame,
    result: StructuralEMResult,
    matrix: pd.DataFrame,
    roles: Mapping[str, str],
    responsibilities: np.ndarray,
    label_map: Mapping[str, str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write ``recurrent_scenarios_latex.csv`` and ``recurrent_scenario_links.csv``."""

    learned_edges = set(result.edges)
    latex_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for _, row in scenarios.iterrows():
        family_id = int(row["family_id"])
        state = family_id - 1
        grouped = _positive_factors_by_role(row, roles)
        flat_sequence = _flat_display_sequence(grouped)
        scenario_id = str(row.get("scenario_id", f"S{family_id}"))
        if scenario_id == "global":
            scenario_id = "S1"

        def role_columns(role: str, label_key: str) -> tuple[str, str]:
            factors = grouped[role]
            if not factors:
                return "", ""
            labels_raw = _split_cell(row.get(label_key, ""))
            labels = [
                labels_raw[index] if index < len(labels_raw) else label_map.get(factor, factor)
                for index, factor in enumerate(factors)
            ]
            return _join_cell(factors), _join_cell(labels)

        a0_ids, a0_labels = role_columns("A0", "A0_labels")
        a1_ids, a1_labels = role_columns("A1", "A1_labels")
        b_ids, b_labels = role_columns("B", "B_labels")
        c_ids, c_labels = role_columns("C", "C_labels")

        mpe_probability = float(row.get("mpe_probability", np.nan))
        mpe_log_probability = float(math.log(max(mpe_probability, 1e-300))) if np.isfinite(mpe_probability) else np.nan

        latex_rows.append({
            "scenario_id": scenario_id,
            "family_id": family_id,
            "family_weight": float(row["omega"]),
            "N_eff": float(row["N_eff"]),
            "hard_assignment_count": _hard_assignment_count(responsibilities, state),
            "scenario_heading": _scenario_heading(row),
            "A0_factor_ids": a0_ids,
            "A0_labels": a0_labels,
            "A1_factor_ids": a1_ids,
            "A1_labels": a1_labels,
            "B_factor_ids": b_ids,
            "B_labels": b_labels,
            "C_factor_ids": c_ids,
            "C_labels": c_labels,
            "n_positive_factors": len(flat_sequence),
            "family_support": float(row["family_positive_support"]),
            "global_support": float(row["global_positive_support"]),
            "support_enrichment": float(row["support_enrichment_ratio"]),
            "exact_observed_pattern": bool(row.get("prototype_exact_mpe_match", False)),
            "n_exact_matching_accidents": int(row.get("n_exact_matching_accidents", 0)),
            "prototype_accident_id": str(row.get("prototype_accident_id", "")),
            "prototype_posterior_membership": float(row.get("prototype_posterior_membership", np.nan)),
            "mpe_probability": mpe_probability,
            "mpe_log_probability": mpe_log_probability,
            "display_sequence": _format_display_sequence(grouped),
            "display_links": _format_display_links(flat_sequence, learned_edges),
        })

        for index in range(len(flat_sequence) - 1):
            source, target = flat_sequence[index], flat_sequence[index + 1]
            learned = (source, target) in learned_edges
            display_type = "SOLID" if learned else "DASHED"
            co_in_family, co_count, co_weighted = _coobservation_stats(matrix, responsibilities, state, source, target)
            link_rows.append({
                "scenario_id": scenario_id,
                "family_id": family_id,
                "source_factor": source,
                "source_label": label_map.get(source, source),
                "source_role": roles[source],
                "target_factor": target,
                "target_label": label_map.get(target, target),
                "target_role": roles[target],
                "learned_bn_edge": learned,
                "display_link_type": display_type,
                "coobserved_in_family": co_in_family,
                "coobserved_accident_count": co_count,
                "posterior_weighted_coobserved_support": co_weighted,
            })

    latex = pd.DataFrame(latex_rows)
    links = pd.DataFrame(link_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    latex.to_csv(output_dir / "recurrent_scenarios_latex.csv", index=False)
    links.to_csv(output_dir / "recurrent_scenario_links.csv", index=False)
    return latex, links


def write_recurrent_scenarios_article(scenarios: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Short manuscript table (decimals; A1 empty shown as —)."""

    article_rows = []
    for _, row in scenarios.iterrows():
        scenario_id = str(row.get("scenario_id", f"S{int(row['family_id'])}"))
        if scenario_id == "global":
            scenario_id = "S1"
        a0 = _join_cell(_split_cell(row.get("A0_labels", ""))) or "—"
        a1 = _join_cell(_split_cell(row.get("A1_labels", ""))) or "—"
        article_rows.append({
            "Scenario": scenario_id,
            "Family_weight": float(row["omega"]),
            "N_eff": float(row["N_eff"]),
            "Scenario_heading": _scenario_heading(row),
            "A0": a0,
            "A1": a1,
            "B": _join_cell(_split_cell(row.get("B_labels", ""))) or "—",
            "C": _join_cell(_split_cell(row.get("C_labels", ""))) or "—",
            "Family_support": float(row["family_positive_support"]),
            "Global_support": float(row["global_positive_support"]),
            "Enrichment": float(row["support_enrichment_ratio"]),
            "Exact_observed_pattern": bool(row.get("prototype_exact_mpe_match", False)),
        })
    article = pd.DataFrame(article_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    article.to_csv(output_dir / "recurrent_scenarios_article.csv", index=False)
    return article
