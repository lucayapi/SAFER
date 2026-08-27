"""Blinded semantic evaluation of Pareto-supported partitions.

Two named evaluators (OpenAI models) score anonymized factor samples on
coherence, distinctiveness and prevention relevance. Their aggregated scores
select the final configuration when the Pareto front has more than one point.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

SEMANTIC_DIMENSIONS = ("coherence", "distinctiveness", "prevention_relevance")
EVALUATOR_IDS = ("evaluator_1", "evaluator_2")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _semantic_evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    validation = dict(config.get("validation") or {})
    values = dict(validation.get("semantic_evaluation") or config.get("semantic_evaluation") or {})
    values.setdefault("enabled", True)
    values.setdefault("units_per_factor", 9)
    values.setdefault("random_state", int(validation.get("random_state", config.get("random_state", 42))))
    values.setdefault("require_api_key", True)
    evaluator_1 = dict(values.get("evaluator_1") or {})
    evaluator_2 = dict(values.get("evaluator_2") or {})
    evaluator_1.setdefault("model", "gpt-5.6-terra")
    evaluator_2.setdefault("model", "gpt-5.6-luna")
    values["evaluator_1"] = evaluator_1
    values["evaluator_2"] = evaluator_2
    return values


def mark_pareto_front(
    frame: pd.DataFrame,
    *,
    stability_col: str = "stability",
    dbcv_col: str = "dbcv_umap",
) -> pd.DataFrame:
    """Flag non-dominated configurations that maximize stability and DBCV."""
    result = frame.copy()
    result["on_pareto"] = False
    result["selected"] = False
    if result.empty or stability_col not in result.columns or dbcv_col not in result.columns:
        return result
    usable = result[stability_col].notna() & result[dbcv_col].notna()
    if not usable.any():
        return result
    pool = result.loc[usable]
    indices = pool.index.to_list()
    stabilities = pool[stability_col].astype(float).to_numpy()
    dbcvs = pool[dbcv_col].astype(float).to_numpy()
    on_pareto = np.ones(len(indices), dtype=bool)
    for i in range(len(indices)):
        if not on_pareto[i]:
            continue
        dominated = (
            (stabilities >= stabilities[i])
            & (dbcvs >= dbcvs[i])
            & ((stabilities > stabilities[i]) | (dbcvs > dbcvs[i]))
        )
        if dominated.any():
            on_pareto[i] = False
    result.loc[np.asarray(indices)[on_pareto], "on_pareto"] = True
    return result


def _stratum_edges(strengths: np.ndarray) -> tuple[float, float]:
    if len(strengths) == 0:
        return 0.33, 0.66
    low, high = np.quantile(strengths, [1 / 3, 2 / 3])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return 0.33, 0.66
    return float(low), float(high)


def sample_membership_stratified_units(
    factor_frame: pd.DataFrame,
    *,
    units_per_factor: int,
    rng: np.random.Generator,
    text_col: str = "sentence",
    strength_col: str = "membership_strength",
    accident_col: str = "accident_id",
) -> list[dict[str, Any]]:
    """Sample factual units stratified by membership strength, preferring distinct accidents."""
    if factor_frame.empty or units_per_factor <= 0:
        return []
    work = factor_frame.copy().reset_index(drop=True)
    positive = work.loc[work[strength_col].astype(float) > 0].reset_index(drop=True)
    if not positive.empty:
        work = positive
    strengths = work[strength_col].astype(float).to_numpy()
    low_cut, high_cut = _stratum_edges(strengths)
    stratum_masks = {
        "high": strengths >= high_cut,
        "mid": (strengths >= low_cut) & (strengths < high_cut),
        "low": strengths < low_cut,
    }
    per_stratum = max(1, int(np.ceil(units_per_factor / 3)))
    selected_positions: list[int] = []
    used_accidents: set[str] = set()

    def _pick(mask: np.ndarray, n_needed: int, *, allow_accident_reuse: bool) -> None:
        nonlocal selected_positions, used_accidents
        if n_needed <= 0 or not mask.any():
            return
        candidates = np.flatnonzero(mask)
        order = rng.permutation(len(candidates))
        for offset in order:
            if len(selected_positions) >= units_per_factor or n_needed <= 0:
                return
            position = int(candidates[int(offset)])
            if position in selected_positions:
                continue
            accident = str(work.iloc[position][accident_col])
            if (
                not allow_accident_reuse
                and accident in used_accidents
                and work.loc[mask, accident_col].nunique() > len(used_accidents)
            ):
                continue
            selected_positions.append(position)
            used_accidents.add(accident)
            n_needed -= 1

    for name in ("high", "mid", "low"):
        remaining = units_per_factor - len(selected_positions)
        if remaining <= 0:
            break
        _pick(stratum_masks[name], min(per_stratum, remaining), allow_accident_reuse=False)
        remaining = units_per_factor - len(selected_positions)
        if remaining > 0:
            _pick(stratum_masks[name], min(per_stratum, remaining), allow_accident_reuse=True)
    remaining = units_per_factor - len(selected_positions)
    if remaining > 0:
        leftover = np.ones(len(work), dtype=bool)
        leftover[selected_positions] = False
        _pick(leftover, remaining, allow_accident_reuse=True)

    samples = []
    for position in selected_positions[:units_per_factor]:
        strength = float(work.iloc[position][strength_col])
        samples.append({
            "text": str(work.iloc[position][text_col]),
            "membership_stratum": (
                "high" if strength >= high_cut else "mid" if strength >= low_cut else "low"
            ),
        })
    return samples

def _anonymous_candidate_ids(configuration_ids: Sequence[str], *, seed: int, evaluator_id: str) -> dict[str, str]:
    ordered = list(configuration_ids)
    rng = np.random.default_rng(
        int(hashlib.sha256(f"{seed}:{evaluator_id}:candidates".encode()).hexdigest()[:16], 16) % (2**32)
    )
    perm = rng.permutation(len(ordered))
    return {
        str(ordered[int(index)]): f"Candidate-{rank + 1:02d}"
        for rank, index in enumerate(perm)
    }


def build_evaluation_package_for_configuration(
    *,
    role: str,
    configuration_id: str,
    units: pd.DataFrame,
    labels: np.ndarray,
    strengths: np.ndarray,
    units_per_factor: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build one anonymizable package (true IDs kept for storage only)."""
    role_units = units.loc[units["_role"].eq(role)].reset_index(drop=True)
    if len(role_units) != len(labels):
        raise ValueError(f"Label/unit mismatch for {role}/{configuration_id}")
    frame = role_units[["_accident_id", "_fact_id", "_text"]].copy()
    frame.rename(
        columns={"_accident_id": "accident_id", "_fact_id": "fact_id", "_text": "sentence"},
        inplace=True,
    )
    frame["membership_strength"] = np.asarray(strengths, dtype=float)
    frame["cluster_label"] = np.asarray(labels, dtype=int)
    factors = []
    for cluster_label in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
        subset = frame.loc[frame["cluster_label"].eq(cluster_label)]
        samples = sample_membership_stratified_units(
            subset,
            units_per_factor=units_per_factor,
            rng=rng,
        )
        factors.append({
            "cluster_label": int(cluster_label),
            "topic_id": f"{role}_{int(cluster_label):03d}",
            "n_units": int(len(subset)),
            "n_accidents": int(subset["accident_id"].nunique()),
            "samples": samples,
        })
    return {
        "role": role,
        "configuration_id": configuration_id,
        "n_factors": len(factors),
        "factors": factors,
    }


def build_blinded_prompt(
    *,
    role: str,
    anonymous_candidate_id: str,
    factors: Sequence[Mapping[str, Any]],
) -> str:
    role_meaning = {
        "A0": "work context before the accident",
        "A1": "adverse condition or latent failure",
        "B": "event or deviation",
        "C": "consequence or injury",
    }.get(role, role)
    blocks = [
        "You evaluate candidate semantic factors for occupational accident analysis.",
        "Score only the displayed factual units. Do not invent external context.",
        f"Accident-process role: {role} ({role_meaning}).",
        f"Candidate identifier: {anonymous_candidate_id}.",
        "For each factor, rate on a 1-5 ordinal scale:",
        "1) coherence: units describe one identifiable accident-related phenomenon",
        "2) distinctiveness: factor is substantively distinguishable from the other factors shown",
        "3) prevention_relevance: retaining the distinction is useful for analysis or prevention",
        "Return strict JSON:",
        '{"candidate_id":"...","factors":[{"factor_id":"Factor-01","coherence":1,"distinctiveness":1,'
        '"prevention_relevance":1,"justification":"short"}]}',
        "",
    ]
    for index, factor in enumerate(factors, start=1):
        factor_id = f"Factor-{index:02d}"
        texts = [str(item.get("text", "")).strip() for item in factor.get("samples", [])]
        texts = [text for text in texts if text]
        joined = "\n".join(f"- {text}" for text in texts) if texts else "- (no units)"
        blocks.append(f"{factor_id} (n_units={factor.get('n_units', '?')}):")
        blocks.append(joined)
        blocks.append("")
    return "\n".join(blocks)


def _parse_json_payload(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def parse_evaluator_response(
    raw_text: str,
    *,
    expected_factor_ids: Sequence[str],
) -> list[dict[str, Any]]:
    payload = _parse_json_payload(raw_text)
    factors = payload.get("factors")
    if not isinstance(factors, list):
        raise ValueError("Evaluator response must contain a factors list")
    by_id = {}
    for item in factors:
        if not isinstance(item, Mapping):
            continue
        factor_id = str(item.get("factor_id", "")).strip()
        if not factor_id:
            continue
        row = {"factor_id": factor_id, "justification": str(item.get("justification", "")).strip()}
        for dimension in SEMANTIC_DIMENSIONS:
            value = item.get(dimension)
            if value is None:
                raise KeyError(f"Missing dimension {dimension} for {factor_id}")
            score = float(value)
            if score < 1 or score > 5:
                raise ValueError(f"Score out of range for {factor_id}.{dimension}: {score}")
            row[dimension] = score
        by_id[factor_id] = row
    missing = [factor_id for factor_id in expected_factor_ids if factor_id not in by_id]
    if missing:
        raise KeyError(f"Missing factor ratings: {missing}")
    return [by_id[factor_id] for factor_id in expected_factor_ids]


def _openai_chat_completion(*, model: str, prompt: str) -> str:
    from openai import OpenAI

    from scgm_text.openai_theme_labels import load_openai_dotenv
    from macro_transfer.openai_utils import apply_openai_chat_temperature, openai_chat_uses_max_completion_tokens

    load_openai_dotenv()
    client = OpenAI()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Evaluator for semantic factor quality in occupational accident analysis. "
                    "Respond with strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    kwargs = apply_openai_chat_temperature(kwargs, model=model, temperature=0.0)
    if openai_chat_uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["max_tokens"] = 4000
    response = client.chat.completions.create(**kwargs)
    return str(response.choices[0].message.content or "")


def score_packages_with_evaluators(
    packages: Sequence[Mapping[str, Any]],
    *,
    role: str,
    config: Mapping[str, Any],
    chat_completion: Callable[..., str] | None = None,
) -> pd.DataFrame:
    """Score every package with evaluator_1 and evaluator_2 under blinded random order."""
    eval_cfg = _semantic_evaluation_config(config)
    if not packages:
        return pd.DataFrame()
    configuration_ids = [str(package["configuration_id"]) for package in packages]
    rows: list[dict[str, Any]] = []
    completer = chat_completion or _openai_chat_completion
    for evaluator_id in EVALUATOR_IDS:
        model = str(eval_cfg[evaluator_id]["model"])
        anonymous_map = _anonymous_candidate_ids(
            configuration_ids,
            seed=int(eval_cfg["random_state"]),
            evaluator_id=evaluator_id,
        )
        # Present candidates in anonymous id order.
        ordered = sorted(packages, key=lambda package: anonymous_map[str(package["configuration_id"])])
        for package in ordered:
            configuration_id = str(package["configuration_id"])
            anonymous_id = anonymous_map[configuration_id]
            factors = list(package.get("factors") or [])
            factor_ids = [f"Factor-{index:02d}" for index in range(1, len(factors) + 1)]
            prompt = build_blinded_prompt(
                role=role,
                anonymous_candidate_id=anonymous_id,
                factors=factors,
            )
            raw = completer(model=model, prompt=prompt)
            parsed = parse_evaluator_response(raw, expected_factor_ids=factor_ids)
            for factor, rating in zip(factors, parsed):
                rows.append({
                    "role": role,
                    "configuration_id": configuration_id,
                    "anonymous_candidate_id": anonymous_id,
                    "evaluator_id": evaluator_id,
                    "model": model,
                    "cluster_label": int(factor["cluster_label"]),
                    "topic_id": str(factor["topic_id"]),
                    "factor_id": rating["factor_id"],
                    "coherence": float(rating["coherence"]),
                    "distinctiveness": float(rating["distinctiveness"]),
                    "prevention_relevance": float(rating["prevention_relevance"]),
                    "justification": rating["justification"],
                })
    return pd.DataFrame(rows)


def aggregate_semantic_scores(factor_scores: pd.DataFrame) -> pd.DataFrame:
    """Average factor ratings within evaluator, then mean across evaluators."""
    if factor_scores.empty:
        return pd.DataFrame(
            columns=[
                "role",
                "configuration_id",
                "coherence",
                "distinctiveness",
                "prevention_relevance",
                "semantic_score",
                "evaluator_1_score",
                "evaluator_2_score",
            ]
        )
    per_eval = (
        factor_scores.groupby(["role", "configuration_id", "evaluator_id"], as_index=False)[
            list(SEMANTIC_DIMENSIONS)
        ]
        .mean()
    )
    per_eval["semantic_score"] = per_eval[list(SEMANTIC_DIMENSIONS)].mean(axis=1)
    wide = per_eval.pivot_table(
        index=["role", "configuration_id"],
        columns="evaluator_id",
        values="semantic_score",
        aggfunc="first",
    )
    for evaluator_id in EVALUATOR_IDS:
        if evaluator_id not in wide.columns:
            wide[evaluator_id] = np.nan
    dims = (
        factor_scores.groupby(["role", "configuration_id"], as_index=False)[list(SEMANTIC_DIMENSIONS)]
        .mean()
    )
    dims["semantic_score"] = dims[list(SEMANTIC_DIMENSIONS)].mean(axis=1)
    merged = dims.merge(wide.reset_index(), on=["role", "configuration_id"], how="left")
    merged = merged.rename(
        columns={
            "evaluator_1": "evaluator_1_score",
            "evaluator_2": "evaluator_2_score",
        }
    )
    return merged


def select_configuration_by_semantic_score(
    candidates: pd.DataFrame,
    semantic_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Select max semantic score among Pareto candidates; tie-break S_R then DBCV."""
    result = mark_pareto_front(candidates)
    result["selected"] = False
    if "semantic_score" in result.columns:
        result = result.drop(columns=["semantic_score"])
    if semantic_scores is None or semantic_scores.empty:
        pareto = result.loc[result["on_pareto"]]
        if len(pareto) == 1:
            selected_id = str(pareto.iloc[0]["configuration_id"])
            result.loc[result["configuration_id"].astype(str).eq(selected_id), "selected"] = True
            return result, selected_id
        return result, ""
    merged = result.merge(
        semantic_scores[
            [
                column
                for column in (
                    "role",
                    "configuration_id",
                    "coherence",
                    "distinctiveness",
                    "prevention_relevance",
                    "semantic_score",
                    "evaluator_1_score",
                    "evaluator_2_score",
                )
                if column in semantic_scores.columns
            ]
        ],
        on=(
            ["role", "configuration_id"]
            if "role" in result.columns and "role" in semantic_scores.columns
            else ["configuration_id"]
        ),
        how="left",
    )
    pool = merged.loc[merged["on_pareto"] & merged["semantic_score"].notna()].copy()
    if pool.empty:
        return merged, ""
    max_score = float(pool["semantic_score"].max())
    tied = pool.loc[np.isclose(pool["semantic_score"].astype(float), max_score, rtol=0.0, atol=1e-12)]
    tied = tied.sort_values(
        ["stability", "dbcv_umap"],
        ascending=[False, False],
        na_position="last",
    )
    selected_id = str(tied.iloc[0]["configuration_id"])
    merged.loc[merged["configuration_id"].astype(str).eq(selected_id), "selected"] = True
    return merged, selected_id


def compute_evaluator_agreement(factor_scores: pd.DataFrame) -> pd.DataFrame:
    """Descriptive agreement between evaluator_1 and evaluator_2 at candidate level."""
    if factor_scores.empty:
        return pd.DataFrame()
    per_eval = (
        factor_scores.groupby(["role", "configuration_id", "evaluator_id"], as_index=False)[
            list(SEMANTIC_DIMENSIONS)
        ]
        .mean()
    )
    per_eval["semantic_score"] = per_eval[list(SEMANTIC_DIMENSIONS)].mean(axis=1)
    rows = []
    for role, role_frame in per_eval.groupby("role"):
        wide = role_frame.pivot_table(
            index="configuration_id",
            columns="evaluator_id",
            values="semantic_score",
            aggfunc="first",
        )
        if not {"evaluator_1", "evaluator_2"}.issubset(wide.columns) or wide.dropna().empty:
            continue
        usable = wide.dropna()
        top_1 = str(usable["evaluator_1"].idxmax())
        top_2 = str(usable["evaluator_2"].idxmax())
        mad = float(np.mean(np.abs(usable["evaluator_1"] - usable["evaluator_2"])))
        if len(usable) >= 2:
            correlation = float(usable["evaluator_1"].corr(usable["evaluator_2"], method="spearman"))
        else:
            correlation = np.nan
        dim_mads = {}
        for dimension in SEMANTIC_DIMENSIONS:
            dim_wide = role_frame.pivot_table(
                index="configuration_id",
                columns="evaluator_id",
                values=dimension,
                aggfunc="first",
            ).dropna()
            if {"evaluator_1", "evaluator_2"}.issubset(dim_wide.columns) and not dim_wide.empty:
                dim_mads[f"mad_{dimension}"] = float(
                    np.mean(np.abs(dim_wide["evaluator_1"] - dim_wide["evaluator_2"]))
                )
            else:
                dim_mads[f"mad_{dimension}"] = np.nan
        rows.append({
            "role": role,
            "n_candidates_compared": int(len(usable)),
            "same_top_ranked": bool(top_1 == top_2),
            "evaluator_1_top": top_1,
            "evaluator_2_top": top_2,
            "mean_abs_score_difference": mad,
            "spearman_semantic_score": correlation,
            **dim_mads,
        })
    return pd.DataFrame(rows)


def ensure_openai_available(config: Mapping[str, Any]) -> None:
    eval_cfg = _semantic_evaluation_config(config)
    if not bool(eval_cfg.get("require_api_key", True)):
        return
    from scgm_text.openai_theme_labels import load_openai_dotenv

    load_openai_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY manquant pour l'évaluation sémantique (evaluator_1 / evaluator_2)."
        )


def run_role_semantic_evaluation(
    *,
    role: str,
    units: pd.DataFrame,
    output_dir: Path,
    candidate_table: pd.DataFrame,
    config: Mapping[str, Any],
    chat_completion: Callable[..., str] | None = None,
    reestimate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build packages for Pareto configs, score them, write artifacts, return tables."""
    eval_cfg = _semantic_evaluation_config(config)
    role_dir = Path(output_dir) / "discovery" / role
    eval_dir = role_dir / "semantic_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    factor_path = eval_dir / "factor_scores.csv"
    candidate_path = eval_dir / "candidate_scores.csv"
    agreement_path = eval_dir / "evaluator_agreement.csv"
    packages_path = eval_dir / "packages.json"

    marked = mark_pareto_front(candidate_table)
    pareto_ids = (
        marked.loc[marked["on_pareto"], "configuration_id"].astype(str).tolist()
        if not marked.empty
        else []
    )
    if len(pareto_ids) <= 1:
        empty_factors = pd.DataFrame()
        empty_scores = pd.DataFrame()
        empty_agreement = pd.DataFrame()
        empty_factors.to_csv(factor_path, index=False)
        empty_scores.to_csv(candidate_path, index=False)
        empty_agreement.to_csv(agreement_path, index=False)
        packages_path.write_text(json.dumps({"role": role, "packages": []}, indent=2), encoding="utf-8")
        return empty_factors, empty_scores, empty_agreement

    if (
        not reestimate
        and factor_path.is_file()
        and candidate_path.is_file()
        and agreement_path.is_file()
    ):
        return (
            pd.read_csv(factor_path),
            pd.read_csv(candidate_path),
            pd.read_csv(agreement_path),
        )

    if chat_completion is None:
        ensure_openai_available(config)

    candidate_dir = role_dir / "candidate_partitions"
    rng = np.random.default_rng(int(eval_cfg["random_state"]) + sum(ord(char) for char in role))
    packages = []
    for configuration_id in pareto_ids:
        labels = np.load(candidate_dir / f"{configuration_id}_labels.npy").astype(np.int32)
        strengths = np.load(candidate_dir / f"{configuration_id}_membership_strength.npy").astype(float)
        packages.append(
            build_evaluation_package_for_configuration(
                role=role,
                configuration_id=configuration_id,
                units=units,
                labels=labels,
                strengths=strengths,
                units_per_factor=int(eval_cfg["units_per_factor"]),
                rng=rng,
            )
        )
    packages_path.write_text(
        json.dumps(
            {
                "version": "semantic_evaluation_packages_v1",
                "role": role,
                "units_per_factor": int(eval_cfg["units_per_factor"]),
                "evaluator_1_model": eval_cfg["evaluator_1"]["model"],
                "evaluator_2_model": eval_cfg["evaluator_2"]["model"],
                "packages": packages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    factor_scores = score_packages_with_evaluators(
        packages,
        role=role,
        config=config,
        chat_completion=chat_completion,
    )
    candidate_scores = aggregate_semantic_scores(factor_scores)
    agreement = compute_evaluator_agreement(factor_scores)
    factor_scores.to_csv(factor_path, index=False)
    candidate_scores.to_csv(candidate_path, index=False)
    agreement.to_csv(agreement_path, index=False)
    (eval_dir / "evaluation_metadata.json").write_text(
        json.dumps({
            "version": "semantic_evaluation_v1",
            "role": role,
            "n_pareto_candidates": len(pareto_ids),
            "evaluator_1_model": eval_cfg["evaluator_1"]["model"],
            "evaluator_2_model": eval_cfg["evaluator_2"]["model"],
        }, indent=2),
        encoding="utf-8",
    )
    return factor_scores, candidate_scores, agreement
