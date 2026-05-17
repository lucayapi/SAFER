"""Inférence exacte (VariableElimination) et lifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def run_bn_queries(
    model: Any,
    queries_config: Optional[Sequence[dict]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute un ensemble de requêtes marginales / conditionnelles et calcule des lifts.

    ``queries_config`` : liste de dicts
    ``{"type": "marginal"|"conditional", "variables": [...], "evidence": {...}}``
    """
    try:
        from pgmpy.inference import VariableElimination
    except ImportError as e:
        raise ImportError("Installez pgmpy : pip install pgmpy") from e

    infer = VariableElimination(model)
    rows = []
    if not queries_config:
        queries_config = default_queries_from_model(model)

    for i, q in enumerate(queries_config):
        qtype = q.get("type", "marginal")
        vars_ = q.get("variables", [])
        ev = q.get("evidence") or {}
        try:
            if qtype == "marginal":
                res = infer.query(variables=vars_, evidence=ev if ev else None)
            else:
                res = infer.query(variables=vars_, evidence=ev)
            vals = np.asarray(res.values).ravel()
            rows.append(
                {
                    "query_id": i,
                    "type": qtype,
                    "variables": ",".join(vars_),
                    "evidence": str(ev),
                    "result_max_prob": float(np.max(vals)) if vals.size else float("nan"),
                    "result_argmax_state": int(np.argmax(vals)) if vals.size else -1,
                }
            )
        except Exception as ex:
            rows.append(
                {
                    "query_id": i,
                    "type": qtype,
                    "variables": ",".join(vars_),
                    "evidence": str(ev),
                    "result_max_prob": float("nan"),
                    "result_argmax_state": -1,
                    "error": str(ex),
                }
            )

    query_df = pd.DataFrame(rows)
    lift_df = compute_lifts_from_model(model, infer)
    return query_df, lift_df


def default_queries_from_model(model: Any) -> List[dict]:
    nodes = list(model.nodes())
    out: List[dict] = []
    for n in nodes[: min(12, len(nodes))]:
        out.append({"type": "marginal", "variables": [n], "evidence": {}})
    return out


def compute_lifts_from_model(model: Any, infer: Any) -> pd.DataFrame:
    """Lift(Y|X) = P(Y=1|X=1)/P(Y=1) pour arcs du modèle (variables binaires 0/1)."""
    lifts = []
    for y in model.nodes():
        parents = list(model.predecessors(y))
        for x in parents:
            try:
                q0 = infer.query(variables=[y])
                arr = np.asarray(q0.values).ravel()
                py1 = float(arr[1]) if arr.size > 1 else float(arr[0])
            except Exception:
                continue
            if py1 <= 1e-12:
                continue
            try:
                q1 = infer.query(variables=[y], evidence={x: 1})
                arr1 = np.asarray(q1.values).ravel()
                p_y1_x1 = float(arr1[1]) if arr1.size > 1 else float(arr1[0])
            except Exception:
                continue
            lifts.append(
                {
                    "parent": x,
                    "child": y,
                    "p_y1": py1,
                    "p_y1_given_x1": p_y1_x1,
                    "lift": p_y1_x1 / py1,
                }
            )
    return pd.DataFrame(lifts)


def conditional_prob_table(
    model: Any,
    child: str,
    parent: str,
) -> pd.DataFrame:
    from pgmpy.inference import VariableElimination

    infer = VariableElimination(model)
    rows = []
    for xv in (0, 1):
        try:
            r = infer.query(variables=[child], evidence={parent: xv})
            v = np.asarray(r.values).ravel()
            p1 = float(v[1]) if len(v) > 1 else float(v[0])
        except Exception:
            p1 = float("nan")
        rows.append({"parent": parent, "parent_value": xv, f"p_{child}_eq_1": p1})
    return pd.DataFrame(rows)
