"""Résumé diagnostics BN."""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import pandas as pd

from .bn_structure import build_blacklist


def run_model_diagnostics(model: Any, model_name: str) -> dict:
    g = nx.DiGraph()
    g.add_edges_from(model.edges())
    nodes = list(model.nodes())
    isolated = [n for n in nodes if g.degree(n) == 0]
    comps = list(nx.weakly_connected_components(g))
    ok, err = True, ""
    try:
        ok = bool(model.check_model())
    except Exception as e:
        ok = False
        err = str(e)
    indeg = dict(g.in_degree())
    max_in = max(indeg.values()) if indeg else 0
    avg_deg = float(sum(dict(g.degree()).values()) / max(1, len(nodes))) if nodes else 0.0
    return {
        "model_name": model_name,
        "n_nodes": len(nodes),
        "n_edges": g.number_of_edges(),
        "acyclic": nx.is_directed_acyclic_graph(g),
        "check_model_ok": ok,
        "check_model_error": err,
        "max_indegree": int(max_in),
        "mean_degree": avg_deg,
        "n_weakly_connected": len(comps),
        "n_isolated": len(isolated),
        "isolated_nodes": " | ".join(isolated[:20]),
    }


def compare_structure_rows(
    name: str,
    model: Any,
    variable_macro_map: Dict[str, str],
) -> dict:
    nodes = list(model.nodes())
    g = nx.DiGraph(model.edges())
    forbidden = set(build_blacklist(nodes, variable_macro_map))
    violations = sum(1 for e in g.edges() if e in forbidden)
    return {
        "structure": name,
        "n_arcs": g.number_of_edges(),
        "n_nodes": len(nodes),
        "macro_order_violations": violations,
        "density": g.number_of_edges() / max(1, len(nodes) * (len(nodes) - 1)),
        "n_isolated": len([n for n in nodes if g.degree(n) == 0]),
    }
