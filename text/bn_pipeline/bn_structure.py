"""Contraintes de structure macro et apprentissage HillClimb (pgmpy)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd

MACRO_ORDER_MAP = {"A0": 0, "A1": 1, "B": 2, "C": 3, "Severity": 4, "Severity_high": 4, "SEVERITY": 4}

# Graphe causal accidentologique imposé (pas de saut A0/A1 → C).
STANDARD_ALLOWED_MACRO_EDGES: Set[Tuple[str, str]] = {
    ("A0", "A1"),
    ("A0", "B"),
    ("A1", "B"),
    ("B", "C"),
}
SEVERITY_ALLOWED_MACRO_EDGES: Set[Tuple[str, str]] = {("C", "Severity")}

MACRO_ONTOLOGY_FR: Dict[str, Tuple[str, str]] = {
    "A0": ("A0 — Contexte", "Travail, activité, environnement, matériel"),
    "A1": ("A1 — Facteurs contributifs", "Défaillances, conditions dangereuses"),
    "B": ("B — Événement", "Accident ou déviation"),
    "C": ("C — Conséquence", "Dommage, blessure, issue"),
}


def _normalize_macro_label(label: str) -> str:
    lab = str(label).strip()
    if lab.startswith("M_"):
        lab = lab[2:]
    if lab in MACRO_ORDER_MAP:
        return lab
    key = lab.upper()
    if key in MACRO_ORDER_MAP:
        return {"SEVERITY": "Severity"}.get(key, key)
    if "Severity" in lab:
        return "Severity"
    return lab


def macro_label_of_node(node: str, variable_macro_map: Dict[str, str]) -> str:
    if node in variable_macro_map:
        return _normalize_macro_label(variable_macro_map[node])
    if str(node).startswith("M_"):
        return _normalize_macro_label(str(node))
    if str(node).startswith("macro_topic_"):
        parts = str(node).split("_")
        if len(parts) >= 3:
            return _normalize_macro_label(parts[2])
    return _normalize_macro_label(str(node))


def macro_rank(node: str, variable_macro_map: Dict[str, str]) -> int:
    lab = macro_label_of_node(node, variable_macro_map)
    return int(MACRO_ORDER_MAP.get(lab, 0))


def allowed_macro_edges_for_nodes(
    nodes: Sequence[str],
    variable_macro_map: Dict[str, str],
) -> Set[Tuple[str, str]]:
    """Arcs autorisés entre macros présentes dans le graphe."""
    macros = {macro_label_of_node(n, variable_macro_map) for n in nodes}
    allowed = {e for e in STANDARD_ALLOWED_MACRO_EDGES if e[0] in macros and e[1] in macros}
    if "Severity" in macros or any(str(n).startswith("Severity") for n in nodes):
        allowed |= {e for e in SEVERITY_ALLOWED_MACRO_EDGES if e[0] in macros and e[1] in macros}
    return allowed


def standard_macro_edge_templates(include_severity: bool = False) -> List[Tuple[str, str]]:
    """Modèle d'arcs pour export CSV et schéma M_*."""
    edges = list(STANDARD_ALLOWED_MACRO_EDGES)
    if include_severity:
        edges.extend(SEVERITY_ALLOWED_MACRO_EDGES)
    templates: List[Tuple[str, str]] = []
    for u, v in edges:
        templates.append((u, v))
        if u != "Severity" and v != "Severity":
            templates.append((f"M_{u}", f"M_{v}"))
        elif v == "Severity":
            templates.append(("M_C", "Severity_high"))
    return sorted(set(templates))


def build_blacklist(
    nodes: Sequence[str],
    variable_macro_map: Dict[str, str],
) -> List[Tuple[str, str]]:
    """
    Blacklist = tous les arcs (parent, enfant) sauf ceux conformes au DAG :

    A0→A1, A0→B, A1→B, B→C (et optionnellement C→Severity).

    Interdit notamment : A0→C, A1→C (saut de B), retours C→B, B→A*, etc.
    """
    nodes = list(nodes)
    allowed_macro = allowed_macro_edges_for_nodes(nodes, variable_macro_map)
    bl: Set[Tuple[str, str]] = set()
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            mu = macro_label_of_node(u, variable_macro_map)
            mv = macro_label_of_node(v, variable_macro_map)
            if mu in ("Severity", "Severity_high") or str(u).startswith("Severity"):
                bl.add((u, v))
                continue
            if (mu, mv) not in allowed_macro:
                bl.add((u, v))
    return sorted(bl)


def _edges_to_nx(edges: Iterable[Tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for u, v in edges:
        g.add_edge(u, v)
    return g


def prune_forbidden_edges(edges: List[Tuple[str, str]], blacklist: Set[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(u, v) for (u, v) in edges if (u, v) not in blacklist]


def break_cycles(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    edges = list(edges)
    g = _edges_to_nx(edges)
    while not nx.is_directed_acyclic_graph(g):
        try:
            cyc = nx.find_cycle(g)
            e = (cyc[-1][0], cyc[-1][1])
            if e in edges:
                edges.remove(e)
            g = _edges_to_nx(edges)
        except (nx.NetworkXNoCycle, nx.NetworkXError, ValueError):
            break
    return edges


def _bic_score_class():
    """pgmpy 0.1.x : ``BicScore`` ; pgmpy 1.x : parfois ``BICScore``."""
    try:
        from pgmpy.estimators import BICScore

        return BICScore
    except ImportError:
        from pgmpy.estimators import BicScore

        return BicScore


def _hill_climb_search(data: pd.DataFrame, score: object):
    """Constructeur HillClimb : pgmpy 0.1.x = ``(data)`` seul ; 1.x peut passer ``scoring_method``."""
    from pgmpy.estimators import HillClimbSearch

    try:
        return HillClimbSearch(data, scoring_method=score)
    except TypeError:
        return HillClimbSearch(data)


def learn_macro_constrained_structure(
    data: pd.DataFrame,
    variable_macro_map: Dict[str, str],
    macro_order: Optional[Dict[str, int]] = None,
    scoring_method: str = "bic",
    max_indegree: int = 3,
    seed: int = 42,
) -> Tuple[object, List[Tuple[str, str]]]:
    """
    Retourne (BayesianNetwork pgmpy, liste d'arcs après application des contraintes).
    """
    del macro_order, scoring_method, seed  # réservés API / extensions
    try:
        from pgmpy.models import BayesianNetwork
    except ImportError as e:
        raise ImportError("Installez pgmpy : pip install pgmpy") from e

    nodes = [c for c in data.columns if c in variable_macro_map]
    df = data[nodes].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    blacklist = set(build_blacklist(nodes, variable_macro_map))
    Scorer = _bic_score_class()
    score = Scorer(df)
    hc = _hill_climb_search(df, score)

    est_sig = inspect.signature(hc.estimate)
    kwargs: dict = {}
    kwargs["scoring_method"] = score
    if "black_list" in est_sig.parameters:
        kwargs["black_list"] = sorted(blacklist)
    elif "blacklist" in est_sig.parameters:
        kwargs["blacklist"] = sorted(blacklist)
    if "max_indegree" in est_sig.parameters:
        kwargs["max_indegree"] = max_indegree

    try:
        model = hc.estimate(**kwargs)
    except TypeError:
        kwargs.pop("scoring_method", None)
        try:
            model = hc.estimate(scoring_method=score, max_indegree=max_indegree)
        except TypeError:
            try:
                model = hc.estimate(max_indegree=max_indegree)
            except TypeError:
                model = hc.estimate()

    edges = prune_forbidden_edges(list(model.edges()), blacklist)
    edges = break_cycles(edges)
    model = BayesianNetwork(edges)
    return model, edges


def learn_unconstrained_structure(
    data: pd.DataFrame,
    variables: Sequence[str],
    max_indegree: int = 3,
) -> Tuple[object, List[Tuple[str, str]]]:
    from pgmpy.models import BayesianNetwork

    nodes = [c for c in variables if c in data.columns]
    df = data[nodes].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    Scorer = _bic_score_class()
    score = Scorer(df)
    hc = _hill_climb_search(df, score)
    est_sig = inspect.signature(hc.estimate)
    kw: dict = {"scoring_method": score}
    if "max_indegree" in est_sig.parameters:
        kw["max_indegree"] = max_indegree
    try:
        model = hc.estimate(**kw)
    except TypeError:
        kw.pop("scoring_method", None)
        try:
            model = hc.estimate(scoring_method=score, max_indegree=max_indegree)
        except TypeError:
            model = hc.estimate()
    edges = break_cycles(list(model.edges()))
    return BayesianNetwork(edges), edges


def macro_chain_edge_list(severity_node: Optional[str] = "Severity_high") -> List[Tuple[str, str]]:
    """Liste d'arcs du DAG macro agrégé M_*."""
    edges = [
        ("M_A0", "M_A1"),
        ("M_A0", "M_B"),
        ("M_A1", "M_B"),
        ("M_B", "M_C"),
    ]
    if severity_node:
        edges.append(("M_C", severity_node))
    return edges


def macro_chain_model(severity_node: Optional[str] = "Severity_high") -> Tuple[object, List[Tuple[str, str]]]:
    """
    DAG macro agrégé : M_A0→M_A1, M_A0→M_B, M_A1→M_B, M_B→M_C [, M_C→gravité].
    """
    from pgmpy.models import BayesianNetwork

    edges = macro_chain_edge_list(severity_node)
    return BayesianNetwork(edges), edges


def export_edge_tables(
    macro_edges: List[Tuple[str, str]],
    topic_edges: List[Tuple[str, str]],
    blacklist: List[Tuple[str, str]],
    allowed_hint: List[Tuple[str, str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(macro_edges, columns=["parent", "child"]).to_csv(out_dir / "bn_macro_edges.csv", index=False)
    pd.DataFrame(topic_edges, columns=["parent", "child"]).to_csv(out_dir / "bn_topic_edges.csv", index=False)
    pd.DataFrame(blacklist, columns=["parent", "child"]).to_csv(out_dir / "forbidden_edges.csv", index=False)
    pd.DataFrame(allowed_hint, columns=["parent", "child"]).to_csv(out_dir / "allowed_edges.csv", index=False)
