"""Constrained observed-factor Bayesian networks and recurrent motifs."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROLES = ("A0", "A1", "B", "C")


def admissible_arcs(roles: Mapping[str, str], within_role: bool = False) -> list[tuple[str, str]]:
    order = {role: i for i, role in enumerate(ROLES)}
    return [(p, c) for p in roles for c in roles
            if p != c and (order[roles[p]] < order[roles[c]] or
                           (within_role and roles[p] == roles[c] and p < c))]


def _local_score(x: np.ndarray, names: Sequence[str], child: str, parents: Sequence[str]) -> tuple[float, float]:
    idx = {name: i for i, name in enumerate(names)}
    y = x[:, idx[child]].astype(np.int64, copy=False)
    ll = 0.0
    if parents:
        positions=np.asarray([idx[p] for p in parents],dtype=int)
        weights=(1 << np.arange(len(parents),dtype=np.int64))
        parent_code=(x[:,positions].astype(np.int64,copy=False)*weights).sum(axis=1)
    else:
        parent_code=np.zeros(len(x),dtype=np.int64)
    cells=np.bincount(parent_code*2+y,minlength=2**(len(parents)+1)).reshape(-1,2)
    totals=cells.sum(axis=1)
    for k,n in zip(cells[:,1],totals):
        if k:
            ll += int(k)*math.log(int(k)/int(n))
        if n-k:
            ll += int(n-k)*math.log(int(n-k)/int(n))
    # Binary child: 2^q free parameters.
    return ll, -2 * ll + (2 ** len(parents)) * math.log(max(1, len(x)))


def _parent_options(names, roles, child, allowed, dmax):
    parents = sorted(p for p, c in allowed if c == child)
    return [combo for d in range(min(dmax, len(parents))+1)
            for combo in itertools.combinations(parents, d)]


def fit_exact_bn(matrix: pd.DataFrame, role_map: Mapping[str, str], dmax: int = 2):
    names = list(matrix.columns)
    x = matrix.to_numpy(dtype=np.int8)
    allowed = set(admissible_arcs(role_map))
    parents, scores, edges = {}, [], []
    for child in names:
        options = _parent_options(names, role_map, child, allowed, dmax)
        scored = [( _local_score(x, names, child, ps)[1], ps) for ps in options]
        best_score, best = min(scored, key=lambda z: (z[0], len(z[1]), z[1]))
        parents[child] = tuple(best)
        edges.extend((p, child) for p in best)
        scores.append({"factor_id": child, "role": role_map[child], "parents": " | ".join(best),
                       "n_parents": len(best), "BIC": best_score})
    return parents, edges, pd.DataFrame(scores)


def fit_greedy_bn(matrix: pd.DataFrame, role_map: Mapping[str, str], dmax: int = 2):
    """Greedy BIC edge additions with explicit cycle checks; includes within-role candidates."""
    names = list(matrix.columns)
    x = matrix.to_numpy(dtype=np.int8)
    allowed = set(admissible_arcs(role_map, within_role=True))
    parents = {n: [] for n in names}
    edges: list[tuple[str, str]] = []

    def creates_cycle(parent, child):
        stack, seen = [child], set()
        while stack:
            node = stack.pop()
            if node == parent:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(c for p, c in edges if p == node)
        return False

    while True:
        options = []
        for p, c in allowed:
            if p in parents[c] or len(parents[c]) >= dmax or creates_cycle(p, c):
                continue
            old = _local_score(x, names, c, parents[c])[1]
            newp = tuple(sorted((*parents[c], p)))
            new = _local_score(x, names, c, newp)[1]
            if new < old - 1e-12:
                options.append((old-new, p, c, newp, new))
        if not options:
            break
        _, p, c, newp, _ = max(options, key=lambda z: (z[0], z[1], z[2]))
        parents[c] = list(newp)
        edges.append((p, c))
    score = pd.DataFrame([{"factor_id": c, "role": role_map[c], "parents": " | ".join(parents[c]),
                           "n_parents": len(parents[c]),
                           "BIC": _local_score(x, names, c, parents[c])[1]}
                          for c in names])
    return {n: tuple(parents[n]) for n in names}, sorted(edges), score


def cpt_jeffreys(matrix: pd.DataFrame, parents: Mapping[str, Sequence[str]], alpha: float = .5):
    x, names = matrix.to_numpy(dtype=np.int8), list(matrix.columns)
    ix = {n: i for i, n in enumerate(names)}
    cpts = {}
    rows = []
    for child in names:
        ps = tuple(parents[child])
        for values in itertools.product((0, 1), repeat=len(ps)):
            mask = np.ones(len(x), dtype=bool)
            for p, v in zip(ps, values):
                mask &= x[:, ix[p]] == v
            n = int(mask.sum()); k = int(x[mask, ix[child]].sum())
            prob = (k + alpha) / (n + 2 * alpha)
            cpts[(child, values)] = prob
            rows.append({"factor_id": child, "parents": " | ".join(ps),
                         "parent_values": " | ".join(f"{p}={v}" for p,v in zip(ps,values)),
                         "n": n, "n_present": k, "p_present_jeffreys": prob})
    return cpts, pd.DataFrame(rows)


def topological_order(names: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[str]:
    incoming = {n: 0 for n in names}; children = defaultdict(list)
    for p, c in edges:
        incoming[c] += 1; children[p].append(c)
    ready = sorted(n for n in names if incoming[n] == 0); order=[]
    while ready:
        n=ready.pop(0); order.append(n)
        for c in sorted(children[n]):
            incoming[c]-=1
            if incoming[c] == 0:
                ready.append(c); ready.sort()
    if len(order) != len(names):
        raise ValueError("BN graph contains a cycle")
    return order


def _connected_factor_set(factors: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    """Return whether selected factors form a weakly connected BN subgraph."""
    selected = set(factors)
    if len(selected) <= 1:
        return True
    adjacency = {factor: set() for factor in selected}
    for parent, child in edges:
        if parent in selected and child in selected:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
    seen = set()
    stack = [next(iter(selected))]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node] - seen)
    return seen == selected


def scenario_candidates(
    matrix: pd.DataFrame,
    role_map: Mapping[str, str],
    edges: Sequence[tuple[str, str]],
    *,
    min_count=5,
    two_b=False,
):
    """Enumerate recurrent role motifs whose factors connect in the learned BN.

    Candidates contain one or two A0/A1 factors, one B factor, and optionally
    one C factor. A two-factor B+C motif is also allowed when no A-role edge
    points directly to either event in the learned graph. ``two_b`` adds a
    sensitivity set with two B factors. The minimum count is a descriptive
    candidate screen on the complete accident matrix.
    """
    a=sorted(n for n in matrix if role_map[n] in {"A0","A1"})
    bs=sorted(n for n in matrix if role_map[n]=="B")
    cs=sorted(n for n in matrix if role_map[n]=="C")
    # Integer bitsets make thresholded enumeration efficient even when the
    # factor dictionary is large: intersections and support counts are cheap.
    n=len(matrix); full=(1<<n)-1
    masks={name:sum((1<<i) for i,v in enumerate(matrix[name].to_numpy(dtype=bool)) if v)
           for name in matrix.columns}
    def intersect(items):
        value=full
        for item in items:
            value &= masks[item]
            if value.bit_count() < min_count:
                return value
        return value
    motifs=[]
    for na in (0,1,2):
        upstream_sets=[()] if na==0 else itertools.combinations(a,na)
        for ups in upstream_sets:
            upstream_mask=intersect(ups)
            if upstream_mask.bit_count()<min_count: continue
            for nb in (1,2) if two_b else (1,):
                for evs in itertools.combinations(bs,nb):
                    base=tuple(sorted((*ups,*evs)))
                    joint=upstream_mask
                    for event in evs: joint &= masks[event]
                    count=joint.bit_count()
                    if count<min_count: continue
                    if len(base)>1 and _connected_factor_set(base, edges):
                        motifs.append(base)
                    for c in cs:
                        motif=tuple(sorted((*base,c)))
                        has_A_parent=any(
                            role_map.get(parent) in {"A0", "A1"}
                            and child in (*evs, c)
                            for parent, child in edges
                        )
                        allow_BC_only=(na > 0 or not has_A_parent)
                        if (allow_BC_only and (joint & masks[c]).bit_count()>=min_count
                                and _connected_factor_set(motif, edges)):
                            motifs.append(motif)
    rows=[]
    for motif in motifs:
        count=intersect(motif).bit_count()
        motif_edges=sorted((p,c) for p,c in edges if p in motif and c in motif)
        rows.append({"scenario_id":"+".join(motif),"factor_ids":" | ".join(motif),
                     "bn_edges":" | ".join(f"{p}->{c}" for p,c in motif_edges),
                     "n_factors":len(motif),"n_accidents":n,"accident_count":count,
                     "support":count/n})
    return pd.DataFrame(rows).drop_duplicates("scenario_id").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["scenario_id","factor_ids","bn_edges","n_factors","n_accidents","accident_count","support"])


def summarize_scenarios(candidates: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Describe selected motifs on the full accident matrix, without tests."""
    rows=[]; n=len(matrix)
    output_columns=[*candidates.columns,"independence_expected_support","lift_over_independence",
      "conditional_C_n","conditional_C_count","conditional_C_support",
      "marginal_C_support","empirical_conditional_C_lift"]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)
    for row in candidates.to_dict("records"):
        fs=str(row["factor_ids"]).split(" | "); event=matrix[fs].all(axis=1)
        k=int(event.sum())
        # Lift of the joint motif over independence of its factor marginals.
        marg=matrix[fs].mean().to_numpy(dtype=float)
        expected=float(np.prod(marg)); support=k/n if n else np.nan
        lift=support/expected if expected>0 else np.nan
        c_factors=[f for f in fs if f.startswith("C_")]
        conditional_n=conditional_count=0
        conditional_support=marginal_c=conditional_lift=np.nan
        if c_factors:
            c_event=matrix[c_factors].all(axis=1)
            antecedents=[f for f in fs if f not in c_factors]
            antecedent_event=matrix[antecedents].all(axis=1)
            conditional_n=int(antecedent_event.sum())
            conditional_count=int((antecedent_event & c_event).sum())
            conditional_support=conditional_count/conditional_n if conditional_n else np.nan
            marginal_c=float(c_event.mean()) if n else np.nan
            conditional_lift=conditional_support/marginal_c if marginal_c and np.isfinite(conditional_support) else np.nan
        rows.append({**row,"accident_count":k,"support":support,
                     "independence_expected_support":expected,
                     "lift_over_independence":lift,"conditional_C_n":conditional_n,
                     "conditional_C_count":conditional_count,"conditional_C_support":conditional_support,
                     "marginal_C_support":marginal_c,"empirical_conditional_C_lift":conditional_lift})
    return pd.DataFrame(rows,columns=output_columns)


def bn_joint_probabilities(candidates: pd.DataFrame, nodes, parents, cpts, edges, *, n_sim=20000, seed=42):
    if candidates.empty:
        return pd.DataFrame(columns=["scenario_id","bn_joint_probability_mc","bn_joint_mc_se",
          "bn_conditional_C_probability_mc","bn_marginal_C_probability_mc","bn_conditional_C_lift_mc","mc_draws"])
    rng=np.random.default_rng(seed); order=topological_order(nodes,edges)
    samples=np.zeros((n_sim,len(nodes)),dtype=np.int8); ix={n:i for i,n in enumerate(nodes)}
    for child in order:
        ps=parents[child]
        if not ps:
            probs=np.full(n_sim,cpts[(child,())])
        else:
            probs=np.empty(n_sim)
            for vals in itertools.product((0,1),repeat=len(ps)):
                mask=np.ones(n_sim,dtype=bool)
                for p,v in zip(ps,vals):mask &= samples[:,ix[p]]==v
                probs[mask]=cpts[(child,vals)]
        samples[:,ix[child]]=(rng.random(n_sim)<probs).astype(np.int8)
    rows=[]
    for row in candidates.to_dict("records"):
        fs=str(row["factor_ids"]).split(" | ")
        scenario_event=samples[:,[ix[f] for f in fs]].all(axis=1)
        joint_probability=float(scenario_event.mean())
        joint_se=math.sqrt(joint_probability*(1-joint_probability)/n_sim)
        c_factors=[f for f in fs if f.startswith("C_")]
        conditional_probability=marginal_probability=conditional_lift=np.nan
        if c_factors:
            c_event=samples[:,[ix[f] for f in c_factors]].all(axis=1)
            antecedents=[f for f in fs if f not in c_factors]
            antecedent_event=samples[:,[ix[f] for f in antecedents]].all(axis=1)
            denom=float(antecedent_event.mean())
            marginal_probability=float(c_event.mean())
            conditional_probability=float((antecedent_event & c_event).mean()/denom) if denom else np.nan
            conditional_lift=conditional_probability/marginal_probability if marginal_probability else np.nan
        rows.append({"scenario_id":row["scenario_id"],"bn_joint_probability_mc":joint_probability,
                     "bn_joint_mc_se":joint_se,"bn_conditional_C_probability_mc":conditional_probability,
                     "bn_marginal_C_probability_mc":marginal_probability,
                     "bn_conditional_C_lift_mc":conditional_lift,"mc_draws":n_sim})
    return pd.DataFrame(rows)

