"""TDM-CC-BU-K inference using the cover-partition groups of Yang and Deng.

An observed basket is the union of independently selected binary themes.  The
likelihood sums over cover partitions, including the probability of every
optional proper subtheme in each partition group.  K bounds multiplicity in
the *cover partition*, as in equation (4.12) of the paper.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import multiprocessing as mp
import sys
from typing import Iterable, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.sparse import csr_matrix


class ComplexityLimitError(RuntimeError):
    """An exact cover enumeration exceeds the configured resource guard."""

    def __init__(self, message: str, basket: tuple[int, ...] | None = None,
                 lower_bound_groups: int | None = None):
        super().__init__(message)
        self.basket = basket
        self.lower_bound_groups = lower_bound_groups


@dataclass(frozen=True)
class Theme:
    items: tuple[int, ...]
    support: int

    @property
    def mask(self) -> int:
        return sum(1 << item for item in self.items)


@dataclass(frozen=True)
class CoverGroup:
    maximal: tuple[int, ...]
    proper: tuple[int, ...]


@dataclass(frozen=True)
class Component:
    theme_ids: tuple[int, ...]
    groups: tuple[CoverGroup, ...]
    maximal_matrix: csr_matrix
    proper_matrix: csr_matrix


@dataclass(frozen=True)
class CompiledObservation:
    basket: tuple[int, ...]
    frequency: int
    components: tuple[Component, ...]


def mine_dictionary(
    baskets: Sequence[tuple[int, ...]], roles: Sequence[str],
    max_length: int = 4, min_count: int = 3,
) -> list[Theme]:
    """All singletons, then frequent cross-category itemsets (exact counts)."""
    counts: Counter[tuple[int, ...]] = Counter()
    for basket in baskets:
        if len(basket) != len(set(basket)) or tuple(sorted(basket)) != basket:
            raise ValueError("Baskets must be sorted sets of item indices")
        for size in range(2, min(max_length, len(basket)) + 1):
            counts.update(combinations(basket, size))
    singles = Counter(item for basket in baskets for item in basket)
    result = [Theme((item,), singles[item]) for item in range(len(roles))]
    result.extend(
        Theme(items, count) for items, count in sorted(counts.items())
        if count >= min_count and len({roles[item] for item in items}) >= 2
    )
    return result


def unique_baskets(baskets: Iterable[tuple[int, ...]]) -> list[tuple[tuple[int, ...], int]]:
    # Audit the expensive baskets first, before investing hours in an EM fit.
    return sorted(Counter(baskets).items(), key=lambda row: (-len(row[0]), row[0]))


def count_cover_groups(
    basket: tuple[int, ...], themes: Sequence[Theme], max_overlap: int,
    cap: int,
) -> tuple[int, bool]:
    """Count distinct cover partitions without storing them, stopping above cap.

    This is a complexity diagnostic, not a likelihood approximation.  Its
    antichain and overlap checks are the same as the group constructor below.
    The boolean reports that the returned count is only a strict lower bound.
    """
    if not basket:
        return 1, False
    basket_mask = sum(1 << item for item in basket)
    available = [theme for theme in themes if theme.mask & basket_mask == theme.mask]
    composites = [theme for theme in available if len(theme.items) > 1]
    singletons = {theme.items[0] for theme in available if len(theme.items) == 1}
    all_singletons = len(singletons.intersection(basket)) == len(basket)
    position = {item: idx for idx, item in enumerate(basket)}
    composite_items = [tuple(position[item] for item in theme.items) for theme in composites]
    incompatibility = []
    for left, theme in enumerate(composites):
        mask = theme.mask
        incompatible = 0
        for right in range(left + 1, len(composites)):
            other_mask = composites[right].mask
            if (mask & other_mask) in (mask, other_mask):
                incompatible |= 1 << right
        incompatibility.append(incompatible)
    item_counts = [0] * len(basket)
    count = 0

    class CountLimit(Exception):
        pass

    def walk(candidates: int) -> None:
        nonlocal count
        if all_singletons or all(item_counts[idx] > 0 or item in singletons
                                 for idx, item in enumerate(basket)):
            count += 1
            if count > cap:
                raise CountLimit
        while candidates:
            bit = candidates & -candidates
            idx = bit.bit_length() - 1
            candidates ^= bit
            if any(item_counts[item] >= max_overlap for item in composite_items[idx]):
                continue
            for item in composite_items[idx]:
                item_counts[item] += 1
            walk(candidates & ~incompatibility[idx])
            for item in composite_items[idx]:
                item_counts[item] -= 1

    try:
        walk((1 << len(composites)) - 1)
    except CountLimit:
        return count, True
    return count, False


def _components_for_basket(
    basket: tuple[int, ...], themes: Sequence[Theme], max_overlap: int,
    max_groups: int,
) -> tuple[Component, ...]:
    if not basket:
        return ()
    basket_mask = sum(1 << j for j in basket)
    available = [i for i, theme in enumerate(themes) if theme.mask & basket_mask == theme.mask]
    parent = {item: item for item in basket}

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for idx in available:
        items = themes[idx].items
        for item in items[1:]:
            parent[root(item)] = root(items[0])
    item_parts: dict[int, list[int]] = defaultdict(list)
    for item in basket:
        item_parts[root(item)].append(item)
    components: list[Component] = []
    for items in item_parts.values():
        item_set = set(items)
        ids = tuple(idx for idx in available if set(themes[idx].items) <= item_set)
        composites = tuple(idx for idx in ids if len(themes[idx].items) > 1)
        singleton = {themes[idx].items[0]: idx for idx in ids if len(themes[idx].items) == 1}
        subthemes = {
            idx: tuple(other for other in ids if other != idx and
                       themes[other].mask & themes[idx].mask == themes[other].mask)
            for idx in composites
        }
        groups: list[CoverGroup] = []
        selected: list[int] = []
        item_counts = {item: 0 for item in items}
        # Each valid set of composite maximal themes is visited exactly once.
        # Missing items are supplied by singleton maximal themes; all proper
        # subthemes are summed analytically in (4.9)--(4.11), never enumerated.
        # Bit masks discard incompatible supersets/subsets before recursion.
        incompatibility = []
        for left, idx in enumerate(composites):
            mask = themes[idx].mask
            incompatible = 0
            for right in range(left + 1, len(composites)):
                other_mask = themes[composites[right]].mask
                if (mask & other_mask) in (mask, other_mask):
                    incompatible |= 1 << right
            incompatibility.append(incompatible)

        def walk(candidates: int) -> None:
            uncovered = [item for item, count in item_counts.items() if count == 0]
            if all(item in singleton for item in uncovered):
                maximal = tuple(sorted(selected + [singleton[item] for item in uncovered]))
                proper = tuple(sorted({other for idx in selected for other in subthemes[idx]}))
                groups.append(CoverGroup(maximal, proper))
                if len(groups) > max_groups:
                    raise ComplexityLimitError(
                        f"Basket {basket}: over {max_groups:,} exact cover groups "
                        f"in a component of {len(items)} items and {len(composites)} "
                        "composite themes. Increase resources or revise the *declared* "
                        "candidate settings; no approximate result was produced.",
                        basket=basket, lower_bound_groups=len(groups),
                    )
            while candidates:
                bit = candidates & -candidates
                position = bit.bit_length() - 1
                candidates ^= bit
                idx = composites[position]
                if any(item_counts[item] >= max_overlap for item in themes[idx].items):
                    continue
                selected.append(idx)
                for item in themes[idx].items:
                    item_counts[item] += 1
                walk(candidates & ~incompatibility[position])
                for item in themes[idx].items:
                    item_counts[item] -= 1
                selected.pop()

        walk((1 << len(composites)) - 1)
        if not groups:
            raise ValueError(f"Basket {basket} cannot be covered by active themes")
        def membership_matrix(kind: str) -> csr_matrix:
            memberships = [getattr(group, kind) for group in groups]
            sizes = np.fromiter((len(row) for row in memberships), dtype=np.int64,
                                count=len(memberships))
            indptr = np.empty(len(groups) + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(sizes, out=indptr[1:])
            indices = np.fromiter((index for row in memberships for index in row),
                                  dtype=np.int32, count=int(indptr[-1]))
            data = np.ones(len(indices), dtype=np.float64)
            return csr_matrix((data, indices, indptr), shape=(len(groups), len(themes)))

        components.append(Component(
            ids, tuple(groups), membership_matrix("maximal"),
            membership_matrix("proper"),
        ))
    return tuple(components)


def evaluate_basket(
    basket: tuple[int, ...], themes: Sequence[Theme], theta: np.ndarray,
    max_overlap: int, max_groups: int, *, map_partition: bool = False,
    components: tuple[Component, ...] | None = None,
) -> tuple[float, np.ndarray, tuple[int, ...]]:
    """Observed log-likelihood, posterior inclusion and optional MAP partition.

    The global all-absent probability is factored out.  Each cover group then
    contributes selected-theme odds and removes absent factors for its optional
    proper subthemes, exactly as equations (4.5)--(4.11).
    """
    if len(theta) != len(themes) or np.any(theta <= 0) or np.any(theta >= 1):
        raise ValueError("theta must lie strictly between 0 and 1 for every active theme")
    log_absent = np.log1p(-theta)
    log_odds = np.log(theta) - log_absent
    ll = float(log_absent.sum())
    posterior = np.zeros(len(themes), dtype=float)
    map_ids: list[int] = []
    if components is None:
        components = _components_for_basket(basket, themes, max_overlap, max_groups)
    for comp in components:
        increments = (comp.maximal_matrix @ log_odds -
                      comp.proper_matrix @ log_absent)
        normalizer = float(logsumexp(increments))
        ll += normalizer
        weights = np.exp(increments - normalizer)
        posterior += np.asarray(comp.maximal_matrix.T @ weights).ravel()
        posterior += np.asarray(comp.proper_matrix.T @ weights).ravel() * theta
        if map_partition:
            # The best *latent partition*, rather than the most probable group.
            scores = (comp.maximal_matrix @ log_odds +
                      comp.proper_matrix @ np.maximum(0.0, log_odds))
            best = comp.groups[int(np.argmax(scores))]
            map_ids.extend(best.maximal)
            map_ids.extend(idx for idx in best.proper if log_odds[idx] > 0)
    return ll, posterior, tuple(sorted(map_ids))


def _evaluate_chunk(
    observations: Sequence[CompiledObservation], themes: Sequence[Theme],
    theta: np.ndarray, max_overlap: int, max_groups: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    ll = 0.0
    expected = np.zeros(len(themes))
    log_no_posterior = np.zeros(len(themes))
    for obs in observations:
        value, posterior, _ = evaluate_basket(
            obs.basket, themes, theta, max_overlap, max_groups,
            components=obs.components,
        )
        ll += obs.frequency * value
        expected += obs.frequency * posterior
        with np.errstate(divide="ignore"):
            log_no_posterior += obs.frequency * np.log1p(-np.minimum(posterior, 1.0))
    return ll, expected, log_no_posterior


def expectation(
    observations: Sequence[CompiledObservation], themes: Sequence[Theme],
    theta: np.ndarray, max_overlap: int, max_groups: int, workers: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Deterministic reduction of independent basket E-steps."""
    count = min(max(1, workers), len(observations))
    batches = [observations[i::count] for i in range(count)]
    if count == 1:
        results = [_evaluate_chunk(batches[0], themes, theta, max_overlap, max_groups)]
    else:
        results = Parallel(n_jobs=count, backend="loky") (
            delayed(_evaluate_chunk)(batch, themes, theta, max_overlap, max_groups)
            for batch in batches
        )
    return (float(sum(value[0] for value in results)),
            np.sum([value[1] for value in results], axis=0),
            np.sum([value[2] for value in results], axis=0))


_POOL_OBSERVATIONS = None
_POOL_THEMES = None
_POOL_K = None
_POOL_MAX_GROUPS = None


def _pool_setup(observations, themes, k, max_groups):
    global _POOL_OBSERVATIONS, _POOL_THEMES, _POOL_K, _POOL_MAX_GROUPS
    _POOL_OBSERVATIONS = observations
    _POOL_THEMES = themes
    _POOL_K = k
    _POOL_MAX_GROUPS = max_groups


def _pool_evaluate_components(task):
    parts, theta = task
    log_absent = np.log1p(-theta)
    log_odds = np.log(theta) - log_absent
    output = []
    for observation_index, component_index, first, last in parts:
        component = _POOL_OBSERVATIONS[observation_index].components[component_index]
        maximal = component.maximal_matrix[first:last]
        proper = component.proper_matrix[first:last]
        increments = maximal @ log_odds - proper @ log_absent
        normalizer = float(logsumexp(increments))
        weights = np.exp(increments - normalizer)
        posterior = (np.asarray(maximal.T @ weights).ravel() +
                     np.asarray(proper.T @ weights).ravel() * theta)
        output.append((observation_index, component_index, first, normalizer, posterior))
    return output


class BasketEvaluator:
    """Persistent fork workers share precompiled sparse groups on Linux.

    Work is balanced by exact group count rather than by basket count. Windows
    uses the serial evaluator for reproducible local checks without duplicating
    large sparse matrices across spawned processes.
    """

    def __init__(self, observations, themes, k, max_groups, workers):
        self.observations = observations
        self.themes = themes
        self.k = k
        self.max_groups = max_groups
        self.workers = min(max(1, workers), len(observations))
        self.pool = None
        maximal_size = max((len(component.groups) for obs in observations
                            for component in obs.components), default=1)
        part_size = max(1000, (maximal_size + self.workers - 1) // self.workers)
        parts = []
        for i, obs in enumerate(observations):
            for j, component in enumerate(obs.components):
                for first in range(0, len(component.groups), part_size):
                    last = min(len(component.groups), first + part_size)
                    parts.append((i, j, first, last))
        loads = [0] * self.workers
        self.parts = [[] for _ in range(self.workers)]
        for part in sorted(parts, key=lambda row: (-(row[3] - row[2]), row[0], row[1], row[2])):
            worker = min(range(self.workers), key=lambda j: (loads[j], j))
            self.parts[worker].append(part)
            loads[worker] += part[3] - part[2]

    def __enter__(self):
        if self.workers > 1 and sys.platform != "win32":
            self.pool = mp.get_context("fork").Pool(
                self.workers,
                initializer=_pool_setup,
                initargs=(self.observations, self.themes, self.k, self.max_groups),
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pool is not None:
            self.pool.close() if exc_type is None else self.pool.terminate()
            self.pool.join()

    def evaluate(self, theta):
        if self.pool is None:
            return _evaluate_chunk(self.observations, self.themes, theta,
                                   self.k, self.max_groups)
        results = self.pool.map(_pool_evaluate_components,
                                [(parts, theta) for parts in self.parts])
        grouped = defaultdict(list)
        for worker_result in results:
            for i, j, first, normalizer, posterior in worker_result:
                grouped[(i, j)].append((first, normalizer, posterior))
        log_absent_total = float(np.log1p(-theta).sum())
        ll = 0.0
        expected = np.zeros(len(self.themes))
        log_no = np.zeros(len(self.themes))
        for i, obs in enumerate(self.observations):
            basket_ll = log_absent_total
            basket_post = np.zeros(len(self.themes))
            for j in range(len(obs.components)):
                pieces = sorted(grouped[(i, j)], key=lambda row: row[0])
                normalizer = float(logsumexp([piece[1] for piece in pieces]))
                basket_ll += normalizer
                for _, local_norm, local_post in pieces:
                    basket_post += np.exp(local_norm - normalizer) * local_post
            ll += obs.frequency * basket_ll
            expected += obs.frequency * basket_post
            with np.errstate(divide="ignore"):
                log_no += obs.frequency * np.log1p(-np.minimum(basket_post, 1.0))
        return ll, expected, log_no


def em_fit(
    observations: Sequence[CompiledObservation], themes: Sequence[Theme],
    initial: np.ndarray, max_overlap: int, max_groups: int, workers: int,
    tolerance: float, max_iterations: int, *,
    iteration_callback=None, start_iteration: int = 0, previous_ll: float = -np.inf,
    stable_iterations: int = 0, evaluator=None,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    n = sum(obs.frequency for obs in observations)
    theta = np.asarray(initial, dtype=float).copy()
    evaluate = evaluator or (lambda value: expectation(
        observations, themes, value, max_overlap, max_groups, workers
    ))
    for iteration in range(start_iteration, max_iterations):
        ll, expected, _ = evaluate(theta)
        if ll + 1e-7 < previous_ll:
            raise ArithmeticError(f"EM likelihood fell at iteration {iteration}: {ll} < {previous_ll}")
        change = (ll - previous_ll) / max(1.0, abs(previous_ll)) if np.isfinite(previous_ll) else np.inf
        stable_iterations = stable_iterations + 1 if change < tolerance else 0
        updated = np.clip(expected / n, 1e-12, 1 - 1e-12)
        if iteration_callback is not None:
            iteration_callback(iteration + 1, updated, ll, stable_iterations)
        theta = updated
        previous_ll = ll
        if stable_iterations >= 3:
            final_ll, _, log_no = evaluate(theta)
            if final_ll + 1e-7 < ll:
                raise ArithmeticError("Final EM likelihood is lower than preceding iteration")
            return theta, final_ll, log_no, iteration + 1
    raise RuntimeError(f"EM did not converge within {max_iterations} iterations")


def deletion_scores(theta: np.ndarray, log_no_posterior: np.ndarray, n: int) -> np.ndarray:
    """Equation (2.7), via P(Z_a=0 | O), without refitting deleted models."""
    if len(theta) != len(log_no_posterior):
        raise ValueError("Mismatched parameter and posterior dimensions")
    return n * np.log1p(-theta) - log_no_posterior


def initial_theta(themes: Sequence[Theme], n: int, scale: float) -> np.ndarray:
    return np.clip(np.array([theme.support / n * scale for theme in themes]), 1e-8, 1 - 1e-8)


def overlap_index(partitions: Sequence[tuple[int, ...]], themes: Sequence[Theme]) -> tuple[int, int, float]:
    count = 0
    for partition in partitions:
        masks = [themes[idx].mask for idx in partition]
        if any(left & right for left, right in combinations(masks, 2)):
            count += 1
    return count, len(partitions), count / len(partitions) if partitions else float("nan")


def compile_observations(
    observations: Sequence[tuple[tuple[int, ...], int]], themes: Sequence[Theme],
    max_overlap: int, max_groups: int, max_total_groups: int,
    progress=None,
) -> list[CompiledObservation]:
    compiled: list[CompiledObservation] = []
    total = 0
    for number, (basket, frequency) in enumerate(observations, 1):
        components = _components_for_basket(basket, themes, max_overlap, max_groups)
        total += sum(len(component.groups) for component in components)
        if total > max_total_groups:
            raise ComplexityLimitError(
                f"Over {max_total_groups:,} exact cover groups across baskets at "
                f"basket {number}/{len(observations)}. No approximate fit was produced.",
                basket=basket, lower_bound_groups=total,
            )
        compiled.append(CompiledObservation(basket, frequency, components))
        if progress is not None and (number % 100 == 0 or number == len(observations)):
            progress(number, len(observations), total)
    return compiled
