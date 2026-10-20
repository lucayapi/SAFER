"""Independent small-case checks against brute-force latent theme selection."""

from itertools import product

import numpy as np
import pytest

from theme_dictionary.model import (
    BasketEvaluator, ComplexityLimitError, Theme, _components_for_basket,
    _pool_setup, _pool_evaluate_components, deletion_scores,
    em_fit, evaluate_basket, expectation, mine_dictionary, unique_baskets,
    compile_observations, count_cover_groups,
)


THEMES = [
    Theme((0,), 8), Theme((1,), 7), Theme((2,), 6),
    Theme((0, 1), 4), Theme((1, 2), 4), Theme((0, 2), 3),
]


def brute_force(basket, themes, theta, k):
    basket_mask = sum(1 << idx for idx in basket)
    mass = 0.0
    included = np.zeros(len(themes))
    best_log = -np.inf
    best_partition = ()
    for bits in product((0, 1), repeat=len(themes)):
        selected = tuple(i for i, bit in enumerate(bits) if bit)
        union = 0
        for idx in selected:
            union |= themes[idx].mask
        if union != basket_mask:
            continue
        cover = tuple(
            idx for idx in selected
            if not any(idx != other and themes[idx].mask != themes[other].mask and
                       themes[idx].mask & themes[other].mask == themes[idx].mask
                       for other in selected)
        )
        if max((sum(bool(themes[idx].mask & (1 << item)) for idx in cover)
                for item in basket), default=0) > k:
            continue
        probability = np.prod([
            theta[idx] if bits[idx] else 1 - theta[idx] for idx in range(len(themes))
        ])
        mass += probability
        included += probability * np.array(bits)
        if np.log(probability) > best_log:
            best_log = np.log(probability)
            best_partition = selected
    return np.log(mass), included / mass, best_partition


@pytest.mark.parametrize("k", [1, 2, 3])
def test_likelihood_and_posteriors_equal_latent_enumeration(k):
    theta = np.array([0.12, 0.25, 0.38, 0.22, 0.17, 0.11])
    for basket in ((), (0,), (0, 1), (0, 1, 2)):
        exact_ll, exact_posterior, best = brute_force(basket, THEMES, theta, k)
        ll, posterior, partition = evaluate_basket(basket, THEMES, theta, k, 100000,
                                                   map_partition=True)
        np.testing.assert_allclose(ll, exact_ll, rtol=0, atol=1e-12)
        np.testing.assert_allclose(posterior, exact_posterior, rtol=0, atol=1e-12)
        assert partition == best


@pytest.mark.parametrize("k", [1, 2, 3])
def test_cover_groups_are_unique_and_match_pruned_latent_partitions(k):
    basket = (0, 1, 2)
    groups = _components_for_basket(basket, THEMES, k, max_groups=100000)[0].groups
    generated = {group.maximal for group in groups}
    assert len(generated) == len(groups)
    assert count_cover_groups(basket, THEMES, k, cap=100000) == (len(groups), False)
    assert count_cover_groups(basket, THEMES, k, cap=1) == (2, True)
    expected = set()
    for bits in product((0, 1), repeat=len(THEMES)):
        selected = tuple(idx for idx, bit in enumerate(bits) if bit)
        if not selected or not all(
            any(item in THEMES[idx].items for idx in selected) for item in basket
        ):
            continue
        maximal = tuple(sorted(
            idx for idx in selected if not any(
                idx != other and THEMES[idx].mask != THEMES[other].mask and
                THEMES[idx].mask & THEMES[other].mask == THEMES[idx].mask
                for other in selected
            )
        ))
        if max(sum(item in THEMES[idx].items for idx in maximal)
               for item in basket) <= k:
            expected.add(maximal)
    assert generated == expected
    for group in groups:
        proper = {
            idx for idx, theme in enumerate(THEMES) if idx not in group.maximal
            and any(theme.mask & THEMES[outer].mask == theme.mask
                    for outer in group.maximal)
        }
        assert set(group.proper) == proper


def test_cross_role_candidates_and_singletons():
    baskets = [(0, 1, 2), (0, 1), (0, 2), (1, 2), ()]
    themes = mine_dictionary(baskets, ["A0", "A0", "B"], max_length=3, min_count=1)
    assert [theme.items for theme in themes] == [
        (0,), (1,), (2,), (0, 1, 2), (0, 2), (1, 2)
    ]
    assert [theme.support for theme in themes] == [3, 3, 3, 1, 2, 2]


def test_deletion_score_matches_explicit_theta_zero():
    theta = np.array([0.12, 0.25, 0.38, 0.22, 0.17, 0.11])
    baskets = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2)]
    compiled = compile_observations(unique_baskets(baskets), THEMES, 3, 10000, 100000)
    ll, _, log_no = expectation(compiled, THEMES, theta, 3, 10000, workers=1)
    scores = deletion_scores(theta, log_no, len(baskets))
    for idx in range(len(THEMES)):
        remaining = THEMES[:idx] + THEMES[idx + 1:]
        reduced = np.delete(theta, idx)
        try:
            reduced_ll = sum(evaluate_basket(basket, remaining, reduced, 3, 10000)[0]
                             for basket in baskets)
        except ValueError as exc:
            if "cannot be covered" not in str(exc):
                raise
            reduced_ll = -np.inf
        np.testing.assert_allclose(scores[idx], ll - reduced_ll, rtol=0, atol=1e-10)


def test_parallel_e_step_and_resource_limit():
    theta = np.array([0.12, 0.25, 0.38, 0.22, 0.17, 0.11])
    compiled = compile_observations(
        unique_baskets([(), (0,), (0, 1), (0, 1, 2)]), THEMES, 3, 10000, 100000
    )
    serial = expectation(compiled, THEMES, theta, 3, 10000, workers=1)
    parallel = expectation(compiled, THEMES, theta, 3, 10000, workers=2)
    for left, right in zip(serial, parallel):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)
    _pool_setup(compiled, THEMES, 3, 10000)
    evaluator = BasketEvaluator(compiled, THEMES, 3, 10000, workers=2)

    class LocalPool:
        def map(self, func, tasks):
            return [func(task) for task in tasks]

    evaluator.pool = LocalPool()
    sharded = evaluator.evaluate(theta)
    for left, right in zip(serial, sharded):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)
    with pytest.raises(ComplexityLimitError):
        _components_for_basket((0, 1, 2), THEMES, 3, max_groups=1)


def test_em_increases_observed_score():
    baskets = [(0,), (1,), (0, 1), (0, 1), (), (0, 1)]
    themes = [Theme((0,), 4), Theme((1,), 4), Theme((0, 1), 3)]
    compiled = compile_observations(unique_baskets(baskets), themes, 2, 10000, 100000)
    initial = np.array([0.2, 0.2, 0.2])
    before = expectation(compiled, themes, initial, 2, 10000)[0]
    fitted, after, _, _ = em_fit(compiled, themes, initial, 2, 10000, 1, 1e-8, 1000)
    assert after >= before - 1e-9
    assert np.all((fitted > 0) & (fitted < 1))


def test_recovers_strong_cross_role_theme():
    baskets = [()] * 40 + [(0,)] * 5 + [(1,)] * 5 + [(0, 1)] * 50
    themes = mine_dictionary(baskets, ["A0", "B"], max_length=2, min_count=3)
    compiled = compile_observations(unique_baskets(baskets), themes, 3, 10000, 100000)
    theta, _, log_no, _ = em_fit(
        compiled, themes, np.array([0.2, 0.2, 0.2]), 3, 10000, 1,
        tolerance=1e-8, max_iterations=1000,
    )
    assert themes[2].items == (0, 1)
    assert theta[2] > 0.3
    assert deletion_scores(theta, log_no, 100)[2] > 0.5 * np.log(100)
