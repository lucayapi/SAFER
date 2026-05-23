"""Tests PKBatchSampler (P labels × K exemples)."""

from __future__ import annotations

import pytest

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.samplers.pk_batch_sampler import (
    PKBatchSampler,
    debug_pk_sampler_batches,
    label_counts_for_indices,
    resolve_batch_triplet_pk_params,
    validate_pk_batch,
)


def _synthetic_labels(n_per_class: int = 20) -> list[int]:
    labels: list[int] = []
    for lab in range(4):
        labels.extend([lab] * n_per_class)
    return labels


def test_pk_batch_has_four_labels_and_k_samples():
    labels = _synthetic_labels()
    sampler = PKBatchSampler(
        labels=labels,
        batch_size=16,
        classes_per_batch=4,
        samples_per_class=4,
        seed=42,
    )
    batch = next(iter(sampler))
    assert len(batch) == 16
    counts = label_counts_for_indices(labels, batch)
    validate_pk_batch(counts, expected_classes=4, expected_samples_per_class=4)


def test_pk_batch_sampling_with_replacement():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    sampler = PKBatchSampler(
        labels=labels,
        batch_size=8,
        classes_per_batch=4,
        samples_per_class=2,
        seed=0,
    )
    batch = next(iter(sampler))
    counts = label_counts_for_indices(labels, batch)
    validate_pk_batch(counts, expected_classes=4, expected_samples_per_class=2)


def test_batch_size_must_equal_p_times_k():
    with pytest.raises(ValueError, match="batch_size"):
        PKBatchSampler(
            labels=[0, 0, 1, 1],
            batch_size=10,
            classes_per_batch=4,
            samples_per_class=4,
        )


def test_resolve_batch_triplet_pk_params_defaults():
    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=".",
        batch_size=64,
        batch_triplet_sampler="pk",
        batch_triplet_classes_per_batch=4,
        batch_triplet_samples_per_class=16,
    )
    params = resolve_batch_triplet_pk_params(cfg, [0, 0, 1, 1, 2, 2, 3, 3])
    assert params.classes_per_batch == 4
    assert params.samples_per_class == 16
    assert params.batch_size == 64


def test_resolve_rejects_mismatched_batch_size():
    cfg = ContrastiveConfig(
        method_name="batch_triplet",
        dataset_path=".",
        batch_size=32,
        batch_triplet_sampler="pk",
        batch_triplet_classes_per_batch=4,
        batch_triplet_samples_per_class=16,
    )
    with pytest.raises(ValueError, match="incompatible"):
        resolve_batch_triplet_pk_params(cfg, [0, 0, 1, 1, 2, 2, 3, 3])


def test_set_epoch_changes_batch_order():
    labels = _synthetic_labels()
    s0 = PKBatchSampler(labels, 16, 4, 4, seed=7)
    s1 = PKBatchSampler(labels, 16, 4, 4, seed=7)
    s0.set_epoch(0)
    s1.set_epoch(1)
    b0 = next(iter(s0))
    b1 = next(iter(s1))
    assert b0 != b1


def test_debug_pk_sampler_batches_runs(capsys):
    labels = _synthetic_labels()
    sampler = PKBatchSampler(labels, 64, 4, 16, seed=1)
    debug_pk_sampler_batches(sampler, labels, n_batches=2)
    out = capsys.readouterr().out
    assert "[PKSampler DEBUG]" in out


def test_validate_rejects_mono_label():
    with pytest.raises(ValueError, match="mono-classe"):
        validate_pk_batch({"0": 64})
