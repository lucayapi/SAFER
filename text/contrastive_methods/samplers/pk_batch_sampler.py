"""
PKBatchSampler (P labels × K exemples) pour Batch Hard Triplet.

BatchTriplet nécessite positifs et négatifs in-batch. Avec pred_label={A0,A1,B,C},
le batch recommandé est P=4, K=batch_size/4. Un batch mono-classe rend la loss
impossible (aucun hard negative).

Note : sentence-transformers 3.4.1 ``GroupByLabelBatchSampler`` produit des batches
homogènes (mono-classe) — incompatible avec BatchHardTripletLoss.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Sequence, Union

import torch
from torch.utils.data import BatchSampler

from contrastive_methods.config import ContrastiveConfig


class SetEpochMixin:
    """Compatibilité Trainer ST : ``set_epoch`` avant chaque epoch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


@dataclass(frozen=True)
class PKParams:
    sampler: str
    batch_size: int
    classes_per_batch: int
    samples_per_class: int
    drop_last: bool = True
    seed: int = 42


def label_counts_for_indices(
    labels: Sequence[int],
    indices: Sequence[int],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for idx in indices:
        key = str(int(labels[idx]))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def validate_pk_batch(
    label_counts: Mapping[str, int],
    *,
    min_labels: int = 2,
    expected_classes: int | None = None,
    expected_samples_per_class: int | None = None,
) -> None:
    n_labels = len(label_counts)
    if n_labels < min_labels:
        raise ValueError(
            f"Batch mono-classe ou insuffisant : {label_counts!r} "
            f"(minimum {min_labels} labels distincts requis pour BatchTriplet)."
        )
    if expected_classes is not None and n_labels != expected_classes:
        raise ValueError(
            f"Batch attend {expected_classes} labels distincts, obtenu {n_labels} : {label_counts!r}."
        )
    if expected_samples_per_class is not None:
        for lab, cnt in label_counts.items():
            if cnt != expected_samples_per_class:
                raise ValueError(
                    f"Label {lab} : attendu {expected_samples_per_class} exemples, obtenu {cnt} "
                    f"dans {label_counts!r}."
                )


def resolve_batch_triplet_pk_params(
    cfg: ContrastiveConfig,
    train_labels: Sequence[int],
) -> PKParams:
    sampler = (cfg.batch_triplet_sampler or "pk").strip().lower()
    if sampler != "pk":
        raise ValueError(
            f"batch_triplet.sampler={sampler!r} non supporté pour PK ; utiliser 'pk'."
        )

    labels_arr = [int(x) for x in train_labels]
    unique_labels = sorted(set(labels_arr))
    n_unique = len(unique_labels)
    if n_unique < 2:
        raise ValueError(
            f"Batch Triplet PK : au moins 2 labels distincts requis dans le train split, "
            f"obtenu {n_unique}."
        )

    p = cfg.batch_triplet_classes_per_batch
    if p is None:
        p = min(4, n_unique)
    p = int(p)

    k = cfg.batch_triplet_samples_per_class
    if k is None:
        if cfg.batch_size % p != 0:
            raise ValueError(
                f"batch_size={cfg.batch_size} doit être divisible par classes_per_batch={p} "
                f"(ou définir samples_per_class explicitement)."
            )
        k = cfg.batch_size // p
    k = int(k)

    batch_size = p * k
    if cfg.batch_size != batch_size:
        raise ValueError(
            f"batch_size={cfg.batch_size} incompatible avec PK : "
            f"classes_per_batch={p} × samples_per_class={k} = {batch_size}. "
            f"Ajuster training.batch_size ou batch_triplet.classes_per_batch / samples_per_class."
        )
    if p < 2:
        raise ValueError(f"classes_per_batch doit être >= 2, obtenu {p}.")
    if k < 2:
        raise ValueError(f"samples_per_class doit être >= 2, obtenu {k}.")
    if p > n_unique:
        raise ValueError(
            f"classes_per_batch={p} > nombre de labels train={n_unique} "
            f"({unique_labels})."
        )

    return PKParams(
        sampler=sampler,
        batch_size=batch_size,
        classes_per_batch=p,
        samples_per_class=k,
        drop_last=True,
        seed=int(cfg.seed),
    )


class PKBatchSampler(SetEpochMixin, BatchSampler):
    """
    P labels distincts × K exemples par label par batch.

    Échantillonne avec remplacement si une classe a moins de K exemples.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        classes_per_batch: int,
        samples_per_class: int,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.labels_list = [int(x) for x in labels]
        self.dataset_size = len(self.labels_list)
        self.batch_size = int(batch_size)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        expected = self.classes_per_batch * self.samples_per_class
        if self.batch_size != expected:
            raise ValueError(
                f"batch_size={self.batch_size} doit valoir "
                f"classes_per_batch×samples_per_class={expected}."
            )
        if self.classes_per_batch < 2:
            raise ValueError("classes_per_batch doit être >= 2.")
        if self.samples_per_class < 2:
            raise ValueError("samples_per_class doit être >= 2.")

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(self.labels_list):
            self.label_to_indices[int(lab)].append(idx)
        self.available_labels = sorted(self.label_to_indices.keys())
        if self.classes_per_batch > len(self.available_labels):
            raise ValueError(
                f"classes_per_batch={self.classes_per_batch} > "
                f"labels disponibles={len(self.available_labels)}."
            )

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.generator.manual_seed(self.seed + self.epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return max(1, self.dataset_size // self.batch_size)
        full = self.dataset_size // self.batch_size
        remainder = self.dataset_size % self.batch_size
        if remainder >= self.classes_per_batch * 2:
            return full + 1
        return max(1, full)

    def __iter__(self) -> Iterator[List[int]]:
        self.generator.manual_seed(self.seed + self.epoch)
        n_batches = len(self)
        labels_tensor = torch.tensor(self.available_labels, dtype=torch.long)

        for _ in range(n_batches):
            perm = torch.randperm(len(self.available_labels), generator=self.generator)
            chosen = [self.available_labels[int(i)] for i in perm[: self.classes_per_batch]]

            batch_indices: List[int] = []
            for lab in chosen:
                pool = self.label_to_indices[lab]
                if len(pool) >= self.samples_per_class:
                    pick = torch.randperm(len(pool), generator=self.generator)[
                        : self.samples_per_class
                    ]
                    batch_indices.extend(int(pool[int(i)]) for i in pick)
                else:
                    picks = torch.randint(
                        len(pool),
                        (self.samples_per_class,),
                        generator=self.generator,
                    )
                    batch_indices.extend(int(pool[int(i)]) for i in picks)

            perm_idx = torch.randperm(len(batch_indices), generator=self.generator)
            batch = [batch_indices[int(i)] for i in perm_idx]
            yield batch


def debug_pk_sampler_batches(
    sampler: PKBatchSampler,
    labels: Sequence[int],
    *,
    n_batches: int = 5,
    expected_classes: int | None = None,
    expected_samples_per_class: int | None = None,
) -> None:
    """Affiche les premiers batchs et vérifie qu'ils sont multi-label."""
    expected_classes = expected_classes or sampler.classes_per_batch
    expected_samples_per_class = expected_samples_per_class or sampler.samples_per_class

    for batch_id, batch in enumerate(sampler):
        if batch_id >= n_batches:
            break
        counts = label_counts_for_indices(labels, batch)
        print(
            f"[PKSampler DEBUG] batch={batch_id} batch_size={len(batch)} "
            f"labels={json.dumps(counts, sort_keys=True)}",
            flush=True,
        )
        validate_pk_batch(
            counts,
            expected_classes=expected_classes,
            expected_samples_per_class=expected_samples_per_class,
        )


def train_label_distribution(labels: Sequence[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for lab in labels:
        key = str(int(lab))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def build_pk_batch_sampler(
    labels: Sequence[int],
    pk_params: PKParams,
) -> PKBatchSampler:
    return PKBatchSampler(
        labels=labels,
        batch_size=pk_params.batch_size,
        classes_per_batch=pk_params.classes_per_batch,
        samples_per_class=pk_params.samples_per_class,
        drop_last=pk_params.drop_last,
        seed=pk_params.seed,
    )
