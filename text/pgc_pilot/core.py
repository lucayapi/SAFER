"""Graph loss, grouped batching, and accident-level paired inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROLES = ("A0", "A1", "B", "C")
EDGES = ((0, 1), (0, 2), (1, 2), (2, 3))


def accident_batches(
    indices: Iterable[int], groups: np.ndarray, *, batch_size: int,
    chunk_size: int, seed: int,
) -> list[list[int]]:
    """Visit every selected unit once; pack distinct accident chunks per batch."""
    if not 0 < chunk_size <= batch_size:
        raise ValueError("chunk_size must be in [1, batch_size]")
    rng = np.random.default_rng(seed)
    by_accident: dict[str, list[int]] = defaultdict(list)
    selected = [int(i) for i in indices]
    for i in selected:
        by_accident[str(groups[i])].append(i)
    chunks: list[tuple[str, list[int]]] = []
    for accident, members in by_accident.items():
        rng.shuffle(members)
        chunks.extend((accident, members[j:j + chunk_size]) for j in range(0, len(members), chunk_size))
    rng.shuffle(chunks)
    batches: list[list[int]] = []
    current: list[int] = []
    seen: set[str] = set()
    for accident, chunk in chunks:
        if current and (len(current) + len(chunk) > batch_size or accident in seen):
            batches.append(current)
            current, seen = [], set()
        current.extend(chunk)
        seen.add(accident)
    if current:
        batches.append(current)
    flat = [i for batch in batches for i in batch]
    if len(flat) != len(selected) or set(flat) != set(selected) or len(flat) != len(set(flat)):
        raise AssertionError("Grouped sampler lost or duplicated a unit")
    if any(len(batch) > batch_size for batch in batches):
        raise AssertionError("Grouped sampler exceeded batch_size")
    return batches


def pgc_loss(
    z: torch.Tensor, labels: torch.Tensor, accidents: list[str],
    edges: tuple[tuple[int, int], ...] = EDGES, eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Mean pairwise angular discrepancy for eligible accident role contrasts.

    Each local role prototype averages L2-normalized unit embeddings. A role
    contrast contributes only if at least two accidents have both role labels
    and a non-degenerate prototype difference. No target labels enter here.
    """
    if z.ndim != 2 or len(labels) != len(z) or len(accidents) != len(z):
        raise ValueError("Incompatible graph-loss inputs")
    z_unit = F.normalize(z.float(), p=2, dim=-1, eps=eps)
    label_values = labels.detach().cpu().tolist()
    by_accident: dict[str, list[int]] = defaultdict(list)
    for i, accident in enumerate(accidents):
        by_accident[str(accident)].append(i)
    prototypes: dict[tuple[str, int], torch.Tensor] = {}
    for accident, positions in by_accident.items():
        for role in range(len(ROLES)):
            selected = [i for i in positions if label_values[i] == role]
            if selected:
                prototypes[(accident, role)] = z_unit[selected].mean(dim=0)
    terms: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for source, target in edges:
        name = f"{ROLES[source]}->{ROLES[target]}"
        directions: list[torch.Tensor] = []
        for accident in by_accident:
            a = prototypes.get((accident, source))
            b = prototypes.get((accident, target))
            if a is None or b is None:
                continue
            difference = b - a
            if float(torch.linalg.vector_norm(difference.detach())) <= eps:
                continue
            directions.append(F.normalize(difference, p=2, dim=0, eps=eps))
        n = len(directions)
        counts[name] = n * (n - 1) // 2
        if n >= 2:
            matrix = torch.stack(directions)
            similarity = matrix @ matrix.T
            upper = torch.triu_indices(n, n, offset=1, device=z.device)
            terms.append((1.0 - similarity[upper[0], upper[1]]).mean())
    return (torch.stack(terms).mean() if terms else z.sum() * 0.0), counts


def confusion_by_accident(
    frame, accident_order: list[str], *, n_roles: int = 4,
) -> np.ndarray:
    """Return one true-by-predicted confusion matrix per accident."""
    accident_index = {name: i for i, name in enumerate(accident_order)}
    result = np.zeros((len(accident_order), n_roles, n_roles), dtype=np.int64)
    for accident, true, predicted in zip(frame["accident_id"], frame["true_id"], frame["pred_id"]):
        result[accident_index[str(accident)], int(true), int(predicted)] += 1
    return result


def balanced_accuracy_from_confusions(counts: np.ndarray) -> np.ndarray:
    """BA for arbitrary leading dimensions ending in a 4x4 confusion matrix."""
    totals = counts.sum(axis=-1)
    correct = np.diagonal(counts, axis1=-2, axis2=-1)
    if np.any(totals == 0):
        raise ValueError("Bootstrap draw is missing a role; cannot compute four-role BA")
    return (correct / totals).mean(axis=-1)
