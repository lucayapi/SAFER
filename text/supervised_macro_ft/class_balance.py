"""Outils de rééquilibrage des classes pour supervised_macro_ft et baselines.

Ce module factorise la logique d'oversampling utilisée initialement dans
`macro_transfer.supervised_baseline` afin qu'elle soit partagée avec le
fine-tuning supervised_macro_ft.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np


def balanced_oversample_arrays(
    X: np.ndarray,
    y: Sequence[int],
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sur-échantillonne des matrices (baseline sklearn MLP)."""
    idx = balanced_oversample_indices(y, seed=seed)
    y_arr = np.asarray(y, dtype=np.int64)
    return X[idx], y_arr[idx]


def balanced_oversample_indices(
    y: Sequence[int],
    indices: Optional[Sequence[int]] = None,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Retourne un vecteur d'indices sur-échantillonné de façon équilibrée.

    - `y` : labels entiers (0…C-1) sur tout le corpus.
    - `indices` : sous-ensemble d'indices à considérer (train fold). Si None,
      on considère `range(len(y))`.
    - Algorithme : pour chaque classe présente dans `indices`, on tire avec
      remise jusqu'à atteindre la taille de la classe majoritaire.
    """
    y_arr = np.asarray(y, dtype=np.int64)
    if indices is None:
        idx = np.arange(len(y_arr), dtype=np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return idx

    sub_y = y_arr[idx]
    classes = np.unique(sub_y)
    if classes.size == 0:
        return idx

    rng = np.random.RandomState(int(seed))
    counts = np.array([np.sum(sub_y == cls) for cls in classes], dtype=np.int64)
    target_n = int(counts.max())
    parts: List[np.ndarray] = []
    for cls in classes:
        cls_idx = idx[sub_y == cls]
        if cls_idx.size == 0:
            continue
        parts.append(rng.choice(cls_idx, size=target_n, replace=True))
    if not parts:
        return idx
    all_idx = np.concatenate(parts)
    rng.shuffle(all_idx)
    return all_idx


def resolve_train_balance(
    model_cfg: Mapping[str, object],
) -> Tuple[bool, Optional[str]]:
    """Déduit la stratégie de rééquilibrage pour l'entraînement.

    Retourne (use_oversampling, class_weight_mode) où :
    - use_oversampling : bool indiquant si l'oversampling doit être appliqué
      sur les indices d'entraînement (fold ou fit final).
    - class_weight_mode : valeur à passer à `build_class_weights` pour la CE
      (ex. "balanced") ou None.

    Règle métier (choisie côté utilisateur) :
    - oversampling=true ET class_weight="balanced" → ValueError explicite.
    - oversampling=true → class_weight ignoré (doit être null / None).
    - class_weight="balanced" seul → CE pondérée (pas d'oversampling).
    - par défaut (aucun des deux) → pas de rééquilibrage.
    """
    raw_oversampling = model_cfg.get("oversampling", False)
    oversampling = bool(raw_oversampling)
    raw_class_weight = model_cfg.get("class_weight")
    class_weight_mode: Optional[str]
    if raw_class_weight is None:
        class_weight_mode = None
    else:
        class_weight_mode = str(raw_class_weight).strip().lower()
        if class_weight_mode in ("none", "null", ""):
            class_weight_mode = None

    if oversampling and class_weight_mode == "balanced":
        raise ValueError(
            "Configuration incohérente : oversampling=true et class_weight='balanced'. "
            "Choisir soit l'oversampling, soit class_weight=balanced, mais pas les deux."
        )

    if oversampling:
        # L'oversampling remplace le besoin de pondération CE.
        return True, None

    return False, class_weight_mode

