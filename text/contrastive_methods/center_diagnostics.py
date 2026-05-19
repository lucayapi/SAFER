"""Diagnostics et export des centres SoftTriple (effectifs uniques)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from contrastive_methods.distance import normalize_distance_metric
from scgm_text.dataset_text_embeddings import ID2LABEL


def _to_numpy(centers: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(centers, torch.Tensor):
        return centers.detach().cpu().numpy()
    return np.asarray(centers, dtype=np.float64)


def _union_find_groups(n: int, edges: List[tuple[int, int]]) -> List[List[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)
    groups_map: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups_map.setdefault(r, []).append(i)
    return list(groups_map.values())


def compute_effective_unique_centers(
    centers: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    *,
    metric: str = "euclidean",
    distance_threshold: float = 0.05,
    similarity_threshold: float = 0.995,
    normalize_centers: bool = False,
) -> Dict[str, Any]:
    """
    Regroupe les centres initiaux par classe en composantes connexes (fusion).
    Retourne summary, per_class, unique_centers (liste par classe).
    """
    arr = _to_numpy(centers)
    if arr.ndim != 3:
        raise ValueError(f"centers attendu [C, K, D], reçu shape {arr.shape}")
    metric = normalize_distance_metric(metric)
    num_classes, k_initial, dim = arr.shape
    if class_names is None:
        class_names = [ID2LABEL.get(c, str(c)) for c in range(num_classes)]
    elif len(class_names) < num_classes:
        class_names = list(class_names) + [
            ID2LABEL.get(c, str(c)) for c in range(len(class_names), num_classes)
        ]

    per_class: List[Dict[str, Any]] = []
    unique_by_class: List[np.ndarray] = []
    total_effective = 0

    for c in range(num_classes):
        cls_centers = arr[c].copy()
        if normalize_centers:
            norms = np.linalg.norm(cls_centers, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            cls_centers = cls_centers / norms

        edges: List[tuple[int, int]] = []
        for i in range(k_initial):
            for j in range(i + 1, k_initial):
                if metric == "euclidean":
                    dist = float(np.linalg.norm(cls_centers[i] - cls_centers[j]))
                    if dist <= float(distance_threshold):
                        edges.append((i, j))
                else:
                    sim = float(np.dot(cls_centers[i], cls_centers[j]))
                    if sim >= float(similarity_threshold):
                        edges.append((i, j))

        if not edges and k_initial > 0:
            groups = [[i] for i in range(k_initial)]
        else:
            groups = _union_find_groups(k_initial, edges)

        group_assignments = [-1] * k_initial
        group_sizes: List[int] = []
        class_unique: List[np.ndarray] = []
        for eff_id, members in enumerate(sorted(groups, key=lambda g: min(g))):
            for m in members:
                group_assignments[m] = eff_id
            group_sizes.append(len(members))
            mean_center = cls_centers[members].mean(axis=0)
            if normalize_centers:
                nrm = np.linalg.norm(mean_center)
                if nrm > 1e-12:
                    mean_center = mean_center / nrm
            class_unique.append(mean_center)

        n_eff = len(groups)
        total_effective += n_eff
        per_class.append(
            {
                "class_id": c,
                "class_name": class_names[c] if c < len(class_names) else str(c),
                "k_initial": k_initial,
                "n_effective_unique": n_eff,
                "group_sizes": group_sizes,
                "group_assignments": group_assignments,
            }
        )
        unique_by_class.append(
            np.stack(class_unique, axis=0) if class_unique else np.zeros((0, dim))
        )

    eff_counts = [p["n_effective_unique"] for p in per_class]
    summary = {
        "num_classes": num_classes,
        "centers_per_class_initial": k_initial,
        "total_initial_centers": num_classes * k_initial,
        "total_effective_unique_centers": total_effective,
        "mean_effective_unique_centers_per_class": float(np.mean(eff_counts)) if eff_counts else 0.0,
        "min_effective_unique_centers_per_class": int(min(eff_counts)) if eff_counts else 0,
        "max_effective_unique_centers_per_class": int(max(eff_counts)) if eff_counts else 0,
        "metric": metric,
        "distance_threshold": float(distance_threshold),
        "similarity_threshold": float(similarity_threshold),
        "normalize_centers": bool(normalize_centers),
    }
    return {
        "summary": summary,
        "per_class": per_class,
        "unique_centers": unique_by_class,
    }


def infer_fold_id(output_dir: Union[str, Path]) -> Optional[int]:
    path_str = str(output_dir).replace("\\", "/")
    match = re.search(r"fold_(\d+)", path_str)
    if match:
        return int(match.group(1))
    return None


def export_softtriple_center_artifacts(
    centers: Union[torch.Tensor, np.ndarray],
    output_dir: Union[str, Path],
    *,
    fold_id: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    metric: str = "euclidean",
    distance_threshold: float = 0.05,
    similarity_threshold: float = 0.995,
    normalize_centers: bool = False,
    hyperparams: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Écrit softtriple_centers_*.pt/csv/json dans {output_dir}/centers/."""
    out_root = Path(output_dir) / "centers"
    out_root.mkdir(parents=True, exist_ok=True)

    if fold_id is None:
        fold_id = infer_fold_id(output_dir)

    raw_np = _to_numpy(centers)
    if diagnostics is None:
        diagnostics = compute_effective_unique_centers(
            raw_np,
            class_names=class_names,
            metric=metric,
            distance_threshold=distance_threshold,
            similarity_threshold=similarity_threshold,
            normalize_centers=normalize_centers,
        )

    torch.save(
        {"centers": torch.from_numpy(raw_np), "fold": fold_id},
        out_root / "softtriple_centers_raw.pt",
    )

    unique_list = diagnostics["unique_centers"]
    max_k = max((u.shape[0] for u in unique_list), default=0)
    num_classes = raw_np.shape[0]
    dim = raw_np.shape[2]
    padded = np.full((num_classes, max_k, dim), np.nan, dtype=np.float64)
    for c, u in enumerate(unique_list):
        if u.shape[0] > 0:
            padded[c, : u.shape[0]] = u
    torch.save(
        {
            "unique_centers_padded": torch.from_numpy(padded),
            "unique_centers_per_class": [torch.from_numpy(u) for u in unique_list],
            "fold": fold_id,
        },
        out_root / "softtriple_effective_centers.pt",
    )

    eff_rows: List[Dict[str, Any]] = []
    assign_rows: List[Dict[str, Any]] = []
    for pc in diagnostics["per_class"]:
        c_id = pc["class_id"]
        c_name = pc["class_name"]
        u_centers = unique_list[c_id]
        for eff_id, gsize in enumerate(pc["group_sizes"]):
            row: Dict[str, Any] = {
                "fold": fold_id if fold_id is not None else "",
                "class_id": c_id,
                "class_name": c_name,
                "effective_center_id": eff_id,
                "group_size": gsize,
            }
            vec = u_centers[eff_id]
            for d, val in enumerate(vec):
                row[f"dim_{d:04d}"] = float(val)
            eff_rows.append(row)
        for init_id, eff_id in enumerate(pc["group_assignments"]):
            assign_rows.append(
                {
                    "fold": fold_id if fold_id is not None else "",
                    "class_id": c_id,
                    "class_name": c_name,
                    "initial_center_id": init_id,
                    "effective_center_id": eff_id,
                }
            )

    pd.DataFrame(eff_rows).to_csv(out_root / "softtriple_effective_centers.csv", index=False)
    pd.DataFrame(assign_rows).to_csv(out_root / "softtriple_center_assignments.csv", index=False)

    payload = {
        "hyperparameters": hyperparams or {},
        "diagnostics": {
            "summary": diagnostics["summary"],
            "per_class": diagnostics["per_class"],
        },
    }
    with open(out_root / "softtriple_center_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_root
