"""MALT transfer utilities (p0 computation, initialization) and training entry."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from malt_text.malt_dataset import MALTTargetDataset
from malt_text.malt_model import MALTTargetModel
from scgm_text.dataset_text_embeddings import ID2LABEL
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.utils_io import save_numpy

__all__ = [
    "compute_p0_target",
    "initialize_nu_global",
    "save_p0_artifacts",
    "save_malt_checkpoint",
    "run_malt_training",
    "run_malt_em_training",
]


def run_malt_training(args):
    from malt_text.malt_em_training import run_malt_em_training

    return run_malt_em_training(args)


def run_malt_em_training(args):
    from malt_text.malt_em_training import run_malt_em_training as _run

    return _run(args)


def compute_p0_target(
    source_model: SCGMEmbeddingNet,
    data_loader: DataLoader,
    device: torch.device,
    tau_macro: float,
) -> Tuple[np.ndarray, np.ndarray]:
    projected: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                embeddings = batch["embedding"]
            else:
                embeddings, _ = batch
            embeddings = embeddings.to(device)
            features = source_model(embeddings)
            mu_y = source_model.mu_y
            logits = torch.nn.functional.normalize(features, p=2, dim=1) @ torch.nn.functional.normalize(
                mu_y, p=2, dim=1
            ).transpose(0, 1)
            prob = torch.softmax(logits / tau_macro, dim=1)
            projected.append(features.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())
    return np.concatenate(projected, axis=0), np.concatenate(probs, axis=0)


def initialize_nu_global(
    projected_source: np.ndarray,
    num_subclasses: int,
    hiddim: int,
    seed: int,
) -> np.ndarray:
    if projected_source.shape[0] < num_subclasses:
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(num_subclasses, hiddim)).astype(np.float32)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=num_subclasses,
            random_state=seed,
            batch_size=min(4096, projected_source.shape[0]),
            n_init="auto",
        )
        centers = kmeans.fit(projected_source).cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return centers / norms


def save_p0_artifacts(
    output_dir: str,
    metadata_df: pd.DataFrame,
    p0: np.ndarray,
) -> None:
    save_numpy(p0, os.path.join(output_dir, "p0_y_target.npy"))
    frame = metadata_df.copy()
    for macro_id, macro_name in ID2LABEL.items():
        frame[f"p0_{macro_name}"] = p0[:, macro_id]
    frame["p0_macro_id"] = p0.argmax(axis=1)
    frame["p0_macro_name"] = [ID2LABEL[int(value)] for value in frame["p0_macro_id"]]
    frame["p0_confidence"] = p0.max(axis=1)
    frame.to_csv(os.path.join(output_dir, "source_macro_pseudo_labels.csv"), index=False)


def save_malt_checkpoint(
    path: str,
    model: MALTTargetModel,
    args: Any,
    input_dim: int,
    label2id: Dict[str, int],
    source_checkpoint: str,
    mu_source: np.ndarray,
    p0: np.ndarray,
    resolved_paths: Dict[str, str],
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args) if hasattr(args, "__dict__") else dict(args),
            "input_dim": input_dim,
            "label2id": label2id,
            "source_checkpoint": source_checkpoint,
            "mu_y_source": mu_source,
            "p0_y_target": p0,
            "resolved_paths": resolved_paths,
            "malt_mode": "em_strict",
        },
        path,
    )
