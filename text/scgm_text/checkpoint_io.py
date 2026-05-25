"""Chargement de checkpoints SCGM end2end."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from scgm_text.projection import projection_from_checkpoint_args
from scgm_text.scgm_text_model import SCGMTextModel


def load_scgm_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> Tuple[SCGMTextModel, Dict[str, Any], Dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    checkpoint_args = dict(checkpoint.get("args", {}))
    if checkpoint_args.get("input_mode") == "precomputed_embeddings":
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses removed precomputed_embeddings pipeline. "
            "Retrain with end2end SCGM."
        )
    if checkpoint.get("pipeline") != "end2end_text" and not checkpoint_args.get(
        "backbone_model_name_or_path"
    ) and not checkpoint_args.get("backbone_name"):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not an end2end SCGM checkpoint (missing backbone)."
        )

    backbone = (
        checkpoint_args.get("backbone_model_name_or_path")
        or checkpoint_args.get("backbone_name")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    proj = projection_from_checkpoint_args(checkpoint_args)
    model = SCGMTextModel(
        hiddim=int(checkpoint_args.get("hiddim", 128)),
        num_classes=int(checkpoint_args.get("n_class", 4)),
        num_subclasses=int(checkpoint_args.get("n_subclass", 32)),
        backbone_model_name_or_path=str(backbone),
        projection=proj,
        pooling=checkpoint_args.get("pooling", "mean"),
        gradient_checkpointing=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint_args, checkpoint
