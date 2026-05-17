"""Chargement de checkpoints SCGM texte."""

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
    input_mode = checkpoint_args.get("input_mode", "precomputed_embeddings")
    input_dim = int(checkpoint.get("input_dim", checkpoint_args.get("hiddim", 128)))
    proj = projection_from_checkpoint_args(checkpoint_args)
    model = SCGMTextModel(
        input_dim=input_dim,
        hiddim=int(checkpoint_args.get("hiddim", input_dim)),
        num_classes=int(checkpoint_args.get("n_class", 4)),
        num_subclasses=int(checkpoint_args.get("n_subclass", 32)),
        projection=proj,
        input_mode=input_mode,
        backbone_model_name_or_path=checkpoint_args.get("backbone_model_name_or_path"),
        pooling=checkpoint_args.get("pooling", "mean"),
        freeze_backbone=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint_args, checkpoint
