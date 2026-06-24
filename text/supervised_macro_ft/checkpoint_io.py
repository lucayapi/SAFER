"""Save/load checkpoint supervised_macro_ft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from supervised_macro_ft.model import SupervisedMacroModel


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    model: SupervisedMacroModel,
    out_dir: str | Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.encoder.state_dict(), out / "hf_model.bin")
    torch.save(model.classifier.state_dict(), out / "classifier_head.pt")
    payload = dict(config or {})
    payload.setdefault("backbone_name", model.backbone_name)
    payload.setdefault("num_classes", model.num_classes)
    payload.setdefault("pooling", model.pooling)
    payload.setdefault("backbone_trainable", model.backbone_trainable)
    payload.setdefault("train_last_n_layers", model.train_last_n_layers)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    map_location: Optional[str] = None,
) -> SupervisedMacroModel:
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    config_path = ckpt_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json manquant dans {ckpt_dir}")
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    model = SupervisedMacroModel(
        backbone_name=str(cfg["backbone_name"]),
        num_classes=int(cfg.get("num_classes", 4)),
        pooling=str(cfg.get("pooling", "mean")),
        backbone_trainable=False,
        train_last_n_layers=None,
        gradient_checkpointing=False,
    )
    loc = map_location or device
    backbone_path = ckpt_dir / "hf_model.bin"
    head_path = ckpt_dir / "classifier_head.pt"
    if not backbone_path.is_file():
        raise FileNotFoundError(f"hf_model.bin manquant dans {ckpt_dir}")
    if not head_path.is_file():
        raise FileNotFoundError(f"classifier_head.pt manquant dans {ckpt_dir}")
    model.backbone.encoder.load_state_dict(
        _torch_load(backbone_path, loc),
        strict=False,
    )
    model.classifier.load_state_dict(_torch_load(head_path, loc))
    model.to(torch.device(device if device != "cpu" or map_location is None else map_location))
    model.eval()
    return model


def read_checkpoint_config(checkpoint_dir: str | Path) -> Dict[str, Any]:
    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"config.json manquant : {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
