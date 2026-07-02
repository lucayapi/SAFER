"""Save/load checkpoint supervised_macro_ft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from supervised_macro_ft.backbone_scaler import BackboneScaler
from supervised_macro_ft.model import SupervisedMacroModel


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _backbone_state_dict(model: SupervisedMacroModel) -> Dict[str, Any]:
    encoder = getattr(model.backbone, "encoder", None)
    if encoder is not None:
        return encoder.state_dict()
    return model.backbone.state_dict()


def _load_backbone_state_dict(model: SupervisedMacroModel, state: Dict[str, Any]) -> None:
    encoder = getattr(model.backbone, "encoder", None)
    if encoder is not None:
        encoder.load_state_dict(state, strict=False)
    else:
        model.backbone.load_state_dict(state, strict=False)


def _is_legacy_checkpoint(cfg: Dict[str, Any], ckpt_dir: Path) -> bool:
    if (ckpt_dir / "projector.pt").is_file():
        return False
    projection = cfg.get("projection")
    if projection is None:
        return True
    return str(projection).strip().lower() in ("none", "null", "", "legacy")


def _attach_backbone_scaler_from_config(model: SupervisedMacroModel, cfg: Dict[str, Any]) -> None:
    scaler = BackboneScaler.from_config(cfg)
    model.set_backbone_scaler(scaler)


def save_checkpoint(
    model: SupervisedMacroModel,
    out_dir: str | Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(_backbone_state_dict(model), out / "hf_model.bin")
    torch.save(model.classifier.state_dict(), out / "classifier_head.pt")
    if model.use_projector:
        torch.save(model.projector.state_dict(), out / "projector.pt")
    payload = dict(config or {})
    payload.setdefault("backbone_name", model.backbone_name)
    payload.setdefault("num_classes", model.num_classes)
    payload.setdefault("pooling", model.pooling)
    payload.setdefault("backbone_trainable", model.backbone_trainable)
    payload.setdefault("train_last_n_layers", model.train_last_n_layers)
    payload.setdefault("standardize_backbone", bool(model.backbone_scaler is not None))
    if model.backbone_scaler is not None:
        payload["backbone_scaler"] = model.backbone_scaler.to_dict()
    if model.use_projector:
        payload.setdefault("projection", model.projection_name)
        payload.setdefault("hiddim", model.hiddim)
        payload.setdefault("dropout", model.dropout)
        if model.proj_hidden is not None:
            payload.setdefault("proj_hidden", model.proj_hidden)
        if model.proj_bottleneck is not None:
            payload.setdefault("proj_bottleneck", model.proj_bottleneck)
        payload.setdefault("proj_alpha", model.proj_alpha)
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

    legacy = _is_legacy_checkpoint(cfg, ckpt_dir)
    model = SupervisedMacroModel(
        backbone_name=str(cfg["backbone_name"]),
        num_classes=int(cfg.get("num_classes", cfg.get("n_classes", 4))),
        pooling=str(cfg.get("pooling", "mean")),
        backbone_trainable=False,
        train_last_n_layers=None,
        gradient_checkpointing=False,
        projection=None if legacy else str(cfg.get("projection", "linear")),
        hiddim=int(cfg.get("hiddim", 512)),
        dropout=float(cfg.get("dropout", 0.0)),
        proj_hidden=cfg.get("proj_hidden"),
        proj_bottleneck=cfg.get("proj_bottleneck"),
        proj_alpha=float(cfg.get("proj_alpha", 0.1)),
        standardize_backbone=bool(cfg.get("standardize_backbone", False)),
    )
    loc = map_location or device
    backbone_path = ckpt_dir / "hf_model.bin"
    head_path = ckpt_dir / "classifier_head.pt"
    if not backbone_path.is_file():
        raise FileNotFoundError(f"hf_model.bin manquant dans {ckpt_dir}")
    if not head_path.is_file():
        raise FileNotFoundError(f"classifier_head.pt manquant dans {ckpt_dir}")
    _load_backbone_state_dict(model, _torch_load(backbone_path, loc))
    model.classifier.load_state_dict(_torch_load(head_path, loc))
    projector_path = ckpt_dir / "projector.pt"
    if projector_path.is_file():
        model.projector.load_state_dict(_torch_load(projector_path, loc))
    _attach_backbone_scaler_from_config(model, cfg)
    model.to(torch.device(device if device != "cpu" or map_location is None else map_location))
    model.eval()
    return model


def read_checkpoint_config(checkpoint_dir: str | Path) -> Dict[str, Any]:
    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"config.json manquant : {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
