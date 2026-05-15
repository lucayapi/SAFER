import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID, VALID_LABELS
from scgm_text.projection import projection_from_checkpoint_args
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.utils_io import load_json, set_seed

EPS = 1e-8


def as_repo_path(repo_root: str, path: str) -> str:
    """Absolute path under repo_root when path is relative (CLI / Jupyter cwd-safe)."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


def _expand_repo_candidates(path: Optional[str], repo_root: Optional[str]) -> List[str]:
    """Return paths to try: repo-anchored first if repo_root is set, then as given."""
    if not path:
        return []
    out: List[str] = []
    if repo_root and not os.path.isabs(path):
        anchored = os.path.normpath(os.path.join(repo_root, path))
        if anchored not in out:
            out.append(anchored)
    if path not in out:
        out.append(path)
    return out


def resolve_existing_path(
    primary: str,
    alt: Optional[str],
    kind: str,
    repo_root: Optional[str] = None,
) -> str:
    candidates: List[str] = []
    for path in (primary, alt):
        for candidate in _expand_repo_candidates(path, repo_root):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    existing = [path for path in candidates if os.path.isfile(path)]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        return existing[0]
    raise FileNotFoundError(
        f"No {kind} file found. Checked: {', '.join(candidates)}"
    )


def resolve_target_embedding_csv(
    primary: str,
    alt: str,
    search_dir: str = "embeddings",
    repo_root: Optional[str] = None,
) -> str:
    try:
        return resolve_existing_path(primary, alt, "target embedding CSV", repo_root=repo_root)
    except FileNotFoundError:
        emb_root = search_dir
        if repo_root and not os.path.isabs(search_dir):
            emb_root = os.path.normpath(os.path.join(repo_root, search_dir))
        patterns = [
            os.path.join(emb_root, "*metallurgie*.csv"),
            os.path.join(emb_root, "*mettalurgie*.csv"),
        ]
        matches: List[str] = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern))
        matches = sorted(set(matches))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]
        raise FileNotFoundError(
            f"No target embedding CSV found. Checked {primary}, {alt}, and glob patterns in {emb_root}"
        )


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_log(probs: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.log(probs.clamp_min(eps))


def cosine_probs(
    left: torch.Tensor,
    right: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    left_norm = F.normalize(left, p=2, dim=-1)
    right_norm = F.normalize(right, p=2, dim=-1)
    logits = left_norm @ right_norm.transpose(-1, -2)
    return torch.softmax(logits / tau, dim=-1)


def validate_source_labels(label2id: Dict[str, int]) -> None:
    if set(label2id.keys()) != VALID_LABELS:
        raise ValueError(f"Source labels must be {sorted(VALID_LABELS)}, got {sorted(label2id)}")


def load_source_scgm(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SCGMEmbeddingNet, Dict[str, Any], Dict[str, int], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    input_dim = int(checkpoint.get("input_dim", checkpoint_args.get("input_dim", 0)))
    if input_dim <= 0:
        raise ValueError("Checkpoint is missing input_dim.")

    label2id = checkpoint.get("label2id", LABEL2ID)
    validate_source_labels(label2id)

    proj = projection_from_checkpoint_args(checkpoint_args)
    model = SCGMEmbeddingNet(
        input_dim=input_dim,
        hiddim=int(checkpoint_args.get("hiddim", 128)),
        num_classes=int(checkpoint_args.get("n_class", 4)),
        num_subclasses=int(checkpoint_args.get("n_subclass", 32)),
        projection=proj,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    mu_y = model.mu_y.detach()
    if tuple(mu_y.shape) != (4, model.hiddim):
        raise ValueError(f"Expected mu_y shape [4, {model.hiddim}], got {tuple(mu_y.shape)}")

    return model, checkpoint_args, label2id, input_dim


def copy_projector_state(
    source_model: SCGMEmbeddingNet,
    target_model: torch.nn.Module,
) -> None:
    src_name = getattr(source_model, "projection_name", None)
    tgt_name = getattr(target_model, "projection_name", None)
    if src_name is not None and tgt_name is not None and src_name != tgt_name:
        raise ValueError(
            f"Projecteurs incompatibles : source={src_name!r}, cible={tgt_name!r}."
        )
    source_sd = source_model.projector.state_dict()
    target_sd = target_model.projector.state_dict()
    if set(source_sd.keys()) != set(target_sd.keys()):
        raise ValueError(
            "Les projecteurs source et cible ont des clés state_dict différentes "
            f"(source={list(source_sd.keys())}, cible={list(target_sd.keys())}). "
            "Vérifiez que le même type de projection est utilisé (identity / linear / mlp)."
        )
    target_model.projector.load_state_dict(source_sd)


def init_target_mu_y(
    source_model: SCGMEmbeddingNet,
    target_model: torch.nn.Module,
) -> None:
    with torch.no_grad():
        target_model.mu_y.copy_(source_model.mu_y.detach())


def build_id2label(label2id: Dict[str, int]) -> Dict[int, str]:
    return {int(value): str(key) for key, value in label2id.items()}


def set_global_seed(seed: int) -> None:
    set_seed(seed)
