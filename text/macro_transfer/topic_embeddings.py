"""Construction de l'espace d'embeddings pour BERTopic intra-macro (TPN)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from macro_transfer.tpn_prototypes import l2_normalize_np

VALID_MODES = frozenset({"initial", "adapted", "mixed"})


def build_topic_embeddings(
    h_initial: np.ndarray,
    h_adapted: np.ndarray | None = None,
    *,
    mode: str = "initial",
    alpha: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Construit les embeddings utilisés pour BERTopic.

    mode:
      - "initial" : h_topic = h_initial
      - "adapted" : h_topic = h_adapted
      - "mixed"   : h_topic = (1-alpha)*h_initial + alpha*h_adapted

    Si normalize=True, normalisation L2 par ligne.
    """
    mode_l = str(mode).strip().lower()
    if mode_l not in VALID_MODES:
        raise ValueError(f"mode invalide {mode!r} ; attendu initial|adapted|mixed")

    h_initial = np.asarray(h_initial, dtype=np.float64)
    if h_initial.ndim != 2:
        raise ValueError(f"h_initial doit être 2D, reçu shape={h_initial.shape}")

    if mode_l == "initial":
        out = h_initial.copy()
    elif mode_l == "adapted":
        if h_adapted is None:
            raise ValueError("h_adapted requis pour mode='adapted'")
        out = np.asarray(h_adapted, dtype=np.float64)
        if out.shape != h_initial.shape:
            raise ValueError(
                f"h_adapted shape {out.shape} incompatible avec h_initial {h_initial.shape}"
            )
    else:
        if h_adapted is None:
            raise ValueError("h_adapted requis pour mode='mixed'")
        h_ad = np.asarray(h_adapted, dtype=np.float64)
        if h_ad.shape != h_initial.shape:
            raise ValueError(
                f"h_adapted shape {h_ad.shape} incompatible avec h_initial {h_initial.shape}"
            )
        a = float(alpha)
        out = (1.0 - a) * h_initial + a * h_ad

    if normalize:
        out = l2_normalize_np(out)
    return out.astype(np.float64, copy=False)


def resolve_topic_embedding_cfg(
    bertopic_cfg: Dict[str, Any],
    *,
    cli_mode: Optional[str] = None,
    cli_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Fusionne bertopic.embedding_space (YAML) et overrides CLI."""
    emb = dict(bertopic_cfg.get("embedding_space") or {})
    mode = cli_mode if cli_mode is not None else emb.get("mode", "initial")
    alpha = cli_alpha if cli_alpha is not None else emb.get("alpha", 0.0)
    normalize = bool(emb.get("normalize", True))
    mode_l = str(mode).strip().lower()
    if mode_l not in VALID_MODES:
        raise ValueError(f"topic embedding mode invalide : {mode!r}")
    return {
        "mode": mode_l,
        "alpha": float(alpha),
        "normalize": normalize,
    }
