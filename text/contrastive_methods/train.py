"""Point d'entrée unique pour les méthodes contrastives."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from contrastive_methods.config import load_contrastive_config
from contrastive_methods.training_softtriple import run_softtriple
from contrastive_methods.training_supcon import run_supcon
from contrastive_methods.training_triplet import run_batch_triplet
from safer_core.text_columns import warn_if_prompt_enabled


def run_contrastive_method(method_name: str, argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Entraînement contrastif : {method_name}")
    parser.add_argument(
        "--config",
        type=str,
        default=f"configs/methods/{method_name}.yaml",
    )
    args, _ = parser.parse_known_args(argv)

    cfg = load_contrastive_config(method_name, args.config)
    warn_if_prompt_enabled(cfg.use_prompt or cfg.use_contextual_prompt_with_summary)

    dispatch = {
        "batch_triplet": run_batch_triplet,
        "supcon": run_supcon,
        "softtriple": run_softtriple,
    }
    if method_name not in dispatch:
        raise ValueError(f"Méthode inconnue : {method_name}")

    emb_path = dispatch[method_name](cfg)
    print(f"[{method_name}] Terminé — embeddings : {emb_path}")
    return 0


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m contrastive_methods.train <batch_triplet|supcon|softtriple> [--config ...]")
    method = sys.argv[1]
    raise SystemExit(run_contrastive_method(method, sys.argv[2:]))


if __name__ == "__main__":
    main()
