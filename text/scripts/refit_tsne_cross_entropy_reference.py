#!/usr/bin/env python3
"""Fit the fixed cross-entropy reference model used for the t-SNE figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replication.runner import set_global_training_seed
from safer_core.io import load_yaml
from supervised_macro_ft.train_runner import run_supervised_macro_ft_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/article_visualization/cross_entropy_qwen3_construction_tsne.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = load_yaml(config_path)
    seed = int(dict(cfg.get("training") or {}).get("seed", 42))
    set_global_training_seed(seed)
    result = run_supervised_macro_ft_training(config_path, cfg=cfg)
    print("OK:", result["output_dir"])
    print("embeddings:", Path(result["output_dir"]) / "embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
