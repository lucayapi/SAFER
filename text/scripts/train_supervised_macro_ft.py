#!/usr/bin/env python3
"""Entraînement supervised_macro_ft (CE + backbone Qwen)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.paths import resolve_repo_path
from supervised_macro_ft.train_runner import run_supervised_macro_ft_training


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/methods/supervised_macro_ft.yaml")
    p.add_argument(
        "--lambda-geo",
        type=float,
        default=None,
        help="Override training.lambda_geo (ex. sweep 0.01–1.0)",
    )
    args = p.parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    overrides: dict[str, float] = {}
    if args.lambda_geo is not None:
        overrides["lambda_geo"] = float(args.lambda_geo)
    result = run_supervised_macro_ft_training(
        cfg_path,
        training_overrides=overrides or None,
    )
    print("OK:", result["output_dir"])
    print("checkpoint:", result["checkpoint_dir"])


if __name__ == "__main__":
    main()
