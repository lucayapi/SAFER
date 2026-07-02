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


def _parse_bool_flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    p.add_argument(
        "--standardize-backbone",
        type=_parse_bool_flag,
        nargs="?",
        const=True,
        default=None,
        help="Override model.standardize_backbone (true/false)",
    )
    args = p.parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    training_overrides: dict[str, float] = {}
    model_overrides: dict[str, bool] = {}
    if args.lambda_geo is not None:
        training_overrides["lambda_geo"] = float(args.lambda_geo)
    if args.standardize_backbone is not None:
        model_overrides["standardize_backbone"] = bool(args.standardize_backbone)
    result = run_supervised_macro_ft_training(
        cfg_path,
        training_overrides=training_overrides or None,
        model_overrides=model_overrides or None,
    )
    print("OK:", result["output_dir"])
    print("checkpoint:", result["checkpoint_dir"])


if __name__ == "__main__":
    main()
