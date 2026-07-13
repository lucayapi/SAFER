#!/usr/bin/env python3
"""Entraînement supervised_macro_ft (CE + backbone Qwen)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.paths import resolve_repo_path
from supervised_macro_ft.train_runner import run_supervised_macro_ft_training


def _parse_bool_flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_int(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in ("null", "none", ""):
        return None
    return int(value)


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
        "--standardize-backbone",
        type=_parse_bool_flag,
        nargs="?",
        const=True,
        default=None,
        help="Override model.standardize_backbone (true/false)",
    )
    p.add_argument(
        "--oversampling",
        type=_parse_bool_flag,
        nargs="?",
        const=True,
        default=None,
        help="Override model.oversampling (true/false)",
    )
    p.add_argument(
        "--class-weight",
        type=str,
        default=None,
        help="Override model.class_weight (null|balanced)",
    )
    p.add_argument(
        "--backbone-trainable",
        type=_parse_bool_flag,
        nargs="?",
        const=True,
        default=None,
        help="Override model.backbone_trainable (true/false)",
    )
    p.add_argument(
        "--train-last-n-layers",
        type=_parse_optional_int,
        default=None,
        help="Override model.train_last_n_layers (int ou null)",
    )
    p.add_argument(
        "--test-corpora",
        type=str,
        default=None,
        help="Override test_corpora (comma-separated, ex. metallurgie,caou)",
    )
    args = p.parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    training_overrides: dict[str, float] = {}
    model_overrides: dict[str, object] = {}
    test_corpora_override: list[str] | None = None
    test_corpora_raw = args.test_corpora or os.environ.get("TEST_CORPORA")
    if test_corpora_raw:
        test_corpora_override = [c.strip() for c in str(test_corpora_raw).split(",") if c.strip()]
    if args.standardize_backbone is not None:
        model_overrides["standardize_backbone"] = bool(args.standardize_backbone)
    if args.oversampling is not None:
        model_overrides["oversampling"] = bool(args.oversampling)
    if args.class_weight is not None:
        cw = str(args.class_weight).strip().lower()
        model_overrides["class_weight"] = None if cw in ("null", "none", "") else cw
    if args.backbone_trainable is not None:
        model_overrides["backbone_trainable"] = bool(args.backbone_trainable)
    if args.train_last_n_layers is not None:
        model_overrides["train_last_n_layers"] = args.train_last_n_layers
    result = run_supervised_macro_ft_training(
        cfg_path,
        training_overrides=training_overrides or None,
        model_overrides=model_overrides or None,
        test_corpora_override=test_corpora_override,
    )
    print("OK:", result["output_dir"])
    print("checkpoint:", result["checkpoint_dir"])


if __name__ == "__main__":
    main()
