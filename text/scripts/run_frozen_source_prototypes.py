#!/usr/bin/env python3
"""CLI baseline Frozen Source Prototypes (sans adaptation)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from macro_transfer.frozen_source_prototypes import run_frozen_source_prototypes
from macro_transfer.fsp_config import (
    resolve_fsp_checkpoint,
    resolve_fsp_method_display_name,
    resolve_fsp_output_dir,
    validate_fsp_base_method,
)
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/frozen_source_prototypes.yaml")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--base-method", type=str, default=None, help="Surcharge method.base_method / model.base_method")
    p.add_argument("--method-display-name", type=str, default=None)
    p.add_argument(
        "--skip-bertopic",
        action="store_true",
        help="Debug uniquement ; ignoré si bertopic.enabled=true dans le YAML",
    )
    return p.parse_args()


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    method_block = dict(cfg.get("method") or {})
    model_cfg = dict(cfg.get("model") or {})
    checkpoints = dict(cfg.get("checkpoints") or {})
    base_method_overridden = False

    if args.base_method:
        base_method = validate_fsp_base_method(args.base_method)
        method_block["base_method"] = base_method
        model_cfg["base_method"] = base_method
        base_method_overridden = True
    else:
        base_method = validate_fsp_base_method(
            model_cfg.get("base_method") or method_block.get("base_method") or "scgm_text"
        )
        method_block.setdefault("base_method", base_method)
        model_cfg.setdefault("base_method", base_method)

    if args.corpus:
        cfg["corpus"] = args.corpus
    corpus = str(cfg.get("corpus") or "metallurgie")

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    elif not cfg.get("output_dir"):
        cfg["output_dir"] = str(
            resolve_fsp_output_dir(corpus, base_method, anchor=TEXT_ROOT)
        )

    ckpt = resolve_fsp_checkpoint(
        base_method,
        model_cfg,
        checkpoints,
        base_method_overridden=base_method_overridden,
    )
    if ckpt:
        model_cfg["checkpoint_path"] = ckpt
    elif base_method == "raw_embedding":
        model_cfg.pop("checkpoint_path", None)
        model_cfg.pop("checkpoint", None)

    if args.method_display_name:
        cfg["method_display_name"] = args.method_display_name
    else:
        cfg["method_display_name"] = resolve_fsp_method_display_name(
            base_method,
            cfg_display=cfg.get("method_display_name"),
            model_display=model_cfg.get("method_display_name"),
        )

    # Merge method-level encoding defaults into model for run_frozen_source_prototypes
    for key in ("backbone_name", "max_seq_length", "encode_batch_size", "device"):
        if key in method_block and key not in model_cfg:
            model_cfg[key] = method_block[key]

    cfg["method"] = method_block
    cfg["model"] = model_cfg

    bertopic_cfg = dict(cfg.get("bertopic") or {})
    if bertopic_cfg.get("enabled", True) and args.skip_bertopic:
        logging.getLogger(__name__).warning(
            "--skip-bertopic ignoré : bertopic.enabled=true dans la config"
        )
    elif args.skip_bertopic and not bertopic_cfg.get("enabled", True):
        cfg["skip_bertopic"] = True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = _parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    cfg = load_yaml(cfg_path)
    _apply_cli_overrides(cfg, args)

    tmp_cfg = TEXT_ROOT / ".tmp_frozen_source_prototypes.runtime.yaml"
    import yaml

    tmp_cfg.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        result = run_frozen_source_prototypes(tmp_cfg)
    finally:
        if tmp_cfg.exists():
            tmp_cfg.unlink()
    print("OK:", result["output_dir"])
    print("metrics:", result["metrics_path"])
    print("predictions:", result["predictions_path"])


if __name__ == "__main__":
    main()
