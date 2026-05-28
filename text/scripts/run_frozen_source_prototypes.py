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
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/frozen_source_prototypes.yaml")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--method-display-name", type=str, default=None)
    return p.parse_args()


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
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.corpus:
        cfg["corpus"] = args.corpus
    if args.method_display_name:
        cfg["method_display_name"] = args.method_display_name

    tmp_cfg = TEXT_ROOT / ".tmp_frozen_source_prototypes.runtime.yaml"
    tmp_cfg.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
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

