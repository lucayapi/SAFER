#!/usr/bin/env python3
"""Transfert test + BERTopic pour supervised_macro_ft."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from supervised_macro_ft.transfer import run_supervised_macro_ft_transfer


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
    p.add_argument("--config", type=str, default="configs/supervised_macro_ft_transfer.yaml")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument(
        "--skip-bertopic",
        action="store_true",
        help="Transfert macro uniquement (run_bertopic=false)",
    )
    p.add_argument(
        "--judge-enable",
        type=_parse_bool_flag,
        nargs="?",
        const=True,
        default=None,
        help="Override judge_enable (true/false) — LLM juge qualité topics",
    )
    args = p.parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    cfg = load_yaml(cfg_path)
    if args.corpus:
        cfg["corpus"] = args.corpus
    if args.skip_bertopic:
        cfg["run_bertopic"] = False
    if args.judge_enable is not None:
        cfg["judge_enable"] = bool(args.judge_enable)
    tmp = TEXT_ROOT / ".tmp_supervised_macro_ft_transfer.yaml"
    import yaml

    tmp.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        result = run_supervised_macro_ft_transfer(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
    print("OK:", result["output_dir"])
    print("metrics:", result["metrics_path"])


if __name__ == "__main__":
    main()
