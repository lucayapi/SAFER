#!/usr/bin/env python3
"""
Balayage λ_pres (= loss_weights.preserve) pour macro_transfer TPN.

Exemple :
  python scripts/run_tpn_macro_transfer_preserve_tuning.py --corpus metallurgie
  python scripts/run_tpn_macro_transfer_preserve_tuning.py --base-methods scgm_text --skip-bertopic
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import (
    default_test_corpus_id,
    macro_transfer_output_dir,
    resolve_test_paths_from_config,
)
from macro_transfer.tpn_encode import (
    resolve_tpn_checkpoint,
    tpn_method_name,
    validate_encoder_name,
)
from macro_transfer.tpn_pipeline import run_tpn_macro_transfer_discovery
from macro_transfer.tune_preserve import (
    DEFAULT_LAMBDA_PRES_GRID,
    collect_run_metrics_row,
    lambda_pres_run_dir,
    parse_base_methods_list,
    parse_lambda_pres_list,
    shared_projected_cache_dir,
    write_preserve_tuning_csv,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/tpn_macro_transfer.yaml")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument(
        "--base-methods",
        type=str,
        default="",
        help="Liste séparée par virgules (défaut : tous les encodeurs TPN)",
    )
    p.add_argument(
        "--lambda-pres-grid",
        type=str,
        default="",
        help=f"Valeurs λ_pres (défaut : {','.join(str(x) for x in DEFAULT_LAMBDA_PRES_GRID)})",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint (tous encodeurs)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip-bertopic", action="store_true")
    p.add_argument("--force-reencode", action="store_true", help="Ignore le cache projections partagé")
    p.add_argument("--dry-run", action="store_true", help="Affiche les runs sans exécuter le pipeline")
    return p.parse_args()


def _build_run_config(
    raw: Dict[str, Any],
    *,
    corpus_id: str,
    base_method: str,
    lambda_pres: float,
    projected_cache: Path,
    force_reencode: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(raw)
    cfg["corpus"] = corpus_id
    cfg["repo_anchor"] = str(TEXT_ROOT)

    method_cfg = dict(cfg.get("method") or {})
    method_cfg["base_method"] = base_method
    cfg["method"] = method_cfg

    loss_weights = dict(cfg.get("loss_weights") or {})
    loss_weights["preserve"] = float(lambda_pres)
    cfg["loss_weights"] = loss_weights

    encoding = dict(cfg.get("encoding") or {})
    encoding["projected_cache_dir"] = str(projected_cache)
    encoding["reuse_projected_embeddings"] = True
    encoding["force_reencode"] = bool(force_reencode)
    cfg["encoding"] = encoding

    if cfg.get("skip_bertopic") is None:
        cfg["skip_bertopic"] = False
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
        stream=sys.stdout,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    args = _parse_args()
    cfg_path = resolve_repo_path(args.config, repo_root=TEXT_ROOT)
    raw = load_yaml(cfg_path)

    corpus_id = args.corpus or raw.get("corpus") or default_test_corpus_id()
    spec, target_data_csv, emb_csv_resolved = resolve_test_paths_from_config(
        {**raw, "corpus": corpus_id}, corpus_id=corpus_id, anchor=TEXT_ROOT
    )
    corpus_id = spec.id

    source_cfg = raw.get("source") or {}
    source_data_csv = source_cfg.get("data_csv")
    if not source_data_csv:
        raise SystemExit("source.data_csv requis dans la config")
    source_data_csv = str(resolve_repo_path(source_data_csv, repo_root=TEXT_ROOT))
    target_data_csv = str(resolve_repo_path(target_data_csv, repo_root=TEXT_ROOT))

    lambda_grid: List[float] = (
        parse_lambda_pres_list(args.lambda_pres_grid)
        if args.lambda_pres_grid.strip()
        else list(DEFAULT_LAMBDA_PRES_GRID)
    )
    base_methods = [validate_encoder_name(m) for m in parse_base_methods_list(args.base_methods)]

    if args.skip_bertopic:
        raw = {**raw, "skip_bertopic": True}

    all_rows: List[Dict[str, Any]] = []
    log = logging.getLogger(__name__)

    for base_method in base_methods:
        method_name = tpn_method_name(base_method)
        method_root = macro_transfer_output_dir(method_name, corpus_id, anchor=TEXT_ROOT)
        tune_root = method_root / "tune_preserve"
        cache_dir = shared_projected_cache_dir(method_root)
        tune_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        method_cfg = dict(raw.get("method") or {})
        method_cfg["base_method"] = base_method
        checkpoints_block = raw.get("checkpoints") or {}
        ckpt = resolve_tpn_checkpoint(
            base_method,
            method_cfg,
            checkpoints_block,
            explicit_checkpoint=args.checkpoint,
            base_method_overridden=bool(args.checkpoint),
        )
        ckpt = str(resolve_repo_path(ckpt, repo_root=TEXT_ROOT))

        log.info(
            "=== Encodeur %s | corpus=%s | lambda_pres grid=%s | cache=%s ===",
            base_method,
            corpus_id,
            lambda_grid,
            cache_dir,
        )

        for lambda_pres in lambda_grid:
            run_dir = lambda_pres_run_dir(tune_root, lambda_pres)
            run_cfg = _build_run_config(
                raw,
                corpus_id=corpus_id,
                base_method=base_method,
                lambda_pres=lambda_pres,
                projected_cache=cache_dir,
                force_reencode=bool(args.force_reencode),
            )

            if args.device:
                enc = dict(run_cfg.get("encoding") or {})
                enc["device"] = args.device
                run_cfg["encoding"] = enc

            log.info("Run lambda_pres=%s -> %s", lambda_pres, run_dir)
            if args.dry_run:
                continue

            emb_csv = emb_csv_resolved if base_method == "scgm_text" else None
            if emb_csv:
                emb_csv = str(resolve_repo_path(emb_csv, repo_root=TEXT_ROOT))
                run_cfg.setdefault("method", {})["emb_csv"] = emb_csv

            run_tpn_macro_transfer_discovery(
                checkpoint=ckpt,
                source_data_csv=source_data_csv,
                target_data_csv=target_data_csv,
                output_dir=str(run_dir),
                config=run_cfg,
                skip_bertopic=bool(run_cfg.get("skip_bertopic", False)),
                device=args.device or run_cfg.get("encoding", {}).get("device", "cuda"),
                epochs=args.epochs,
                emb_csv=emb_csv,
            )

            row = collect_run_metrics_row(
                corpus_id=corpus_id,
                base_method=base_method,
                method_name=method_name,
                lambda_pres=lambda_pres,
                output_dir=run_dir,
                checkpoint=ckpt,
            )
            all_rows.append(row)

        per_encoder_csv = tune_root / "preserve_tuning_metrics.csv"
        encoder_rows = [r for r in all_rows if r.get("base_method") == base_method]
        if encoder_rows:
            write_preserve_tuning_csv(encoder_rows, per_encoder_csv)
            log.info("CSV encodeur : %s", per_encoder_csv)

    global_csv = TEXT_ROOT / "output_test" / corpus_id / "macro_transfer" / "preserve_tuning_metrics.csv"
    if all_rows and not args.dry_run:
        write_preserve_tuning_csv(all_rows, global_csv)
        log.info("CSV global : %s (%d lignes)", global_csv, len(all_rows))
    elif args.dry_run:
        log.info("Dry-run terminé (%d runs prévus)", len(base_methods) * len(lambda_grid))
    else:
        log.warning("Aucune métrique collectée")


if __name__ == "__main__":
    main()
