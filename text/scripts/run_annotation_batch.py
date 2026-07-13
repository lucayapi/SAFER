#!/usr/bin/env python3
"""CLI annotation Batch OpenAI (submit / status / download / ingest / pipeline)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

TEXT_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_ROOT))

from annotation.batch_builder import prepare_batch_input
from annotation.batch_client import (
    download_batch_results,
    has_in_progress_jobs,
    list_batch_chunks,
    refresh_batch_state,
    retrieve_batch_status,
    submit_batch,
    submit_batch_chunks,
    upload_batch_file,
)
from annotation.batch_config import BatchAnnotationConfig, load_batch_config
from annotation.batch_runner import ingest_batch_results
from annotation.cache import get_batch_paths


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--config",
        type=str,
        default="configs/annotation_batch.yaml",
        help="Chemin YAML (relatif à text/)",
    )
    parent.add_argument("--run-id", type=str, default=None, help="Reprendre une run existante")
    parent.add_argument("--input-csv", type=str, default=None)
    parent.add_argument("--n-accidents", type=str, default=None)
    parent.add_argument("--units-per-accident", type=str, default=None)
    parent.add_argument("--openai-model", type=str, default=None)
    parent.add_argument("--reasoning-effort", type=str, default=None)
    parent.add_argument("--pass-mode", type=str, default=None, choices=["pass1", "pass2"])
    parent.add_argument("--batch-id", type=str, default=None, help="ID batch OpenAI (status/download)")
    parent.add_argument("--force-resubmit", action="store_true", help="Soumettre un nouveau batch")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/annotation_batch.yaml",
        help="Chemin YAML (relatif à text/)",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Reprendre une run existante")
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--n-accidents", type=str, default=None)
    parser.add_argument("--units-per-accident", type=str, default=None)
    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--pass-mode", type=str, default=None, choices=["pass1", "pass2"])
    parser.add_argument("--batch-id", type=str, default=None, help="ID batch OpenAI (status/download)")
    parser.add_argument("--force-resubmit", action="store_true", help="Soumettre un nouveau batch")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("submit", parents=[parent], help="Prépare JSONL, upload et soumet le batch")
    sub.add_parser("status", parents=[parent], help="Rafraîchit le statut OpenAI")
    sub.add_parser("download", parents=[parent], help="Télécharge output/errors si completed")
    sub.add_parser("ingest", parents=[parent], help="Parse résultats, cache JSONL et export XLSX")

    pipeline = sub.add_parser(
        "pipeline",
        parents=[parent],
        help="submit -> poll status -> download -> ingest",
    )
    pipeline.add_argument(
        "--poll-interval-sec",
        type=float,
        default=None,
        help="Intervalle entre deux status (défaut: config YAML)",
    )
    pipeline.add_argument(
        "--max-wait-sec",
        type=float,
        default=None,
        help="Timeout total (défaut: illimité)",
    )

    return parser.parse_args(argv)


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.run_id:
        overrides["run_id"] = args.run_id
    if args.input_csv:
        overrides["input_csv"] = args.input_csv
    if args.n_accidents is not None:
        overrides["n_accidents"] = args.n_accidents
    if args.units_per_accident is not None:
        overrides["units_per_accident"] = args.units_per_accident
    if args.openai_model:
        overrides["openai_model"] = args.openai_model
    if args.reasoning_effort:
        overrides["reasoning_effort"] = args.reasoning_effort
    if args.pass_mode:
        overrides["pass_mode"] = args.pass_mode
    return overrides


def _load_cfg(args: argparse.Namespace) -> BatchAnnotationConfig:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = TEXT_ROOT / config_path
    return load_batch_config(config_path, overrides=_config_overrides(args))


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_submit(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    prep = prepare_batch_input(cfg)
    if prep["n_batch_requests"] == 0:
        _print_json(
            {
                "run_id": cfg.run_id,
                "message": "Aucune requête à soumettre (tout est déjà en cache).",
                **{k: prep[k] for k in prep if k not in {"dataframe", "todo_dataframe"}},
            }
        )
        return 0

    chunks = prep.get("batch_input_chunks") or []
    if not chunks:
        batch_input_path = Path(prep.get("batch_input_path") or "")
        if not batch_input_path.is_file():
            raise FileNotFoundError("Aucun fichier batch_input généré.")
        input_file_id = upload_batch_file(batch_input_path)
        state = submit_batch(
            cfg,
            input_file_id=input_file_id,
            force_resubmit=args.force_resubmit,
        )
        n_chunks = 1
    else:
        state = submit_batch_chunks(
            cfg,
            chunks,
            force_resubmit=args.force_resubmit,
        )
        n_chunks = len(chunks)

    _print_json(
        {
            "run_id": cfg.run_id,
            "outputs_dir": str(cfg.outputs_dir),
            "batch_input_path": prep.get("batch_input_path"),
            "batch_input_chunks": chunks,
            "n_chunks": n_chunks,
            "n_batch_requests": prep["n_batch_requests"],
            "n_cache_hits": prep["n_cache_hits"],
            **state,
        }
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    state = retrieve_batch_status(cfg, batch_id=args.batch_id)
    _print_json({"run_id": cfg.run_id, "outputs_dir": str(cfg.outputs_dir), **state})
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    result = download_batch_results(cfg, batch_id=args.batch_id)
    _print_json({"run_id": cfg.run_id, "outputs_dir": str(cfg.outputs_dir), **result})
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    _, meta = ingest_batch_results(cfg)
    _print_json({"run_id": cfg.run_id, **meta})
    return 0


def _submit_prepared_batch(
    cfg: BatchAnnotationConfig,
    prep: dict[str, Any],
    *,
    force_resubmit: bool,
) -> dict[str, Any]:
    if prep["n_batch_requests"] == 0:
        return {"action": "done", "message": "Aucune requête à soumettre (cache à jour)."}

    chunks = prep.get("batch_input_chunks") or []
    if chunks:
        return submit_batch_chunks(cfg, chunks, force_resubmit=force_resubmit)

    batch_input_path = Path(prep.get("batch_input_path") or "")
    if not batch_input_path.is_file():
        raise FileNotFoundError("Aucun fichier batch_input généré.")
    input_file_id = upload_batch_file(batch_input_path)
    return submit_batch(cfg, input_file_id=input_file_id, force_resubmit=force_resubmit)


def _print_batch_progress(cfg: BatchAnnotationConfig, state: dict[str, Any]) -> None:
    status = str(state.get("status") or "")
    counts = state.get("request_counts") or {}
    chunk_summaries = [
        {
            "batch_id": chunk.get("batch_id"),
            "status": chunk.get("status"),
            "completed": (chunk.get("request_counts") or {}).get("completed"),
            "total": (chunk.get("request_counts") or {}).get("total"),
        }
        for chunk in list_batch_chunks(state)
    ]
    print(
        f"[batch] run_id={cfg.run_id} status={status} "
        f"completed={counts.get('completed', 0)}/{counts.get('total', 0)} "
        f"failed={counts.get('failed', 0)} chunks={len(chunk_summaries)}"
    )


def _poll_until_idle(
    cfg: BatchAnnotationConfig,
    *,
    interval: float,
    deadline: Optional[float],
) -> dict[str, Any]:
    terminal_failure = {"failed", "expired", "cancelled"}
    while True:
        state = refresh_batch_state(cfg)
        _print_batch_progress(cfg, state)
        chunks = list_batch_chunks(state)
        if not has_in_progress_jobs(chunks):
            failed_only = chunks and all(
                str(chunk.get("status") or "") in terminal_failure for chunk in chunks
            )
            if failed_only:
                return state
            completed = [
                chunk for chunk in chunks if str(chunk.get("status") or "") == "completed"
            ]
            if completed or not chunks:
                return state
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Timeout pipeline atteint.")
        time.sleep(interval)


def _download_completed_chunks(cfg: BatchAnnotationConfig) -> dict[str, Any]:
    try:
        return download_batch_results(cfg, partial=True)
    except RuntimeError as exc:
        if "Aucun chunk terminé" not in str(exc):
            raise
        return {"n_chunks_downloaded": 0, "partial": True}


def cmd_pipeline(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    interval = float(args.poll_interval_sec or cfg.status_poll_interval_sec)
    deadline = time.time() + float(args.max_wait_sec) if args.max_wait_sec else None
    terminal_failure = {"failed", "expired", "cancelled"}
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"[pipeline] cycle={cycle} prepare…")
            prep = prepare_batch_input(cfg)
            if prep["n_batch_requests"] == 0:
                print("[pipeline] Toutes les unités sont déjà annotées (cache).")
                if prep.get("n_batch_exhausted", 0) > 0:
                    print(
                        f"[pipeline] {prep['n_batch_exhausted']} lignes en échec batch "
                        f"(max_batch_retries={cfg.max_batch_retries}) — non resoumises."
                    )
                try:
                    _, meta = ingest_batch_results(cfg)
                except FileNotFoundError:
                    meta = {"message": "Export final sans nouveau batch output."}
                _print_json({"run_id": cfg.run_id, "pipeline": "done", "cycles": cycle, **meta})
                return 0

            print(
                f"[pipeline] requêtes à soumettre={prep['n_batch_requests']} "
                f"cache_hits={prep['n_cache_hits']} "
                f"batch_exhausted={prep.get('n_batch_exhausted', 0)}"
            )
            state = _submit_prepared_batch(cfg, prep, force_resubmit=args.force_resubmit)
            action = state.get("action")

            if action == "wait":
                print(state.get("message") or "Batch en cours — attente…")
                state = _poll_until_idle(cfg, interval=interval, deadline=deadline)
            elif action == "download_first":
                print(state.get("message") or "Download requis avant nouveau submit.")
                _download_completed_chunks(cfg)
                _, meta = ingest_batch_results(cfg)
                _print_json({"run_id": cfg.run_id, "pipeline": "ingested_pending", **meta})
                continue
            elif action == "ingest_first":
                print(state.get("message") or "Ingest requis avant nouveau submit.")
                _, meta = ingest_batch_results(cfg)
                _print_json({"run_id": cfg.run_id, "pipeline": "ingested_pending", **meta})
                continue
            elif action in {"noop", "done"}:
                _print_json({"run_id": cfg.run_id, "pipeline": action, **state})
                return 0 if action == "done" else 1

            state = _poll_until_idle(cfg, interval=interval, deadline=deadline)
            chunks = list_batch_chunks(state)
            if chunks and all(str(chunk.get("status") or "") in terminal_failure for chunk in chunks):
                _print_json({"run_id": cfg.run_id, "pipeline": "failed", **state})
                return 1

            dl = _download_completed_chunks(cfg)
            if dl.get("n_chunks_downloaded", 0) == 0:
                _print_json(
                    {
                        "run_id": cfg.run_id,
                        "pipeline": "no_download",
                        "message": "Aucun chunk terminé à télécharger.",
                        **state,
                    }
                )
                return 1

            _, meta = ingest_batch_results(cfg)
            prep_after = prepare_batch_input(cfg)
            if (
                meta.get("batch_new", 0) == 0
                and prep_after["n_batch_requests"] > 0
                and prep_after["n_batch_requests"] == prep["n_batch_requests"]
            ):
                _print_json(
                    {
                        "run_id": cfg.run_id,
                        "pipeline": "stalled",
                        "message": (
                            "Aucune progression après ingest — arrêt pour éviter une boucle "
                            "infinie. Vérifiez les lignes en échec dans le cache."
                        ),
                        "cycle": cycle,
                        "download": dl,
                        **meta,
                    }
                )
                return 1
            _print_json(
                {
                    "run_id": cfg.run_id,
                    "pipeline": "cycle_done",
                    "cycle": cycle,
                    "download": dl,
                    **meta,
                }
            )
    except TimeoutError as exc:
        print(str(exc))
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    handlers = {
        "submit": cmd_submit,
        "status": cmd_status,
        "download": cmd_download,
        "ingest": cmd_ingest,
        "pipeline": cmd_pipeline,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
