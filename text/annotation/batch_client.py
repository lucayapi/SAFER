"""Client OpenAI Batch API + persistance batch_state.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv

from annotation.batch_config import BatchAnnotationConfig
from annotation.cache import get_batch_paths

IN_PROGRESS_STATUSES = frozenset({"validating", "in_progress", "finalizing"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "expired", "cancelled"})
TERMINAL_STATUSES = frozenset({"completed", *TERMINAL_FAILURE_STATUSES})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_batch_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def save_batch_state(path: Path, state: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    return path


def batch_state_path(cfg: BatchAnnotationConfig) -> Path:
    _, _, _, state_path = get_batch_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    return state_path


def _request_counts_to_dict(counts: Any) -> Dict[str, int]:
    if counts is None:
        return {}
    return {
        "total": int(getattr(counts, "total", 0) or 0),
        "completed": int(getattr(counts, "completed", 0) or 0),
        "failed": int(getattr(counts, "failed", 0) or 0),
    }


def batch_object_to_state(batch: Any, *, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prev = dict(previous or {})
    status = str(getattr(batch, "status", "") or "")
    state: Dict[str, Any] = {
        "batch_id": getattr(batch, "id", prev.get("batch_id")),
        "input_file_id": getattr(batch, "input_file_id", prev.get("input_file_id")),
        "output_file_id": getattr(batch, "output_file_id", prev.get("output_file_id")),
        "error_file_id": getattr(batch, "error_file_id", prev.get("error_file_id")),
        "status": status,
        "request_counts": _request_counts_to_dict(getattr(batch, "request_counts", None)),
        "submitted_at": prev.get("submitted_at"),
        "completed_at": prev.get("completed_at"),
        "downloaded_at": prev.get("downloaded_at"),
        "ingested_at": prev.get("ingested_at"),
        "updated_at": _utc_now_iso(),
    }
    for key in ("chunk_index", "batch_input_path", "n_requests", "file_size_bytes"):
        if key in prev:
            state[key] = prev[key]
    if status == "completed" and not state.get("completed_at"):
        state["completed_at"] = _utc_now_iso()
    return state


def list_batch_chunks(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise l'état mono- ou multi-batch en liste de jobs."""
    chunks = state.get("chunks")
    if isinstance(chunks, list) and chunks:
        return [dict(chunk) for chunk in chunks]
    if state.get("batch_id"):
        return [dict(state)]
    return []


def has_in_progress_jobs(chunks: List[Dict[str, Any]]) -> bool:
    return any(str(chunk.get("status") or "") in IN_PROGRESS_STATUSES for chunk in chunks)


def aggregate_batch_status(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synthétise le statut global à partir des jobs batch."""
    if not chunks:
        return {"status": None, "request_counts": {"total": 0, "completed": 0, "failed": 0}}

    statuses = [str(chunk.get("status") or "") for chunk in chunks]
    if any(status in IN_PROGRESS_STATUSES for status in statuses):
        aggregate_status = "in_progress"
    elif statuses and all(status == "completed" for status in statuses):
        aggregate_status = "completed"
    elif any(status == "completed" for status in statuses):
        aggregate_status = "partial"
    elif any(status in TERMINAL_FAILURE_STATUSES for status in statuses):
        aggregate_status = "failed"
    else:
        aggregate_status = statuses[-1] or "unknown"

    counts = {"total": 0, "completed": 0, "failed": 0}
    for chunk in chunks:
        chunk_counts = chunk.get("request_counts") or {}
        counts["total"] += int(chunk_counts.get("total") or 0)
        counts["completed"] += int(chunk_counts.get("completed") or 0)
        counts["failed"] += int(chunk_counts.get("failed") or 0)

    return {"status": aggregate_status, "request_counts": counts}


def has_pending_download_jobs(chunks: List[Dict[str, Any]]) -> bool:
    return any(
        str(chunk.get("status") or "") == "completed" and not chunk.get("downloaded_at")
        for chunk in chunks
    )


def has_pending_ingest_jobs(chunks: List[Dict[str, Any]]) -> bool:
    return any(
        str(chunk.get("status") or "") == "completed"
        and chunk.get("downloaded_at")
        and not chunk.get("ingested_at")
        for chunk in chunks
    )


def mark_chunks_ingested(
    cfg: BatchAnnotationConfig,
    *,
    batch_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Marque les chunks téléchargés comme ingérés dans batch_state.json."""
    state_path = batch_state_path(cfg)
    chunks = list_batch_chunks(load_batch_state(state_path))
    if not chunks:
        return {}

    allowed = {str(batch_id) for batch_id in (batch_ids or []) if batch_id}
    now = _utc_now_iso()
    updated = 0
    for chunk in chunks:
        bid = str(chunk.get("batch_id") or "")
        if allowed and bid not in allowed:
            continue
        if str(chunk.get("status") or "") != "completed":
            continue
        if not chunk.get("downloaded_at"):
            continue
        if chunk.get("ingested_at"):
            continue
        chunk["ingested_at"] = now
        updated += 1

    if updated:
        return _persist_chunk_state(state_path, chunks)
    return load_batch_state(state_path)


def _persist_chunk_state(state_path: Path, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = aggregate_batch_status(chunks)
    state: Dict[str, Any] = {
        "version": 2,
        "chunks": chunks,
        "status": aggregate["status"],
        "request_counts": aggregate["request_counts"],
        "updated_at": _utc_now_iso(),
    }
    if len(chunks) == 1:
        state.update(chunks[0])
    save_batch_state(state_path, state)
    return state


def upload_batch_file(path: Path, *, client: Any = None) -> str:
    load_openai_dotenv()
    if client is None:
        client = _get_client()
    with open(path, "rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    return str(uploaded.id)


def _create_batch_job(
    cfg: BatchAnnotationConfig,
    *,
    input_file_id: str,
    client: Any,
) -> Dict[str, Any]:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=cfg.batch_endpoint,
        completion_window=cfg.completion_window,
    )
    state = batch_object_to_state(batch)
    state["submitted_at"] = _utc_now_iso()
    return state


def refresh_batch_state(
    cfg: BatchAnnotationConfig,
    *,
    client: Any = None,
) -> Dict[str, Any]:
    """Rafraîchit tous les jobs OpenAI connus pour cette run."""
    load_openai_dotenv()
    if client is None:
        client = _get_client()

    state_path = batch_state_path(cfg)
    previous = load_batch_state(state_path)
    chunks = list_batch_chunks(previous)

    refreshed: List[Dict[str, Any]] = []
    for chunk in chunks:
        bid = chunk.get("batch_id")
        if not bid:
            refreshed.append(dict(chunk))
            continue
        batch = client.batches.retrieve(bid)
        refreshed.append(batch_object_to_state(batch, previous=chunk))

    if not refreshed:
        return previous
    return _persist_chunk_state(state_path, refreshed)


def submit_batch(
    cfg: BatchAnnotationConfig,
    *,
    input_file_id: str,
    client: Any = None,
    force_resubmit: bool = False,
) -> Dict[str, Any]:
    load_openai_dotenv()
    if client is None:
        client = _get_client()

    state_path = batch_state_path(cfg)
    previous = refresh_batch_state(cfg, client=client)
    existing = list_batch_chunks(previous)
    if has_in_progress_jobs(existing) and not force_resubmit:
        return {
            **previous,
            "action": "wait",
            "message": "Un batch est encore in_progress — attendez sa fin avant un nouveau submit.",
        }
    if existing and not force_resubmit:
        return {
            **previous,
            "action": "noop",
            "message": "Des jobs existent déjà. Relancez après download/ingest ou utilisez --force-resubmit.",
        }

    job_state = _create_batch_job(cfg, input_file_id=input_file_id, client=client)
    return _persist_chunk_state(state_path, [job_state])


def submit_batch_chunks(
    cfg: BatchAnnotationConfig,
    chunks: List[Dict[str, Any]],
    *,
    client: Any = None,
    force_resubmit: bool = False,
) -> Dict[str, Any]:
    """Soumet au plus le premier chunk JSONL (limite tokens en file OpenAI)."""
    load_openai_dotenv()
    if client is None:
        client = _get_client()

    if not chunks:
        raise ValueError("Aucun chunk batch à soumettre.")

    state_path = batch_state_path(cfg)
    existing = list_batch_chunks(refresh_batch_state(cfg, client=client))

    if has_in_progress_jobs(existing) and not force_resubmit:
        aggregate = aggregate_batch_status(existing)
        return {
            "version": 2,
            "chunks": existing,
            "status": aggregate["status"],
            "request_counts": aggregate["request_counts"],
            "action": "wait",
            "message": (
                "Un batch est encore in_progress. Attendez la fin, puis "
                "download + ingest avant de relancer submit."
            ),
            "updated_at": _utc_now_iso(),
        }

    if has_pending_download_jobs(existing) and not force_resubmit:
        aggregate = aggregate_batch_status(existing)
        return {
            "version": 2,
            "chunks": existing,
            "status": aggregate["status"],
            "request_counts": aggregate["request_counts"],
            "action": "download_first",
            "message": (
                "Des chunks terminés ne sont pas encore téléchargés. "
                "Lancez download puis ingest avant un nouveau submit."
            ),
            "updated_at": _utc_now_iso(),
        }

    if has_pending_ingest_jobs(existing) and not force_resubmit:
        aggregate = aggregate_batch_status(existing)
        return {
            "version": 2,
            "chunks": existing,
            "status": aggregate["status"],
            "request_counts": aggregate["request_counts"],
            "action": "ingest_first",
            "message": (
                "Des chunks téléchargés ne sont pas encore ingérés. "
                "Lancez ingest avant un nouveau submit."
            ),
            "updated_at": _utc_now_iso(),
        }

    first = chunks[0]
    input_path = Path(str(first["batch_input_path"]))
    input_file_id = upload_batch_file(input_path, client=client)
    job_state = _create_batch_job(cfg, input_file_id=input_file_id, client=client)
    new_chunk = {
        "chunk_index": int(first.get("chunk_index", len(existing))),
        "batch_input_path": str(input_path),
        "n_requests": int(first.get("n_requests") or 0),
        "file_size_bytes": int(first.get("file_size_bytes") or 0),
        **job_state,
    }
    updated_chunks = [*existing, new_chunk]
    state = _persist_chunk_state(state_path, updated_chunks)
    state["action"] = "submitted"
    state["submitted_chunk"] = new_chunk
    state["pending_input_chunks"] = max(0, len(chunks) - 1)
    return state


def retrieve_batch_status(
    cfg: BatchAnnotationConfig,
    *,
    batch_id: Optional[str] = None,
    client: Any = None,
) -> Dict[str, Any]:
    load_openai_dotenv()
    if client is None:
        client = _get_client()

    state_path = batch_state_path(cfg)
    previous = load_batch_state(state_path)
    chunks = list_batch_chunks(previous)
    if batch_id:
        chunks = [chunk for chunk in chunks if chunk.get("batch_id") == batch_id]
        if not chunks:
            raise ValueError(f"batch_id introuvable dans l'état local : {batch_id!r}")
    elif not chunks:
        raise ValueError("batch_id introuvable : soumettez d'abord un batch ou passez --batch-id.")

    refreshed: List[Dict[str, Any]] = []
    for chunk in chunks:
        bid = chunk.get("batch_id")
        if not bid:
            refreshed.append(dict(chunk))
            continue
        batch = client.batches.retrieve(bid)
        refreshed.append(batch_object_to_state(batch, previous=chunk))

    if batch_id:
        aggregate = aggregate_batch_status(refreshed)
        state = {**refreshed[0], "status": aggregate["status"], "request_counts": aggregate["request_counts"]}
        if len(list_batch_chunks(previous)) == 1:
            save_batch_state(state_path, state)
        else:
            merged = []
            for chunk in list_batch_chunks(previous):
                if chunk.get("batch_id") == batch_id:
                    merged.append(refreshed[0])
                else:
                    merged.append(chunk)
            state = _persist_chunk_state(state_path, merged)
        return state

    return _persist_chunk_state(state_path, refreshed)


def download_batch_file(
    file_id: str,
    dest_path: Path,
    *,
    client: Any = None,
) -> Path:
    load_openai_dotenv()
    if client is None:
        client = _get_client()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    file_response = client.files.content(file_id)
    dest_path.write_text(file_response.text, encoding="utf-8")
    return dest_path


def _append_jsonl_files(source_paths: List[Path], dest_path: Path) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(dest_path, "w", encoding="utf-8") as dest_handle:
        for source_path in source_paths:
            if not source_path.is_file():
                continue
            with open(source_path, "r", encoding="utf-8") as src_handle:
                for line in src_handle:
                    if line.strip():
                        dest_handle.write(line if line.endswith("\n") else line + "\n")
                        n_lines += 1
    return n_lines


def _batch_output_part_path(output_path: Path, chunk: Dict[str, Any], *, suffix: str) -> Path:
    part_index = int(chunk.get("chunk_index", 0))
    batch_id = str(chunk.get("batch_id") or "unknown")
    slug = batch_id.replace("batch_", "")[:12]
    return output_path.with_name(f"{output_path.stem}__{slug}__part{part_index + 1:03d}{suffix}")


def collect_batch_output_paths(cfg: BatchAnnotationConfig) -> List[Path]:
    """Tous les fichiers de résultats batch disponibles localement."""
    _, output_path, _, _ = get_batch_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )
    paths = sorted(cfg.outputs_dir.glob(f"{output_path.stem}__*.jsonl"))
    if output_path.is_file():
        paths.append(output_path)
    seen: set[str] = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def download_batch_results(
    cfg: BatchAnnotationConfig,
    *,
    batch_id: Optional[str] = None,
    partial: bool = True,
    client: Any = None,
) -> Dict[str, Any]:
    state = retrieve_batch_status(cfg, batch_id=batch_id, client=client)
    chunks = list_batch_chunks(state)
    if not chunks:
        raise RuntimeError("Aucun batch enregistré pour cette run.")

    if batch_id is None and not partial:
        not_done = [chunk for chunk in chunks if chunk.get("status") != "completed"]
        if not_done:
            pending = [chunk.get("batch_id") for chunk in not_done]
            raise RuntimeError(
                f"Tous les chunks ne sont pas terminés. En attente : {pending}. "
                "Utilisez partial=True (défaut) pour télécharger les chunks complétés."
            )

    _, output_path, error_path, _ = get_batch_paths(
        cfg.outputs_dir,
        model_id=cfg.openai_model,
        prompt_version=cfg.prompt_version,
        artifact_slug=cfg.artifact_slug,
    )

    output_parts: List[Path] = []
    error_parts: List[Path] = []
    downloaded_jobs: List[str] = []
    state_path = batch_state_path(cfg)
    all_chunks = list_batch_chunks(load_batch_state(state_path))

    for chunk in chunks:
        status = str(chunk.get("status") or "")
        if status != "completed":
            continue
        if not chunk.get("output_file_id"):
            raise RuntimeError(
                f"Batch {chunk.get('batch_id')!r} terminé mais output_file_id absent."
            )
        out_part = _batch_output_part_path(output_path, chunk, suffix=".jsonl")
        download_batch_file(str(chunk["output_file_id"]), out_part, client=client)
        output_parts.append(out_part)
        downloaded_jobs.append(str(chunk.get("batch_id")))
        chunk["downloaded_at"] = _utc_now_iso()
        if chunk.get("error_file_id"):
            err_part = _batch_output_part_path(error_path, chunk, suffix=".jsonl")
            download_batch_file(str(chunk["error_file_id"]), err_part, client=client)
            error_parts.append(err_part)

    if not output_parts:
        raise RuntimeError("Aucun chunk terminé à télécharger.")

    merged_sources = sorted(
        {
            path
            for path in cfg.outputs_dir.glob(f"{output_path.stem}__*.jsonl")
            if path.is_file()
        },
        key=lambda p: p.name,
    )
    n_output_lines = _append_jsonl_files(merged_sources, output_path)

    if all_chunks:
        by_id = {str(chunk.get("batch_id")): chunk for chunk in all_chunks if chunk.get("batch_id")}
        for chunk in chunks:
            bid = str(chunk.get("batch_id") or "")
            if bid in by_id and chunk.get("downloaded_at"):
                by_id[bid]["downloaded_at"] = chunk["downloaded_at"]
        _persist_chunk_state(state_path, list(by_id.values()))

    result = {
        "batch_output_path": str(output_path),
        "batch_errors_path": None,
        "status": state.get("status"),
        "n_output_lines": n_output_lines,
        "n_chunks_downloaded": len(output_parts),
        "downloaded_batch_ids": downloaded_jobs,
        "chunk_output_paths": [str(path) for path in output_parts],
        "partial": partial,
    }
    if error_parts:
        merged_errors = sorted(
            {
                path
                for path in cfg.outputs_dir.glob(f"{error_path.stem}__*.jsonl")
                if path.is_file()
            },
            key=lambda p: p.name,
        )
        _append_jsonl_files(merged_errors, error_path)
        result["batch_errors_path"] = str(error_path)
    return result
