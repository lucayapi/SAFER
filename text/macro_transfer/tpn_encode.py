"""Encodage modulaire pour le pipeline TPN (encodeur gelé au choix)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch

from contrastive_methods.config import ContrastiveConfig, load_contrastive_config
from contrastive_methods.eval_geometry import encode_contrastive_texts
from macro_transfer.encode import load_target_metadata
from safer_core.paths import resolve_repo_path
from scgm_text.data_metadata import LABEL2ID, VALID_LABELS, load_filtered_metadata
from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
from scgm_text.utils_io import create_doc_id_if_missing
from scgm_text.dataset_text_raw import TextRawDataset

logger = logging.getLogger(__name__)

EncoderName = Literal["softtriple", "supcon", "batch_triplet", "scgm_text"]


def resolve_scgm_encode_log_every_batches(
    yaml_value: Optional[Union[int, str]] = None,
) -> int:
    """
    Journaliser la progression tous les N batches (défaut 1 = chaque batch).

    Priorité : variables d'environnement > ``encoding.log_every_batches`` (YAML) > 1.
    """
    for key in ("TPN_ENCODE_LOG_EVERY_BATCHES", "TPN_ENCODE_LOG_EVERY"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return max(1, int(raw))
    if yaml_value is not None and str(yaml_value).strip() != "":
        return max(1, int(yaml_value))
    return 1


def scgm_encode_log_every_batches() -> int:
    """Rétrocompat tests / appels sans YAML."""
    return resolve_scgm_encode_log_every_batches()


def _should_log_scgm_encode_batch(
    batch_idx: int,
    total_batches: int,
    *,
    log_every_batches: int,
    force: bool = False,
) -> bool:
    if force or total_batches <= 0:
        return bool(force)
    if batch_idx <= 0:
        return False
    if batch_idx == 1 or batch_idx >= total_batches:
        return True
    every = max(1, log_every_batches)
    return batch_idx % every == 0


def _log_scgm_encode_progress(
    log_label: str,
    done: int,
    total: int,
    *,
    batch_idx: int,
    total_batches: int,
    t0: float,
    input_mode: str,
    batch_size: int,
    device: str,
    force: bool = False,
    log_every_batches: int = 1,
) -> None:
    """Progression encodage SCGM (visible dans slurm-*.out via tail -f)."""
    if total <= 0:
        return
    log_every = max(1, int(log_every_batches))
    if not force and not _should_log_scgm_encode_batch(
        batch_idx,
        total_batches,
        log_every_batches=log_every,
        force=False,
    ):
        return
    elapsed = time.monotonic() - t0
    pct = 100.0 * float(done) / float(total)
    rate = float(done) / elapsed if elapsed > 0 else 0.0
    eta_s = (float(total - done) / rate) if rate > 0 else 0.0
    logger.info(
        "SCGM encode [%s] batch %d/%d | %d/%d (%.1f%%) mode=%s eff_batch=%d "
        "device=%s elapsed=%.0fs eta=%.0fs",
        log_label,
        batch_idx,
        total_batches,
        done,
        total,
        pct,
        input_mode,
        batch_size,
        device,
        elapsed,
        eta_s,
    )

CONTRASTIVE_ENCODERS: tuple[str, ...] = ("softtriple", "supcon", "batch_triplet")


def tpn_method_name(base_method: str) -> str:
    """Nom de sortie macro_transfer : ``tpn_<encodeur>``."""
    base = str(base_method).strip()
    if not base:
        raise ValueError("base_method vide")
    if base.startswith("tpn_"):
        return base
    return f"tpn_{base}"


def validate_encoder_name(method: str) -> EncoderName:
    m = str(method).strip()
    if m.startswith("tpn_"):
        m = m[4:]
    allowed = (*CONTRASTIVE_ENCODERS, "scgm_text")
    if m not in allowed:
        raise ValueError(
            f"Encodeur TPN non supporté : {method!r}. Attendu : {', '.join(allowed)}"
        )
    return m  # type: ignore[return-value]


def default_contrastive_config_path(base_method: str, repo_anchor: Path) -> Path:
    return resolve_repo_path(
        f"configs/methods/{base_method}.yaml",
        repo_root=repo_anchor,
    )


def resolve_tpn_checkpoint(
    base_method: str,
    method_cfg: dict,
    checkpoints_block: Optional[dict] = None,
    *,
    explicit_checkpoint: Optional[str] = None,
    base_method_overridden: bool = False,
) -> str:
    """
    Priorité : explicit_checkpoint > (si override CLI) checkpoints[base_method]
    > method.checkpoint > checkpoints[base_method].
    """
    if explicit_checkpoint:
        return str(explicit_checkpoint)
    block = checkpoints_block or {}
    if base_method_overridden:
        ckpt = block.get(base_method)
        if ckpt:
            return str(ckpt)
        raise ValueError(
            f"Checkpoint manquant pour {base_method!r} "
            f"(base_method surchargé par CLI : utiliser checkpoints.{base_method})"
        )
    ckpt = method_cfg.get("checkpoint")
    if ckpt:
        return str(ckpt)
    ckpt = block.get(base_method)
    if ckpt:
        return str(ckpt)
    raise ValueError(
        f"Checkpoint manquant pour {base_method!r} "
        f"(method.checkpoint ou checkpoints.{base_method})"
    )


def _resolve_scgm_input_mode(
    checkpoint_args: dict,
    checkpoint: Optional[dict] = None,
) -> str:
    """
    Déduit le mode d'encodage SCGM à partir des métadonnées du checkpoint.

    Les checkpoints end2end récents n'ont pas toujours ``input_mode`` ; l'ancien défaut
    ``precomputed_embeddings`` provoquait à tort l'exigence de ``emb_csv``.
    """
    if checkpoint_args.get("input_mode") == "precomputed_embeddings":
        return "precomputed_embeddings"
    if (
        checkpoint_args.get("pipeline") == "end2end_text"
        or (checkpoint or {}).get("pipeline") == "end2end_text"
        or checkpoint_args.get("backbone_model_name_or_path")
        or checkpoint_args.get("backbone_name")
    ):
        return "text"
    return "text"


def scgm_checkpoint_input_mode(checkpoint: str) -> str:
    """Retourne le mode d'encodage SCGM : end2end checkpoints → text."""
    from scgm_text.checkpoint_io import load_scgm_checkpoint

    _, checkpoint_args, raw = load_scgm_checkpoint(checkpoint, map_location="cpu")
    return _resolve_scgm_input_mode(checkpoint_args, raw)


def _load_contrastive_cfg(
    base_method: str,
    contrastive_config: Optional[Union[str, Path]],
    repo_anchor: Path,
) -> ContrastiveConfig:
    if contrastive_config:
        cfg_path = resolve_repo_path(contrastive_config, repo_root=repo_anchor)
    else:
        cfg_path = default_contrastive_config_path(base_method, repo_anchor)
    return load_contrastive_config(base_method, config_path=cfg_path)


def _encode_contrastive_for_tpn(
    base_method: str,
    texts: list[str],
    checkpoint: str,
    *,
    contrastive_config: Optional[Union[str, Path]],
    device: str,
    batch_size: int,
    repo_anchor: Path,
) -> np.ndarray:
    cfg = _load_contrastive_cfg(base_method, contrastive_config, repo_anchor)
    ckpt_dir = Path(checkpoint)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    z = encode_contrastive_texts(
        cfg,
        texts,
        checkpoint_dir=ckpt_dir,
        batch_size=int(batch_size),
        device=device,
    )
    return np.asarray(z, dtype=np.float64)


def _prepare_scgm_metadata(
    data_csv: str,
    emb_csv: str,
    *,
    text_col: str,
    label_col: str,
    pred_ok_col: str,
    group_col: str,
    filter_pred_ok: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if filter_pred_ok:
        meta = load_filtered_metadata(
            data_csv,
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            text_col=text_col,
        )
        meta, dim_columns = merge_metadata_with_embeddings(meta, emb_csv)
        return meta, dim_columns

    meta = load_target_metadata(data_csv, text_col=text_col)
    meta = create_doc_id_if_missing(meta.copy())
    if label_col in meta.columns:
        labels = meta[label_col]
        valid = labels.notna() & labels.isin(VALID_LABELS)
        meta["label_id"] = labels.map(LABEL2ID).fillna(0).astype(np.int64)
        meta.loc[~valid, "label_id"] = 0
    else:
        meta["label_id"] = np.int64(0)
    if group_col not in meta.columns:
        meta[group_col] = meta.get("doc_id", np.arange(len(meta)))
    meta, dim_columns = merge_metadata_with_embeddings(meta, emb_csv)
    return meta, dim_columns


def _project_scgm_embeddings(
    checkpoint: str,
    data_csv: str,
    emb_csv: str,
    *,
    text_col: str,
    label_col: str,
    pred_ok_col: str,
    group_col: str,
    filter_pred_ok: bool,
    batch_size: int,
    max_seq_length: int,
    device: str,
    n_expected: Optional[int] = None,
    log_label: str = "corpus",
    log_every_batches: int = 1,
) -> np.ndarray:
    from scgm_text.batch_utils import batch_to_device, forward_features
    from scgm_text.checkpoint_io import load_scgm_checkpoint
    from scgm_text.collate import make_text_collate_fn
    from scgm_text.dataset_text_embeddings import TextEmbeddingDataset

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, checkpoint_args, raw_ckpt = load_scgm_checkpoint(checkpoint, map_location="cpu")
    model.to(dev)
    model.eval()
    input_mode = _resolve_scgm_input_mode(checkpoint_args, raw_ckpt)
    projected: list[np.ndarray] = []
    t0 = time.monotonic()
    logger.info(
        "SCGM encode [%s] démarré mode=%s n_expected=%s checkpoint=%s",
        log_label,
        input_mode,
        n_expected if n_expected is not None else "?",
        checkpoint,
    )

    if input_mode == "text":
        if filter_pred_ok:
            dataset = TextRawDataset(
                data_csv=data_csv,
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
                text_col=text_col,
            )
        else:
            meta = load_target_metadata(data_csv, text_col=text_col)
            meta = create_doc_id_if_missing(meta.copy())
            if label_col in meta.columns:
                labels = meta[label_col]
                valid = labels.notna() & labels.isin(VALID_LABELS)
                meta["label_id"] = labels.map(LABEL2ID).fillna(0).astype(np.int64)
                meta.loc[~valid, "label_id"] = 0
            else:
                meta["label_id"] = np.int64(0)
            if group_col not in meta.columns:
                meta[group_col] = meta.get("doc_id", np.arange(len(meta)))
            dataset = TextRawDataset(
                data_csv=data_csv,
                label_col=label_col,
                pred_ok_col=pred_ok_col,
                group_col=group_col,
                text_col=text_col,
                metadata_df=meta,
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_args.get("backbone_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B")
        )
        collate_fn = make_text_collate_fn(tokenizer, max_seq_length)
        eff_batch = min(batch_size, 32)
        n_total = len(dataset)
        logger.info(
            "SCGM encode [%s] text: %d unités, eff_batch=%d max_seq=%d",
            log_label,
            n_total,
            eff_batch,
            max_seq_length,
        )
        total_batches = (n_total + eff_batch - 1) // eff_batch
        log_every = max(1, int(log_every_batches))
        logger.info(
            "SCGM encode [%s] progression : 1 log tous les %d batch(es) (~%d lignes)",
            log_label,
            log_every,
            (total_batches + log_every - 1) // log_every + 1,
        )
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, n_total, eff_batch), start=1):
                end = min(start + eff_batch, n_total)
                items = [dataset[index] for index in range(start, end)]
                batch = batch_to_device(collate_fn(items), dev)
                features = forward_features(model, batch)
                projected.append(features.cpu().numpy())
                _log_scgm_encode_progress(
                    log_label,
                    end,
                    n_total,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    t0=t0,
                    input_mode=input_mode,
                    batch_size=eff_batch,
                    device=str(dev),
                    force=(end == n_total),
                    log_every_batches=log_every,
                )
    else:
        if not emb_csv:
            raise ValueError("scgm_text (mode precomputed_embeddings) requiert emb_csv")
        meta, dim_columns = _prepare_scgm_metadata(
            data_csv,
            emb_csv,
            text_col=text_col,
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            filter_pred_ok=filter_pred_ok,
        )
        dataset = TextEmbeddingDataset(
            data_csv=data_csv,
            emb_csv=emb_csv,
            label_col=label_col,
            pred_ok_col=pred_ok_col,
            group_col=group_col,
            metadata_df=meta,
            dim_columns=dim_columns,
        )
        n_total = len(dataset)
        logger.info(
            "SCGM encode [%s] precomputed: %d unités, batch=%d emb_csv=%s",
            log_label,
            n_total,
            batch_size,
            emb_csv,
        )
        total_batches = (n_total + batch_size - 1) // batch_size
        log_every = max(1, int(log_every_batches))
        logger.info(
            "SCGM encode [%s] progression : 1 log tous les %d batch(es) (~%d lignes)",
            log_label,
            log_every,
            (total_batches + log_every - 1) // log_every + 1,
        )
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, n_total, batch_size), start=1):
                end = min(start + batch_size, n_total)
                batch_embeddings = []
                for index in range(start, end):
                    embedding, _, _ = dataset[index]
                    batch_embeddings.append(embedding)
                embeddings = torch.stack(batch_embeddings).to(dev)
                features = forward_features(model, embeddings)
                projected.append(features.cpu().numpy())
                _log_scgm_encode_progress(
                    log_label,
                    end,
                    n_total,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    t0=t0,
                    input_mode=input_mode,
                    batch_size=batch_size,
                    device=str(dev),
                    force=(end == n_total),
                    log_every_batches=log_every,
                )

    z = np.concatenate(projected, axis=0)
    logger.info(
        "SCGM encode [%s] terminé shape=%s elapsed=%.0fs",
        log_label,
        tuple(z.shape),
        time.monotonic() - t0,
    )
    if n_expected is not None and len(z) != n_expected:
        raise ValueError(
            f"SCGM projection : {len(z)} lignes projetées, attendu {n_expected} "
            f"(filter_pred_ok={filter_pred_ok})"
        )
    return np.asarray(z, dtype=np.float64)


def encode_corpus_for_tpn(
    base_method: str,
    texts: list[str],
    checkpoint: str,
    *,
    contrastive_config: Optional[Union[str, Path]] = None,
    data_csv: Optional[str] = None,
    emb_csv: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 8,
    repo_anchor: Optional[Path] = None,
    text_col: str = "sentence",
    label_col: str = "pred_label",
    pred_ok_col: str = "pred_ok",
    group_col: str = "accident_id",
    filter_pred_ok: bool = True,
    max_seq_length: int = 256,
    scgm_infer_batch_size: int = 512,
    log_label: str = "corpus",
    log_every_batches: int = 1,
) -> np.ndarray:
    """
    Encode un corpus pour TPN.

    - ``softtriple`` / ``supcon`` / ``batch_triplet`` : liste ``texts`` (ordre aligné).
    - ``scgm_text`` : ``data_csv`` (+ ``emb_csv`` si embeddings figés) ; ``texts`` sert
      uniquement à valider ``len(texts)`` si fourni.
    """
    encoder = validate_encoder_name(base_method)
    anchor = repo_anchor or Path(__file__).resolve().parents[1]

    if encoder in CONTRASTIVE_ENCODERS:
        if not texts:
            raise ValueError("texts vide pour encodeur contrastif")
        return _encode_contrastive_for_tpn(
            encoder,
            texts,
            checkpoint,
            contrastive_config=contrastive_config,
            device=device,
            batch_size=batch_size,
            repo_anchor=anchor,
        )

    if encoder != "scgm_text":
        raise ValueError(f"Encodeur inconnu : {encoder}")

    if not data_csv:
        raise ValueError("scgm_text requiert data_csv")
    n_expected = len(texts) if texts else None
    return _project_scgm_embeddings(
        checkpoint,
        data_csv,
        emb_csv or "",
        text_col=text_col,
        label_col=label_col,
        pred_ok_col=pred_ok_col,
        group_col=group_col,
        filter_pred_ok=filter_pred_ok,
        batch_size=scgm_infer_batch_size,
        max_seq_length=max_seq_length,
        device=device,
        n_expected=n_expected,
        log_label=log_label,
        log_every_batches=log_every_batches,
    )
