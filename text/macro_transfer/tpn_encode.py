"""Encodage modulaire pour le pipeline TPN (encodeur gelé au choix)."""

from __future__ import annotations

import logging
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


def scgm_checkpoint_input_mode(checkpoint: str) -> str:
    """Retourne le mode d'encodage SCGM : end2end checkpoints → text."""
    from scgm_text.checkpoint_io import load_scgm_checkpoint

    _, checkpoint_args, _ = load_scgm_checkpoint(checkpoint, map_location="cpu")
    if checkpoint_args.get("pipeline") == "end2end_text":
        return "text"
    if checkpoint_args.get("input_mode") == "precomputed_embeddings":
        return "precomputed_embeddings"
    return "text"


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
) -> np.ndarray:
    from scgm_text.batch_utils import batch_to_device, forward_features
    from scgm_text.checkpoint_io import load_scgm_checkpoint
    from scgm_text.collate import make_text_collate_fn
    from scgm_text.dataset_text_embeddings import TextEmbeddingDataset

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, checkpoint_args, _ = load_scgm_checkpoint(checkpoint, map_location="cpu")
    model.to(dev)
    model.eval()
    input_mode = checkpoint_args.get("input_mode", "precomputed_embeddings")
    projected: list[np.ndarray] = []

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
        with torch.no_grad():
            for start in range(0, len(dataset), eff_batch):
                end = min(start + eff_batch, len(dataset))
                items = [dataset[index] for index in range(start, end)]
                batch = batch_to_device(collate_fn(items), dev)
                features = forward_features(model, batch)
                projected.append(features.cpu().numpy())
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
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch_embeddings = []
                for index in range(start, end):
                    embedding, _, _ = dataset[index]
                    batch_embeddings.append(embedding)
                embeddings = torch.stack(batch_embeddings).to(dev)
                features = forward_features(model, embeddings)
                projected.append(features.cpu().numpy())

    z = np.concatenate(projected, axis=0)
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
    )
