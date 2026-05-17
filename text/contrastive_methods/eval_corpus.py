"""Évaluation géométrique BTP / test avec best model contrastif."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.eval_geometry import evaluate_embeddings_geometry
from contrastive_methods.export import export_st_embeddings, embeddings_to_dataframe
from contrastive_methods.metrics import METHOD_DISPLAY
from safer_core.io import save_metrics_geometry
from safer_core.paths import layout_method_output


def _load_st_model(checkpoint_dir: Path, cfg: ContrastiveConfig):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(checkpoint_dir), trust_remote_code=True)


def _load_softtriple_encoder(checkpoint_dir: Path, cfg: ContrastiveConfig, device: str):
    import torch
    from contrastive_methods.losses.softtriple import HFTextEncoder, encode_texts_with_hf_encoder

    encoder = HFTextEncoder(cfg.backbone_name, gradient_checkpointing=False).to(device)
    ckpt = checkpoint_dir / "hf_model.bin"
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    encoder.encoder.load_state_dict(state)
    return encoder, encode_texts_with_hf_encoder


def evaluate_contrastive_on_csv(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    data_csv: Path,
    *,
    corpus: str = "btp",
    embeddings_out: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Encode un CSV avec le best checkpoint et calcule metrics_geometry."""
    cfg_eval = ContrastiveConfig(
        method_name=cfg.method_name,
        dataset_path=Path(data_csv),
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        group_col=cfg.group_col,
        pred_ok_col=cfg.pred_ok_col,
        backbone_name=cfg.backbone_name,
        max_seq_length=cfg.max_seq_length,
        encode_batch_size=cfg.encode_batch_size,
        eval_batch_size=cfg.eval_batch_size,
    )
    dataset = prepare_text_dataset(cfg_eval)
    texts = dataset.metadata_df[dataset.text_col].astype(str).tolist()
    labels = dataset.metadata_df[cfg.label_col].to_numpy()
    display = METHOD_DISPLAY.get(cfg.method_name, cfg.method_name)

    if cfg.method_name == "softtriple":
        from contrastive_methods.st_common import get_device

        device = get_device()
        encoder, encode_fn = _load_softtriple_encoder(checkpoint_dir, cfg, device)
        emb = encode_fn(
            encoder,
            texts,
            batch_size=cfg.encode_batch_size,
            device=device,
            max_length=cfg.max_seq_length,
        )
    else:
        model = _load_st_model(checkpoint_dir, cfg)
        emb = model.encode(
            texts,
            batch_size=cfg.encode_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    if embeddings_out is not None:
        frame = embeddings_to_dataframe(dataset.metadata_df["doc_id"].to_numpy(), emb)
        embeddings_out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(embeddings_out, index=False)

    row = evaluate_embeddings_geometry(emb, labels, method=f"{display}_{corpus}")
    if metrics_dir is not None:
        stem = f"metrics_geometry_{corpus}"
        save_metrics_geometry(row, metrics_dir, stem=stem)
    return row


def evaluate_btp_and_test(
    cfg: ContrastiveConfig,
    checkpoint_dir: Path,
    output_root: Path,
) -> Dict[str, Path]:
    """Écrit metrics_geometry_btp.csv et metrics_geometry_test.csv."""
    layout = layout_method_output(cfg.method_name, str(output_root))
    metrics_dir = Path(layout["metrics"])
    emb_dir = Path(layout["embeddings"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    btp_csv = cfg.dataset_path
    test_csv = cfg.test_data_csv
    paths = {}
    evaluate_contrastive_on_csv(
        cfg,
        checkpoint_dir,
        btp_csv,
        corpus="btp",
        embeddings_out=emb_dir / "final_embeddings_btp.csv",
        metrics_dir=metrics_dir,
    )
    paths["btp"] = metrics_dir / "metrics_geometry_btp.csv"
    if test_csv.is_file():
        evaluate_contrastive_on_csv(
            cfg,
            checkpoint_dir,
            test_csv,
            corpus="test",
            embeddings_out=emb_dir / "final_embeddings_test.csv",
            metrics_dir=metrics_dir,
        )
        paths["test"] = metrics_dir / "metrics_geometry_test.csv"
    return paths
