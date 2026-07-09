"""Évaluation géométrique BTP / test avec best model contrastif."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from contrastive_methods.config import ContrastiveConfig
from contrastive_methods.data import prepare_text_dataset
from contrastive_methods.eval_geometry import encode_contrastive_texts, evaluate_embeddings_geometry
from contrastive_methods.export import embeddings_to_dataframe
from contrastive_methods.metrics import METHOD_DISPLAY
from contrastive_methods.post_eval import run_post_eval_on_corpus
from contrastive_methods.hf_training_common import get_device
from safer_core.io import save_metrics_geometry
from safer_core.paths import layout_method_output
from safer_core.test_corpus import method_test_results_dir


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
        backbone_trainable=cfg.backbone_trainable,
        train_last_n_layers=cfg.train_last_n_layers,
        cache_backbone_embeddings=cfg.cache_backbone_embeddings,
        use_projector=cfg.use_projector,
        projection=cfg.projection,
        hiddim=cfg.hiddim,
    )
    dataset = prepare_text_dataset(cfg_eval)
    texts = dataset.metadata_df[dataset.text_col].astype(str).tolist()
    labels = dataset.metadata_df[cfg.label_col].to_numpy()
    display = METHOD_DISPLAY.get(cfg.method_name, cfg.method_name)

    emb = encode_contrastive_texts(
        cfg,
        texts,
        checkpoint_dir=checkpoint_dir,
        batch_size=cfg.encode_batch_size,
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
    """Écrit metrics_geometry_btp.csv, metrics_geometry_test.csv et classification si post_eval."""
    layout = layout_method_output(cfg.method_name, str(output_root))
    metrics_dir = Path(layout["metrics"])
    emb_dir = Path(layout["embeddings"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    btp_csv = cfg.dataset_path
    test_csv = cfg.test_data_csv
    paths: Dict[str, Path] = {}
    btp_dataset = prepare_text_dataset(cfg)
    btp_df = btp_dataset.metadata_df

    evaluate_contrastive_on_csv(
        cfg,
        checkpoint_dir,
        btp_csv,
        corpus="btp",
        embeddings_out=emb_dir / "final_embeddings_btp.csv",
        metrics_dir=metrics_dir,
    )
    paths["btp"] = metrics_dir / "metrics_geometry_btp.csv"

    if cfg.post_eval_enabled:
        run_post_eval_on_corpus(
            cfg,
            checkpoint_dir,
            btp_df,
            btp_df,
            cfg.text_col,
            device,
            corpus="btp",
            metrics_dir=metrics_dir,
        )
        paths["classification_btp"] = metrics_dir / "metrics_classification_btp.csv"

    if test_csv.is_file():
        test_root = method_test_results_dir(cfg.method_name, cfg.test_corpus)
        test_metrics = test_root / "metrics"
        test_emb_out = test_root / "embeddings"
        test_metrics.mkdir(parents=True, exist_ok=True)
        test_emb_out.mkdir(parents=True, exist_ok=True)
        evaluate_contrastive_on_csv(
            cfg,
            checkpoint_dir,
            test_csv,
            corpus="test",
            embeddings_out=test_emb_out / "final_embeddings_test.csv",
            metrics_dir=test_metrics,
        )
        paths["test"] = test_metrics / "metrics_geometry_test.csv"
        if cfg.post_eval_enabled:
            test_cfg = ContrastiveConfig(
                method_name=cfg.method_name,
                dataset_path=Path(test_csv),
                text_col=cfg.text_col,
                label_col=cfg.label_col,
                group_col=cfg.group_col,
                pred_ok_col=cfg.pred_ok_col,
                backbone_name=cfg.backbone_name,
                max_seq_length=cfg.max_seq_length,
                encode_batch_size=cfg.encode_batch_size,
                eval_batch_size=cfg.eval_batch_size,
                backbone_trainable=cfg.backbone_trainable,
                train_last_n_layers=cfg.train_last_n_layers,
                cache_backbone_embeddings=cfg.cache_backbone_embeddings,
                use_projector=cfg.use_projector,
                projection=cfg.projection,
                hiddim=cfg.hiddim,
                post_eval_enabled=cfg.post_eval_enabled,
                post_eval_classifier=cfg.post_eval_classifier,
                post_eval_class_weight=cfg.post_eval_class_weight,
                post_eval_oversampling=cfg.post_eval_oversampling,
            )
            test_dataset = prepare_text_dataset(test_cfg)
            test_df = test_dataset.metadata_df
            run_post_eval_on_corpus(
                cfg,
                checkpoint_dir,
                btp_df,
                test_df,
                cfg.text_col,
                device,
                corpus="test",
                metrics_dir=test_metrics,
            )
            paths["classification_test"] = test_metrics / "metrics_classification_test.csv"
    return paths
