"""Orchestration entraînement supervised_macro_ft (CV + fit final)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import resolve_test_corpus
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_raw import TextRawDataset
from supervised_macro_ft.backbone_scaler import BackboneScaler, should_standardize_backbone
from supervised_macro_ft.checkpoint_io import save_checkpoint
from safer_core.classification_eval import (
    build_and_save_predictions,
    export_projected_embeddings,
    resolve_test_corpora,
    save_classification_outputs,
    summarize_ood_classification,
)
from supervised_macro_ft.class_balance import balanced_oversample_indices
from supervised_macro_ft.class_balance import resolve_train_balance
from supervised_macro_ft.config_validation import validate_macro_ft_startup
from supervised_macro_ft.cv import run_group_kfold_cv
from supervised_macro_ft.embedding_cache import (
    BackboneHiddenDataset,
    collate_hidden_batch,
    encode_projected_matrix,
    load_backbone_hidden_for_corpus,
    load_or_build_backbone_hidden,
    should_cache_backbone_embeddings,
)
from supervised_macro_ft.model import SupervisedMacroModel, model_kwargs_from_cfg
from supervised_macro_ft.run_logging import log_cv_summary, log_phase, log_run_complete, log_test_metrics
from supervised_macro_ft.train_loop import (
    build_class_weights,
    evaluate_loader_with_predictions,
    fit_model,
)

logger = logging.getLogger(__name__)


def _load_tokenizer(backbone_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def _resolve_cfg(
    config_path: str | Path | None,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    training_overrides: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    test_corpora_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if cfg is None:
        if config_path is None:
            raise ValueError("config_path ou cfg requis")
        cfg = load_yaml(Path(config_path))
    if model_overrides:
        model_section = dict(cfg.get("model") or {})
        model_section.update(model_overrides)
        cfg = {**cfg, "model": model_section}
    if training_overrides:
        train_section = dict(cfg.get("training") or {})
        train_section.update(training_overrides)
        cfg = {**cfg, "training": train_section}
    if test_corpora_override:
        cfg = {**cfg, "test_corpora": list(test_corpora_override)}
    return cfg


def _prepare_backbone_hidden(
    *,
    anchor: Path,
    out_dir: Path,
    dataset: TextRawDataset,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    tokenizer,
    device: torch.device,
    backbone_hidden: Optional[np.ndarray] = None,
    shared_cache_dir: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if backbone_hidden is not None:
        return backbone_hidden, model_cfg
    use_hidden_cache = should_cache_backbone_embeddings(model_cfg)
    if not use_hidden_cache:
        return None, model_cfg
    emb_csv = model_cfg.get("backbone_emb_csv")
    if emb_csv:
        model_cfg = {
            **model_cfg,
            "backbone_emb_csv": str(resolve_repo_path(str(emb_csv), repo_root=anchor)),
        }
    cache_dir = shared_cache_dir if shared_cache_dir is not None else out_dir / "cache"
    cache_model = SupervisedMacroModel(**model_kwargs_from_cfg(model_cfg)).to(device)
    hidden = load_or_build_backbone_hidden(
        model=cache_model,
        dataset=dataset,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        cache_dir=cache_dir,
        device=device,
    )
    logger.info(
        "[macro_ft] Mode cache backbone actif : entraînement projecteur+tête uniquement (%s)",
        hidden.shape,
    )
    return hidden, model_cfg


def prepare_shared_backbone_hidden(
    cfg: Dict[str, Any],
    *,
    cache_dir: str | Path,
    anchor: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Charge ou construit le cache backbone partagé (tuning)."""
    anchor = anchor or Path(__file__).resolve().parents[1]
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})
    if not should_cache_backbone_embeddings(model_cfg):
        return None
    backbone_name = str(model_cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B"))
    tokenizer = _load_tokenizer(backbone_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_csv = resolve_repo_path(data_cfg.get("data_csv", "dataset/data_btp.csv"), repo_root=anchor)
    dataset = TextRawDataset(
        str(data_csv),
        label_col=str(data_cfg.get("label_col", "pred_label")),
        pred_ok_col=str(data_cfg.get("pred_ok_col", "pred_ok")),
        group_col=str(data_cfg.get("group_col", "accident_id")),
        text_col=data_cfg.get("text_col"),
    )
    hidden, _ = _prepare_backbone_hidden(
        anchor=anchor,
        out_dir=Path(cache_dir),
        dataset=dataset,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        device=device,
        shared_cache_dir=Path(cache_dir),
    )
    return hidden


def run_supervised_macro_ft_cv(
    cfg: Dict[str, Any],
    *,
    combo_output_dir: str | Path,
    backbone_hidden: Optional[np.ndarray] = None,
    shared_cache_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """CV seule (GroupKFold) ; retourne métriques agrégées dont mean_balanced_accuracy."""
    result = run_supervised_macro_ft_training(
        None,
        cv_only=True,
        cfg=cfg,
        output_dir_override=combo_output_dir,
        backbone_hidden=backbone_hidden,
        shared_cache_dir=shared_cache_dir,
    )
    cv_summary = pd.DataFrame(result.get("cv_summary") or [])
    row: Dict[str, Any] = {"combo_output_dir": str(combo_output_dir)}
    if not cv_summary.empty:
        row.update(cv_summary.iloc[0].to_dict())
    row["selection_score"] = float(row.get("mean_balanced_accuracy", float("nan")))
    return row


def run_supervised_macro_ft_training(
    config_path: str | Path | None,
    *,
    cv_only: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
    output_dir_override: Optional[str | Path] = None,
    backbone_hidden: Optional[np.ndarray] = None,
    shared_cache_dir: Optional[str | Path] = None,
    training_overrides: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    test_corpora_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    anchor = Path(__file__).resolve().parents[1]
    cfg = _resolve_cfg(
        config_path,
        cfg=cfg,
        training_overrides=training_overrides,
        model_overrides=model_overrides,
        test_corpora_override=test_corpora_override,
    )
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})
    method_name = str(cfg.get("method_name", "supervised_macro_ft"))
    if output_dir_override is not None:
        out_dir = Path(output_dir_override)
    else:
        out_dir = resolve_repo_path(str(cfg.get("output_dir", "output/supervised_macro_ft")), repo_root=anchor)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = str(model_cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B"))
    tokenizer = _load_tokenizer(backbone_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolve_train_balance(model_cfg)

    data_csv = resolve_repo_path(data_cfg.get("data_csv", "dataset/data_btp.csv"), repo_root=anchor)
    dataset = TextRawDataset(
        str(data_csv),
        label_col=str(data_cfg.get("label_col", "pred_label")),
        pred_ok_col=str(data_cfg.get("pred_ok_col", "pred_ok")),
        group_col=str(data_cfg.get("group_col", "accident_id")),
        text_col=data_cfg.get("text_col"),
    )

    batch_size = int(train_cfg.get("batch_size", 32))
    max_length = int(model_cfg.get("max_seq_length", 256))
    collate_fn = make_text_collate_fn(tokenizer, max_length)

    shared_dir = Path(shared_cache_dir) if shared_cache_dir is not None else None
    backbone_hidden, model_cfg = _prepare_backbone_hidden(
        anchor=anchor,
        out_dir=out_dir,
        dataset=dataset,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        device=device,
        backbone_hidden=backbone_hidden,
        shared_cache_dir=shared_dir,
    )
    use_hidden_cache = backbone_hidden is not None and should_cache_backbone_embeddings(model_cfg)
    use_oversampling, class_weight_mode = validate_macro_ft_startup(
        model_cfg,
        train_cfg,
        device=device,
        n_samples=len(dataset),
        backbone_hidden_available=backbone_hidden is not None,
    )
    if should_standardize_backbone(model_cfg) and not use_hidden_cache:
        logger.warning(
            "[macro_ft] standardize_backbone=true requiert cache_backbone_embeddings (backbone gelé) ; "
            "standardisation ignorée."
        )

    label_col = str(data_cfg.get("label_col", "pred_label"))

    n_folds = int(train_cfg.get("n_folds", 3))
    seed = int(train_cfg.get("seed", 42))
    selection_metric = str(train_cfg.get("selection_metric", "balanced_accuracy"))
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_phase(
        "Phase 1/3 — CV GroupKFold",
        detail=f"{n_folds} folds, metric={selection_metric}",
    )
    fold_rows, cv_summary, cv_history = run_group_kfold_cv(
        dataset,
        tokenizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        n_folds=n_folds,
        seed=seed,
        device=device,
        fold_out_root=str(out_dir),
        backbone_hidden=backbone_hidden,
        save_fold_checkpoints=not cv_only,
    )
    cv_dir = out_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(cv_dir / "cv_per_fold.csv", index=False)
    cv_summary.to_csv(cv_dir / "cv_summary.csv", index=False)
    cv_summary.to_csv(out_dir / "kfold_summary.csv", index=False)
    if not cv_history.empty:
        cv_history.to_csv(cv_dir / "train_history.csv", index=False)
        logger.info("[macro_ft] Historique CV exporté : %s", cv_dir / "train_history.csv")
    log_cv_summary(cv_summary, selection_metric=selection_metric)

    if cv_only:
        return {
            "method_name": method_name,
            "output_dir": str(out_dir),
            "cv_summary": cv_summary.to_dict(orient="records"),
            "backbone_cache_used": bool(use_hidden_cache),
            "selection_score": float(
                cv_summary.iloc[0]["mean_balanced_accuracy"] if not cv_summary.empty else float("nan")
            ),
        }

    # Fit final 100 % BTP
    log_phase("Phase 2/3 — Fit final 100 % BTP")
    model = SupervisedMacroModel(**model_kwargs_from_cfg(model_cfg)).to(device)

    if use_hidden_cache and backbone_hidden is not None and should_standardize_backbone(model_cfg):
        all_idx = np.arange(len(dataset), dtype=np.int64)
        model.set_backbone_scaler(BackboneScaler.fit(backbone_hidden, all_idx))

    class_weight = build_class_weights(
        dataset.label_ids.tolist(),
        int(model_cfg.get("n_classes", 4)),
        class_weight_mode,
    )
    if use_hidden_cache and backbone_hidden is not None:
        final_idx = np.arange(len(dataset), dtype=np.int64)
        if use_oversampling:
            final_idx = balanced_oversample_indices(
                dataset.label_ids, final_idx, seed=seed
            )
            logger.info(
                "[macro_ft] Fit final oversampling : %d -> %d exemples",
                len(dataset),
                len(final_idx),
            )
        final_ds = BackboneHiddenDataset(backbone_hidden, dataset.label_ids, final_idx)
        train_loader = DataLoader(
            final_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_hidden_batch,
        )
    else:
        if use_oversampling:
            from torch.utils.data import Subset

            final_idx = balanced_oversample_indices(
                dataset.label_ids,
                np.arange(len(dataset), dtype=np.int64),
                seed=seed,
            )
            logger.info(
                "[macro_ft] Fit final oversampling : %d -> %d exemples",
                len(dataset),
                len(final_idx),
            )
            final_ds = Subset(dataset, final_idx.tolist())
            train_loader = DataLoader(
                final_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
            )
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model, final_metrics, final_history = fit_model(
        model,
        train_loader,
        val_loader=None,
        train_cfg=train_cfg,
        device=device,
        class_weight=class_weight,
        run_label="final_fit",
    )
    logger.info(
        "[macro_ft] Fit final terminé — train_loss=%.4f (epoch %s)",
        float(final_metrics.get("train_loss", float("nan"))),
        final_metrics.get("epoch", "?"),
    )
    if final_history:
        final_hist_df = pd.DataFrame(
            [{"phase": "final", "fold": -1, **row} for row in final_history]
        )
        final_hist_df.to_csv(out_dir / "train_history_final.csv", index=False)
        logger.info("[macro_ft] Historique fit final exporté : %s", out_dir / "train_history_final.csv")

    ckpt_dir = out_dir / "checkpoints" / "best_model"
    save_checkpoint(
        model,
        ckpt_dir,
        config={**model_cfg, **train_cfg, "method_name": cfg.get("method_name", "supervised_macro_ft")},
    )

    exp_cfg = dict(cfg.get("exports") or {})
    if bool(exp_cfg.get("save_btp_embeddings", True)):
        emb_dir = out_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        meta_df = dataset.get_metadata_df()
        text_col = str(data_cfg.get("text_col", "sentence"))
        group_col = str(data_cfg.get("group_col", "accident_id"))

        if use_hidden_cache and backbone_hidden is not None:
            z_btp = encode_projected_matrix(
                model,
                backbone_hidden,
                batch_size=batch_size,
                device=device,
                show_progress=True,
                progress_desc="export_btp_z",
            )
        else:
            from supervised_macro_ft.inference import encode_texts

            texts = meta_df[text_col].astype(str).tolist()
            z_btp = encode_texts(
                model,
                tokenizer,
                texts,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
                show_progress=True,
                progress_desc="export_btp_z",
            )

        export_projected_embeddings(
            z_btp,
            meta_df,
            emb_dir,
            "btp",
            label_col=label_col,
            group_col=group_col,
            text_col=text_col,
        )
        logger.info("[macro_ft] Embeddings BTP exportés : %s", emb_dir)

    test_corpora = resolve_test_corpora(cfg)
    test_metrics_by_corpus: Dict[str, Any] = {}
    emb_dir = out_dir / "embeddings"
    cache_dir = out_dir / "cache"
    text_col = str(data_cfg.get("text_col", "sentence"))
    log_phase("Phase 3/3 — Eval classification OOD", detail=", ".join(test_corpora))
    for idx, corpus_id in enumerate(test_corpora):
        try:
            spec = resolve_test_corpus(corpus_id, anchor=anchor)
            logger.info(
                "[macro_ft] Phase 3 — corpus %s (%s)",
                corpus_id,
                spec.display_name,
            )
            test_ds = TextRawDataset(
                str(spec.data_csv),
                label_col=label_col,
                pred_ok_col=str(data_cfg.get("pred_ok_col", "pred_ok")),
                group_col=str(data_cfg.get("group_col", "accident_id")),
                text_col=data_cfg.get("text_col"),
            )
            test_meta = test_ds.get_metadata_df()
            texts = test_meta[text_col].astype(str).tolist()
            n_test = len(texts)
            logger.info("[macro_ft] Phase 3 %s — %d exemples à traiter", corpus_id, n_test)

            if use_hidden_cache:
                logger.info(
                    "[macro_ft] Phase 3 %s — chargement h (CSV ou cache, pas de forward Qwen si CSV présent)",
                    corpus_id,
                )
                h_ood = load_backbone_hidden_for_corpus(
                    meta_df=test_meta,
                    texts=texts,
                    emb_csv=spec.emb_csv,
                    cache_path=cache_dir / f"backbone_hidden_{corpus_id}.npy",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    batch_size=batch_size,
                    device=device,
                )
                z_ood = encode_projected_matrix(
                    model,
                    h_ood,
                    batch_size=batch_size,
                    device=device,
                    show_progress=True,
                    progress_desc=f"export_{corpus_id}_z",
                )
                eval_loader = DataLoader(
                    BackboneHiddenDataset(h_ood, test_ds.label_ids),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_hidden_batch,
                )
            else:
                logger.info(
                    "[macro_ft] Phase 3 %s — encodage z puis eval (2 passes Qwen+ψ)",
                    corpus_id,
                )
                from supervised_macro_ft.inference import encode_texts

                z_ood = encode_texts(
                    model,
                    tokenizer,
                    texts,
                    max_length=max_length,
                    batch_size=batch_size,
                    device=device,
                    show_progress=True,
                    progress_desc=f"export_{corpus_id}_z",
                )
                eval_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
                )

            logger.info("[macro_ft] Phase 3 %s — export projected embeddings", corpus_id)
            export_projected_embeddings(
                z_ood,
                test_meta,
                emb_dir,
                str(corpus_id),
                label_col=label_col,
                group_col=str(data_cfg.get("group_col", "accident_id")),
                text_col=text_col,
            )
            corpus_metrics, corpus_details = evaluate_loader_with_predictions(
                model,
                eval_loader,
                device,
                show_progress=True,
                progress_desc=f"eval_{corpus_id}",
            )
            test_metrics_by_corpus[str(corpus_id)] = corpus_metrics
            build_and_save_predictions(
                test_meta,
                corpus_details,
                out_dir,
                str(corpus_id),
                method_name=method_name,
                text_col=text_col,
                group_col=str(data_cfg.get("group_col", "accident_id")),
                label_col=label_col,
                also_transfer_alias=(idx == len(test_corpora) - 1),
            )
            log_test_metrics(corpus_metrics, corpus=corpus_id)
        except Exception as exc:
            logger.warning("[macro_ft] Eval test corpus %s ignorée : %s", corpus_id, exc)

    logger.info("[macro_ft] Phase 3 — eval BTP in-domain (holdout final)")
    if use_hidden_cache and backbone_hidden is not None:
        btp_loader = DataLoader(
            BackboneHiddenDataset(backbone_hidden, dataset.label_ids),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_hidden_batch,
        )
    else:
        btp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    btp_metrics, btp_details = evaluate_loader_with_predictions(
        model,
        btp_loader,
        device,
        show_progress=True,
        progress_desc="eval_btp",
    )
    test_metrics_by_corpus["btp"] = btp_metrics
    btp_meta_for_preds = dataset.get_metadata_df()
    build_and_save_predictions(
        btp_meta_for_preds,
        btp_details,
        out_dir,
        "btp",
        method_name=method_name,
        text_col=text_col,
        group_col=str(data_cfg.get("group_col", "accident_id")),
        label_col=label_col,
    )

    cross_domain_summary = pd.DataFrame()
    if test_metrics_by_corpus:
        save_classification_outputs(
            out_dir,
            method_name=method_name,
            metrics_by_corpus=test_metrics_by_corpus,
            cv_summary=cv_summary,
        )
        cross_domain_summary = summarize_ood_classification(
            {k: v for k, v in test_metrics_by_corpus.items() if k != "btp"},
            cv_summary,
            model_name=method_name,
        )
        logger.info("[macro_ft] Synthèse cross-domain : %s", metrics_dir / "cross_domain_generalization.csv")

    result = {
        "method_name": method_name,
        "output_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),
        "cv_summary": cv_summary.to_dict(orient="records"),
        "final_fit_metrics": final_metrics,
        "test_metrics_by_corpus": test_metrics_by_corpus,
        "cross_domain_summary": cross_domain_summary.to_dict(orient="records"),
        "backbone_cache_used": bool(use_hidden_cache),
        "oversampling": bool(use_oversampling),
        "class_weight": class_weight_mode,
        "standardize_backbone": bool(model.backbone_scaler is not None),
        "train_history_paths": {
            "cv": str(cv_dir / "train_history.csv") if not cv_history.empty else None,
            "final": str(out_dir / "train_history_final.csv") if final_history else None,
        },
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log_run_complete(
        output_dir=str(out_dir),
        checkpoint_dir=str(ckpt_dir),
        summary_path=str(out_dir / "train_summary.json"),
    )
    return result
