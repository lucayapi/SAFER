"""Logs structurés pour le pipeline supervised_macro_ft."""

from __future__ import annotations

import logging
import sys
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd
import torch

from supervised_macro_ft.embedding_cache import should_cache_backbone_embeddings
from supervised_macro_ft.model import model_kwargs_from_cfg, validate_macro_ft_projection

logger = logging.getLogger(__name__)

_PREFIX = "[macro_ft]"


def log_step_start(
    step: str,
    *,
    n_samples: int,
    batch_size: int,
    detail: Optional[str] = None,
) -> None:
    """Annonce le début d'une passe batchée (encodage, eval, export)."""
    n_batches = max((n_samples + batch_size - 1) // batch_size, 1) if batch_size > 0 else 1
    msg = f"{_PREFIX} {step} — {n_samples} exemples, batch_size={batch_size} (~{n_batches} batches)"
    if detail:
        msg += f" | {detail}"
    logger.info(msg)


def batched_progress(
    loader: Iterable[Any],
    *,
    desc: str,
    total: Optional[int] = None,
    show_progress: bool = True,
) -> Iterator[Any]:
    """Itère sur un DataLoader avec barre tqdm (stdout, mininterval=10s) si activé."""
    if not show_progress:
        yield from loader
        return
    from tqdm import tqdm

    n_total = total if total is not None else (len(loader) if hasattr(loader, "__len__") else None)
    yield from tqdm(
        loader,
        total=n_total,
        desc=desc,
        unit="batch",
        file=sys.stdout,
        mininterval=10.0,
        dynamic_ncols=True,
    )


def log_step_done(step: str, *, n_samples: int, detail: Optional[str] = None) -> None:
    msg = f"{_PREFIX} {step} terminé — {n_samples} exemples"
    if detail:
        msg += f" | {detail}"
    logger.info(msg)


def log_phase(title: str, *, detail: Optional[str] = None) -> None:
    if detail:
        logger.info("%s === %s — %s ===", _PREFIX, title, detail)
    else:
        logger.info("%s === %s ===", _PREFIX, title)


def _backbone_mode_label(model_cfg: Mapping[str, Any]) -> str:
    backbone_trainable = bool(model_cfg.get("backbone_trainable", True))
    train_last_n = model_cfg.get("train_last_n_layers")
    if not backbone_trainable:
        return "gelé + cache embeddings"
    if train_last_n is not None:
        try:
            n = int(train_last_n)
            if n > 0:
                return f"entraînable partiel (train_last_n_layers={n})"
        except (TypeError, ValueError):
            pass
    return "entraînable complet"


def _training_path_label(
    model_cfg: Mapping[str, Any],
    *,
    backbone_hidden_available: bool,
) -> str:
    if backbone_hidden_available and should_cache_backbone_embeddings(model_cfg):
        return "embeddings cachés (projecteur + tête)"
    return "texte → Qwen forward à chaque batch"


def resolve_effective_use_amp(
    train_cfg: Mapping[str, Any],
    *,
    backbone_trainable: bool,
    device: torch.device,
) -> tuple[bool, str]:
    if "use_amp" in train_cfg:
        enabled = bool(train_cfg.get("use_amp"))
        return enabled, "config training.use_amp"
    if backbone_trainable and device.type == "cuda":
        return True, "auto (backbone trainable + CUDA)"
    return False, "désactivé (backbone gelé ou CPU)"


def log_effective_config(
    model_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    device: torch.device,
    use_oversampling: bool,
    class_weight_mode: Optional[str],
    n_samples: Optional[int] = None,
    backbone_hidden_available: bool = False,
) -> None:
    """Récapitulatif de la config effective au démarrage."""
    projection = validate_macro_ft_projection(model_cfg.get("projection", "mlp_sklearn"))
    mk = model_kwargs_from_cfg(model_cfg)
    use_amp, amp_reason = resolve_effective_use_amp(
        train_cfg,
        backbone_trainable=bool(model_cfg.get("backbone_trainable", True)),
        device=device,
    )
    lines = [
        f"device={device}",
        f"projection={projection}, hiddim={mk.get('hiddim')}",
        f"backbone={model_cfg.get('backbone_name')} | mode={_backbone_mode_label(model_cfg)}",
        f"chemin entraînement={_training_path_label(model_cfg, backbone_hidden_available=backbone_hidden_available)}",
        f"gradient_checkpointing={mk.get('gradient_checkpointing')}",
        f"use_amp={use_amp} ({amp_reason})",
        f"oversampling={use_oversampling}, class_weight={class_weight_mode or 'none'}",
        f"n_folds={train_cfg.get('n_folds', 3)}, epochs={train_cfg.get('epochs', 30)}",
        f"selection_metric={train_cfg.get('selection_metric', 'balanced_accuracy')}",
        f"batch_size={train_cfg.get('batch_size', 32)}, lr_projector={train_cfg.get('lr_projector', 1e-3)}, "
        f"lr_head={train_cfg.get('lr_head', 1e-3)}, lr_backbone={train_cfg.get('lr_backbone', 2e-5)}",
        f"early_stopping_patience={train_cfg.get('early_stopping_patience', 2)}",
    ]
    if n_samples is not None:
        lines.append(f"n_samples={n_samples}")
    logger.info("%s Config effective :", _PREFIX)
    for line in lines:
        logger.info("%s   %s", _PREFIX, line)


def log_trainable_param_counts(
    *,
    n_backbone: int,
    n_projector: int,
    n_head: int,
) -> None:
    logger.info(
        "%s Paramètres trainables : backbone=%d, projecteur=%d, tête=%d",
        _PREFIX,
        n_backbone,
        n_projector,
        n_head,
    )


def log_cv_fold_start(
    fold_id: int,
    n_folds: int,
    *,
    n_train: int,
    n_val: int,
    oversampled: bool,
    use_hidden_cache: bool,
) -> None:
    extra = []
    if oversampled:
        extra.append("oversampled")
    if not use_hidden_cache:
        extra.append("forward texte")
    suffix = f", {', '.join(extra)}" if extra else ""
    logger.info(
        "%s CV fold %d/%d démarré (train=%d, val=%d%s)",
        _PREFIX,
        fold_id + 1,
        n_folds,
        n_train,
        n_val,
        suffix,
    )


def log_cv_fold_done(
    fold_id: int,
    n_folds: int,
    metrics: Mapping[str, Any],
    *,
    selection_metric: str,
) -> None:
    sel = str(selection_metric).strip().lower()
    score = metrics.get(sel, metrics.get(f"val_{sel}", metrics.get("balanced_accuracy")))
    logger.info(
        "%s CV fold %d/%d terminé — accuracy=%.4f macro_f1=%.4f balanced_accuracy=%.4f %s=%.4f epoch=%s",
        _PREFIX,
        fold_id + 1,
        n_folds,
        float(metrics.get("accuracy", float("nan"))),
        float(metrics.get("macro_f1", float("nan"))),
        float(metrics.get("balanced_accuracy", float("nan"))),
        selection_metric,
        float(score) if score is not None else float("nan"),
        metrics.get("epoch", "?"),
    )


def log_cv_summary(
    cv_summary: pd.DataFrame,
    *,
    selection_metric: str,
) -> None:
    if cv_summary.empty:
        logger.warning("%s CV terminée — résumé vide", _PREFIX)
        return
    row = cv_summary.iloc[0]
    logger.info("%s CV terminée — résumé agrégé (%d folds) :", _PREFIX, int(row.get("n_folds", 0)))
    for key in ("mean_accuracy", "mean_macro_f1", "mean_balanced_accuracy", "mean_loss"):
        if key in row:
            std_key = key.replace("mean_", "std_")
            std_val = row.get(std_key)
            if std_val is not None and pd.notna(std_val):
                logger.info(
                    "%s   %s=%.4f ± %.4f",
                    _PREFIX,
                    key,
                    float(row[key]),
                    float(std_val),
                )
            else:
                logger.info("%s   %s=%.4f", _PREFIX, key, float(row[key]))
    col = f"mean_{selection_metric}" if not str(selection_metric).startswith("mean_") else selection_metric
    if col in row:
        logger.info("%s   sélection (%s)=%.4f", _PREFIX, selection_metric, float(row[col]))


def log_fit_start(
    run_label: str,
    *,
    epochs: int,
    selection_metric: str,
    use_amp: bool,
    n_train: Optional[int] = None,
) -> None:
    msg = (
        f"{_PREFIX} [{run_label}] démarrage — epochs={epochs}, "
        f"selection_metric={selection_metric}, use_amp={use_amp}"
    )
    if n_train is not None:
        msg += f", n_train={n_train}"
    logger.info(msg)


def log_early_stop(run_label: str, *, epoch: int, patience: int) -> None:
    logger.info(
        "%s [%s] early stopping à epoch=%d (patience=%d)",
        _PREFIX,
        run_label,
        epoch,
        patience,
    )


def log_test_metrics(test_metrics: Mapping[str, Any], *, corpus: str) -> None:
    if not test_metrics:
        logger.warning("%s Eval test (%s) — aucune métrique", _PREFIX, corpus)
        return
    logger.info(
        "%s Eval test (%s) — accuracy=%.4f macro_f1=%.4f balanced_accuracy=%.4f loss=%.4f",
        _PREFIX,
        corpus,
        float(test_metrics.get("accuracy", float("nan"))),
        float(test_metrics.get("macro_f1", float("nan"))),
        float(test_metrics.get("balanced_accuracy", float("nan"))),
        float(test_metrics.get("loss", float("nan"))),
    )


def log_run_complete(
    *,
    output_dir: str,
    checkpoint_dir: str,
    summary_path: Optional[str] = None,
) -> None:
    logger.info("%s === Entraînement terminé ===", _PREFIX)
    logger.info("%s   output_dir=%s", _PREFIX, output_dir)
    logger.info("%s   checkpoint=%s", _PREFIX, checkpoint_dir)
    if summary_path:
        logger.info("%s   train_summary=%s", _PREFIX, summary_path)
