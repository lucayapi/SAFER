"""Baseline macro transfer: Frozen Source Prototypes (no adaptation)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from macro_transfer.constants import LABEL2ID, MACRO_NAMES
from macro_transfer.encode import load_target_metadata
from macro_transfer.tpn_full_encoder import FullEncoderTPNModel, encode_texts_corpus
from safer_core.io import load_yaml
from safer_core.paths import resolve_repo_path
from safer_core.test_corpus import default_test_corpus_id, resolve_test_paths_from_config
from scgm_text.dataset_text_embeddings import load_filtered_metadata

logger = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"l2_normalize attend un array 2D, reçu shape={arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def compute_source_prototypes(
    embeddings: np.ndarray,
    labels: Sequence[str],
    macros: Sequence[str],
    normalize: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    z = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=object)
    if z.ndim != 2:
        raise ValueError(f"embeddings source doit être 2D, reçu {z.shape}")
    if len(z) != len(y):
        raise ValueError(f"Mismatch source embeddings/labels: {len(z)} vs {len(y)}")

    protos = np.zeros((len(macros), z.shape[1]), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for i, macro in enumerate(macros):
        mask = y.astype(str) == str(macro)
        n_src = int(mask.sum())
        if n_src <= 0:
            raise ValueError(f"Aucun exemple source pour macro {macro!r}.")
        p = z[mask].mean(axis=0)
        if normalize:
            denom = np.linalg.norm(p)
            if denom <= 1e-12:
                raise ValueError(f"Prototype source nul pour macro {macro!r}.")
            p = p / denom
        protos[i] = p
        rows.append(
            {
                "macro": str(macro),
                "n_source": n_src,
                "prototype_norm": float(np.linalg.norm(p)),
            }
        )
    return protos, pd.DataFrame(rows)


def pairwise_distances_to_prototypes(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    metric: str,
) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    metric_n = str(metric).strip().lower()
    if metric_n == "cosine":
        sim = z @ p.T
        return 1.0 - sim
    if metric_n == "sqeuclidean":
        diff = z[:, None, :] - p[None, :, :]
        return np.sum(diff * diff, axis=2)
    raise ValueError(f"distance_metric non supporté: {metric!r}")


def softmax_over_negative_distances(
    distances: np.ndarray,
    tau: float,
    eps: float = 1e-12,
) -> np.ndarray:
    if tau <= 0:
        raise ValueError(f"tau doit être > 0, reçu {tau}")
    d = np.asarray(distances, dtype=np.float64)
    logits = -d / float(tau)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), eps, None)


def assign_macros_from_source_prototypes(
    target_embeddings: np.ndarray,
    prototypes: np.ndarray,
    macros: Sequence[str],
    tau: float,
    metric: str,
    eps: float = 1e-12,
) -> dict[str, Any]:
    distances = pairwise_distances_to_prototypes(target_embeddings, prototypes, metric=metric)
    probs = softmax_over_negative_distances(distances, tau=tau, eps=eps)
    top = probs.argmax(axis=1)
    pred_macro = np.array([str(macros[i]) for i in top], dtype=object)
    confidence = probs.max(axis=1)
    sort_p = np.sort(probs, axis=1)
    margin = sort_p[:, -1] - sort_p[:, -2] if probs.shape[1] >= 2 else np.zeros(len(probs))
    entropy = -(probs * np.log(np.clip(probs, eps, None))).sum(axis=1)
    return {
        "pred_macro": pred_macro,
        "probs": probs,
        "distances": distances,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
    }


def evaluate_macro_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probs: np.ndarray,
    macros: Sequence[str],
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=object).astype(str)
    y_pred_arr = np.asarray(y_pred, dtype=object).astype(str)
    y_true_id = np.array([LABEL2ID.get(v, -1) for v in y_true_arr], dtype=np.int64)
    y_pred_id = np.array([LABEL2ID.get(v, -1) for v in y_pred_arr], dtype=np.int64)
    mask = (y_true_id >= 0) & (y_pred_id >= 0)
    if not bool(mask.any()):
        return {
            "n_eval": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "mean_confidence": float("nan"),
            "mean_margin": float("nan"),
            "mean_entropy": float("nan"),
            "classification_report": {},
            "confusion_matrix": np.zeros((len(macros), len(macros)), dtype=np.int64),
        }

    yt = y_true_id[mask]
    yp = y_pred_id[mask]
    p = np.asarray(probs, dtype=np.float64)[mask]
    p_sorted = np.sort(p, axis=1)
    metrics = {
        "n_eval": int(len(yt)),
        "accuracy": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "mean_confidence": float(p.max(axis=1).mean()),
        "mean_margin": float((p_sorted[:, -1] - p_sorted[:, -2]).mean()) if p.shape[1] >= 2 else 0.0,
        "mean_entropy": float((-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)).mean()),
        "classification_report": classification_report(
            yt,
            yp,
            labels=list(range(len(macros))),
            target_names=list(macros),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(len(macros)))),
    }
    return metrics


def _check_required_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [c for c in required if c and c not in df.columns]
    if missing:
        raise ValueError(f"{context}: colonnes manquantes {missing}")


def _validate_no_nan_inf(name: str, x: np.ndarray) -> None:
    arr = np.asarray(x)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contient NaN/Inf")


def _resolve_output_dir(cfg: dict[str, Any], corpus: str, anchor: Path) -> Path:
    out_cfg = cfg.get("output_dir")
    if out_cfg:
        return resolve_repo_path(str(out_cfg), repo_root=anchor)
    return resolve_repo_path(
        Path("output_test") / corpus / "macro_transfer" / "frozen_source_prototypes",
        repo_root=anchor,
    )


def run_frozen_source_prototypes(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path)
    cfg = load_yaml(cfg_path)
    anchor = Path(__file__).resolve().parents[1]

    corpus = str(cfg.get("corpus") or default_test_corpus_id())
    source_cfg = dict(cfg.get("source") or {})
    target_cfg = dict(cfg.get("target") or {})
    model_cfg = dict(cfg.get("model") or {})
    tr_cfg = dict(cfg.get("prototype_transfer") or {})
    exp_cfg = dict(cfg.get("exports") or {})

    macros = [str(m) for m in tr_cfg.get("macros", list(MACRO_NAMES))]
    if macros != list(MACRO_NAMES):
        raise ValueError(
            f"Ordre macros invalide. Attendu {list(MACRO_NAMES)}, reçu {macros}."
        )
    metric = str(tr_cfg.get("distance_metric", "cosine")).lower()
    normalize_embeddings = bool(tr_cfg.get("normalize_embeddings", True))
    tau = float(tr_cfg.get("tau", 0.07))
    eps = float(tr_cfg.get("eps", 1e-12))
    if tau <= 0:
        raise ValueError(f"tau doit être > 0, reçu {tau}")

    source_data_csv = resolve_repo_path(source_cfg.get("dataset_path", "dataset/data_btp.csv"), repo_root=anchor)
    _spec, target_csv_auto, _emb = resolve_test_paths_from_config(
        {"corpus": corpus, "target": {}},
        corpus_id=corpus,
        anchor=anchor,
    )
    target_data_csv = resolve_repo_path(target_cfg.get("dataset_path", str(target_csv_auto)), repo_root=anchor)

    source_text_col = str(source_cfg.get("text_col", "sentence"))
    source_label_col = str(source_cfg.get("label_col", "pred_label"))
    source_pred_ok_col = str(source_cfg.get("pred_ok_col", "pred_ok"))
    source_group_col = str(source_cfg.get("group_col", "accident_id"))
    filter_pred_ok = bool(source_cfg.get("filter_pred_ok", True))

    target_text_col = str(target_cfg.get("text_col", "sentence"))
    target_label_col = target_cfg.get("label_col", None)
    target_group_col = str(target_cfg.get("group_col", "accident_id"))

    source_df = load_filtered_metadata(
        str(source_data_csv),
        label_col=source_label_col,
        pred_ok_col=source_pred_ok_col,
        group_col=source_group_col,
        text_col=source_text_col,
    ) if filter_pred_ok else pd.read_csv(source_data_csv)
    target_df = load_target_metadata(str(target_data_csv), text_col=target_text_col)

    _check_required_columns(source_df, [source_text_col, source_label_col], "source")
    _check_required_columns(target_df, [target_text_col], "target")
    if target_group_col and target_group_col not in target_df.columns:
        target_df[target_group_col] = np.arange(len(target_df))

    source_df = source_df[source_df[source_text_col].astype(str).str.strip().ne("")].reset_index(drop=True)
    target_df = target_df[target_df[target_text_col].astype(str).str.strip().ne("")].reset_index(drop=True)

    checkpoint_cfg = model_cfg.get("checkpoint_path")
    checkpoint = (
        str(resolve_repo_path(checkpoint_cfg, repo_root=anchor))
        if checkpoint_cfg
        else None
    )
    base_method = str(model_cfg.get("base_method", "scgm_text"))
    method_display_name = str(
        cfg.get("method_display_name")
        or model_cfg.get("method_display_name")
        or ("SCGM + prototypes source gelés" if base_method == "scgm_text" else f"{base_method} + prototypes source gelés")
    )
    backbone_name = str(model_cfg.get("backbone_name", "Qwen/Qwen3-Embedding-0.6B"))
    max_seq_length = int(model_cfg.get("max_seq_length", 256))
    batch_size = int(model_cfg.get("encode_batch_size", 32))
    device = str(model_cfg.get("device", "cuda"))

    model = FullEncoderTPNModel(
        base_method=base_method,
        checkpoint=checkpoint,
        backbone_name=backbone_name,
        max_seq_length=max_seq_length,
        pooling="mean",
        freeze_backbone=True,
        device=device,
    )
    model.eval()

    z_source = encode_texts_corpus(
        model,
        source_df[source_text_col].astype(str).tolist(),
        batch_size=batch_size,
        log_label="source_frozen",
    )
    z_target = encode_texts_corpus(
        model,
        target_df[target_text_col].astype(str).tolist(),
        batch_size=batch_size,
        log_label="target_frozen",
    )
    if normalize_embeddings:
        z_source = l2_normalize(z_source, eps=eps)
        z_target = l2_normalize(z_target, eps=eps)

    _validate_no_nan_inf("z_source", z_source)
    _validate_no_nan_inf("z_target", z_target)
    prototypes, proto_df = compute_source_prototypes(
        z_source,
        source_df[source_label_col].astype(str).to_numpy(),
        macros,
        normalize=normalize_embeddings if metric == "cosine" else False,
    )
    _validate_no_nan_inf("prototypes", prototypes)

    assign = assign_macros_from_source_prototypes(
        z_target,
        prototypes,
        macros=macros,
        tau=tau,
        metric=metric,
        eps=eps,
    )
    probs = np.asarray(assign["probs"], dtype=np.float64)
    dists = np.asarray(assign["distances"], dtype=np.float64)
    _validate_no_nan_inf("distances", dists)
    _validate_no_nan_inf("probs", probs)
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6):
        raise ValueError("Certaines lignes de probs ne somment pas à 1.")

    out_dir = _resolve_output_dir(cfg, corpus, anchor)
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir = out_dir / "transfer"
    emb_dir = out_dir / "embeddings"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.DataFrame(
        {
            "method": method_display_name,
            "pred_macro": assign["pred_macro"],
            "confidence": assign["confidence"],
            "margin": assign["margin"],
            "entropy": assign["entropy"],
        }
    )
    if "index" in target_df.columns:
        preds["index"] = target_df["index"].to_numpy()
    if target_group_col in target_df.columns:
        preds[target_group_col] = target_df[target_group_col].to_numpy()
    if "fact_id" in target_df.columns:
        preds["fact_id"] = target_df["fact_id"].to_numpy()
    preds["sentence"] = target_df[target_text_col].astype(str).to_numpy()
    for i, m in enumerate(macros):
        preds[f"prob_{m}"] = probs[:, i]
        preds[f"dist_{m}"] = dists[:, i]
    if target_label_col and target_label_col in target_df.columns:
        preds["true_macro"] = target_df[target_label_col].astype(str).to_numpy()
    preds.to_csv(transfer_dir / "target_macro_predictions.csv", index=False)

    if exp_cfg.get("save_prototypes", True):
        proto_export = proto_df.copy()
        for j in range(prototypes.shape[1]):
            proto_export[f"dim_{j:04d}"] = prototypes[:, j]
        proto_export.to_csv(transfer_dir / "source_prototypes.csv", index=False)

    metrics_out: dict[str, Any] = {
        "method": method_display_name,
        "n_source": int(len(source_df)),
        "n_target": int(len(target_df)),
        "distance_metric": metric,
        "tau": tau,
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "mean_confidence": float(np.mean(assign["confidence"])) if len(assign["confidence"]) else float("nan"),
        "mean_entropy": float(np.mean(assign["entropy"])) if len(assign["entropy"]) else float("nan"),
    }
    if target_label_col and target_label_col in target_df.columns:
        eval_metrics = evaluate_macro_predictions(
            target_df[target_label_col].astype(str).to_numpy(),
            assign["pred_macro"],
            probs,
            macros,
        )
        cm = np.asarray(eval_metrics.pop("confusion_matrix"))
        cls_rep = eval_metrics.pop("classification_report")
        metrics_out.update(eval_metrics)
        pd.DataFrame(cm, index=macros, columns=macros).to_csv(transfer_dir / "confusion_matrix.csv")
        pd.DataFrame(cls_rep).T.to_csv(transfer_dir / "classification_report.csv", index=True)
    else:
        pd.DataFrame(
            np.zeros((len(macros), len(macros)), dtype=np.int64),
            index=macros,
            columns=macros,
        ).to_csv(transfer_dir / "confusion_matrix.csv")

    with open(transfer_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    bertopic_all_cols = ["pred_macro", "confidence"] + [f"prob_{m}" for m in macros]
    bertopic_df = preds[[c for c in [target_group_col, "fact_id", "sentence"] if c in preds.columns] + bertopic_all_cols]
    bertopic_df.to_csv(transfer_dir / "bertopic_input_all.csv", index=False)
    for m in macros:
        bertopic_df[bertopic_df["pred_macro"] == m].to_csv(transfer_dir / f"bertopic_input_{m}.csv", index=False)

    if exp_cfg.get("save_target_embeddings", True):
        np.save(emb_dir / "target_embeddings.npy", z_target)
        target_df[[c for c in [target_group_col, "fact_id", target_text_col] if c in target_df.columns]].to_csv(
            emb_dir / "target_embeddings_metadata.csv",
            index=False,
        )
    if exp_cfg.get("save_source_embeddings", False):
        np.save(emb_dir / "source_embeddings.npy", z_source)

    if normalize_embeddings:
        source_norm_mean = float(np.linalg.norm(z_source, axis=1).mean())
        target_norm_mean = float(np.linalg.norm(z_target, axis=1).mean())
        logger.info("Normes moyennes (normalize_embeddings=true): source=%.4f target=%.4f", source_norm_mean, target_norm_mean)
        if not (0.9 <= source_norm_mean <= 1.1 and 0.9 <= target_norm_mean <= 1.1):
            logger.warning("Normes moyennes éloignées de 1.0 (source=%.4f, target=%.4f)", source_norm_mean, target_norm_mean)

    return {
        "output_dir": str(out_dir),
        "transfer_dir": str(transfer_dir),
        "metrics_path": str(transfer_dir / "metrics.json"),
        "predictions_path": str(transfer_dir / "target_macro_predictions.csv"),
    }

