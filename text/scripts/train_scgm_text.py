import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scgm_text.batch_utils import batch_to_device, forward_features, unpack_batch
from scgm_text.collate import make_text_collate_fn
from scgm_text.config_parsing import normalize_backbone_trainability
from scgm_text.data_metadata import ID2LABEL, LABEL2ID
from scgm_text.dataset_text_raw import TextRawDataset, build_text_dataloaders, split_by_group
from scgm_text.fidelity import (
    apply_config_to_args,
    apply_scgm_strict_defaults,
    describe_fidelity_mode,
    flatten_config_yaml,
)
from scgm_text.logging_utils import append_jsonl, create_run_dirs, init_metrics_csv
from scgm_text.metrics import (
    accuracy,
    balanced_accuracy,
    count_active_clusters,
    homogeneity_purity_safe,
    macro_f1,
    mean_entropy,
    q_assignment_distribution,
    subtype_alignment_diagnostics,
)
from scgm_text.optimizers import build_optimizer
from scgm_text.projection import normalize_projection_name
from scgm_text.schedulers import step_scheduler
from scgm_text.scgm_text_model import SCGMTextModel
from scgm_text.training_diagnostics import (
    assert_scgm_trainability,
    measure_backbone_weight_change,
    print_end2end_startup,
    print_grad_norms,
    print_trainable_parameters,
    snapshot_backbone_weights,
    verify_backbone_updated,
)
from scgm_text.sinkhorn_estep import sinkhorn_assign

SCGM_PROGRESS_EVERY_DEFAULT = 50


def _log_progress(tag: str, step: int, total: int, *, every: int) -> None:
    if every <= 0 or total <= 0:
        return
    if step == 1 or step == total or step % every == 0:
        pct = 100.0 * step / total
        print(f"[SCGM] {tag} {step}/{total} ({pct:.0f}%)", flush=True)


from metrics.geometry import (
    GEOMETRY_METRIC_KEYS,
    PRIMARY_SELECTION_METRIC,
    build_geometry_metrics_row,
)
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output, resolve_output_dir
from safer_core.seed import set_seed
from scgm_text.utils_io import ensure_dir, load_yaml_config, save_json

BASE_METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "ls1",
    "ls2",
    "ls3",
    "ls_div1",
    "ls_div2",
    "ls_div3",
    "loss_macro",
    "loss_latent",
    "lr",
    "optimizer",
    "scheduler",
    "projection",
    "fidelity_mode",
    "train_entropy_pz",
    "train_entropy_py_z",
    "n_active_z",
    "z_usage_entropy",
    "sinkhorn_n_active_z",
    "sinkhorn_assignment_entropy",
    "sinkhorn_converged",
    "sinkhorn_n_iter",
    "sinkhorn_err_mean_final",
    "sinkhorn_lmd_effective",
    "train_eta2_macro_balanced",
    "train_eta2_weighted",
    "val_eta2_macro_balanced",
    "val_eta2_weighted",
    "val_eta2_macro_balanced_perc",
]

CLASSIFIER_METRIC_FIELDS = [
    "train_acc",
    "train_macro_f1",
    "val_acc",
    "val_macro_f1",
    "val_balanced_acc",
]

SUBTYPE_METRIC_FIELDS = [
    "train_nmi_subtype",
    "train_ari_subtype",
    "train_homogeneity_subtype",
    "train_purity_subtype",
]


def build_metric_fields(args: argparse.Namespace) -> List[str]:
    fields = list(BASE_METRIC_FIELDS)
    if getattr(args, "compute_classifier_diagnostics", False):
        fields.extend(CLASSIFIER_METRIC_FIELDS)
    if getattr(args, "compute_subtype_diagnostics", False):
        fields.extend(SUBTYPE_METRIC_FIELDS)
    return fields


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SCGM-G end-to-end on raw text.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/methods/scgm_text.yaml",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace backbone (alias: backbone_model_name_or_path in checkpoint).",
    )
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"])
    parser.add_argument("--train_last_n_layers", type=int, default=None)
    parser.add_argument(
        "--backbone_trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, Qwen forward is no_grad; only projector + SCGM head train.",
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lr_backbone", type=float, default=None)
    parser.add_argument("--lr_projector", type=float, default=None)
    parser.add_argument("--lr_head", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--weight_decay_backbone", type=float, default=None)
    parser.add_argument("--weight_decay_projector", type=float, default=None)
    parser.add_argument("--weight_decay_head", type=float, default=None)
    parser.add_argument("--backbone_weight_decay", type=float, default=None)
    parser.add_argument("--head_weight_decay", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug_grad_norm", action="store_true")
    parser.add_argument("--verify_backbone_update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_csv", type=str, default="dataset/data_btp.csv")
    parser.add_argument("--output_dir", type=str, default="output/scgm_text")
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument("--pred_ok_col", type=str, default="pred_ok")
    parser.add_argument("--group_col", type=str, default="accident_id")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None, help="Alias legacy → lr_head si lr_head absent.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "cosine_warm_restarts"],
    )
    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--hiddim", type=int, default=128)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--n_subclass", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lmd", type=float, default=25.0)
    parser.add_argument("--lmd_start", type=float, default=None)
    parser.add_argument("--lmd_warmup_epochs", type=int, default=0)
    parser.add_argument("--n_iter_estep", type=int, default=5)
    parser.add_argument("--sinkhorn_max_iter", type=int, default=500)
    parser.add_argument("--sinkhorn_tol", type=float, default=1e-4)
    parser.add_argument(
        "--sinkhorn_tol_mode",
        type=str,
        default="mean",
        choices=["mean", "sum", "marginal_l1"],
    )
    parser.add_argument("--sinkhorn_check_every", type=int, default=10)
    parser.add_argument("--sinkhorn_eps", type=float, default=1e-12)
    parser.add_argument(
        "--sinkhorn_normalize_input",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sinkhorn_sample_prior",
        type=str,
        default="uniform",
        choices=["uniform", "macro_balanced"],
        help="Sinkhorn E-step sample marginal b: uniform (paper) or macro_balanced.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--projection",
        type=str,
        default="mlp",
        choices=["fc", "linear", "mlp"],
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--beta3", type=float, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=SCGM_PROGRESS_EVERY_DEFAULT,
        help="Log E-step/M-step/eval every N batches (0=disable intermediate logs).",
    )
    parser.add_argument("--smoke_epochs", type=int, default=None)
    parser.add_argument(
        "--best_checkpoint_metric",
        type=str,
        default="eta2_macro_balanced_perc",
        choices=["eta2_macro_balanced_perc", "eta2_macro_balanced"],
        help="Critère de sélection du best_model.pt (géométrie, pas F1).",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="Si >1, entraînement K-fold groupé (accident_id) avec kfold_summary.csv.",
    )
    parser.add_argument(
        "--final_fit_full_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Entraînement sur 100 %% BTP (pas de split val).",
    )
    parser.add_argument(
        "--test_corpus",
        type=str,
        default=None,
        help="Identifiant configs/test_corpora.yaml (prioritaire sur test_data_csv)",
    )
    parser.add_argument(
        "--test_data_csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--compute_classifier_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calcule accuracy / F1 (diagnostic secondaire, hors sélection).",
    )
    parser.add_argument(
        "--compute_subtype_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calcule NMI/ARI subtype sur le train (diagnostic secondaire).",
    )
    return parser.parse_args(argv)


def apply_config(args: argparse.Namespace, config_path: Optional[str]) -> None:
    if not config_path:
        return
    raw = load_yaml_config(config_path)
    flat = flatten_config_yaml(raw) if any(k in raw for k in ("model", "training", "data")) else raw
    apply_config_to_args(args, flat)


def finalize_args(args: argparse.Namespace) -> None:
    apply_scgm_strict_defaults(args)

    args.backbone_model_name_or_path = str(
        getattr(args, "backbone_model_name_or_path", None) or args.backbone_name
    )

    if args.lr_backbone is not None:
        args.backbone_lr = float(args.lr_backbone)
    if args.lr_head is not None:
        args.head_lr = float(args.lr_head)
    elif args.lr is not None:
        args.lr_head = float(args.lr)
        args.head_lr = float(args.lr)
    elif getattr(args, "head_lr", None) is None:
        args.head_lr = float(getattr(args, "lr_head", 1e-3))

    legacy_optimizer = str(getattr(args, "optimizer", "adamw")).strip().lower()
    if legacy_optimizer == "sgd":
        raise ValueError(
            "optimizer=sgd n'est plus supporté dans configs/methods/scgm_text.yaml. "
            "Utilisez optimizer: adamw."
        )
    if legacy_optimizer not in ("adamw", ""):
        raise ValueError(
            f"Optimiseur SCGM non supporté: {legacy_optimizer!r}. Seul adamw est disponible."
        )
    args.optimizer = "adamw"
    if args.weight_decay_backbone is not None:
        args.backbone_weight_decay = float(args.weight_decay_backbone)
    if args.weight_decay_projector is not None:
        args.head_weight_decay = float(args.weight_decay_projector)
    if args.weight_decay_head is not None:
        args.head_weight_decay = float(args.weight_decay_head)
    elif args.weight_decay is not None:
        wd = float(args.weight_decay)
        if args.backbone_weight_decay is None:
            args.backbone_weight_decay = wd
        if args.head_weight_decay is None:
            args.head_weight_decay = wd

    args.backbone_trainable, args.train_last_n_layers = normalize_backbone_trainability(
        bool(getattr(args, "backbone_trainable", True)),
        getattr(args, "train_last_n_layers", None),
    )
    args.effective_gradient_checkpointing = bool(
        getattr(args, "gradient_checkpointing", False)
    ) and bool(args.backbone_trainable)
    if getattr(args, "gradient_checkpointing", False) and not args.effective_gradient_checkpointing:
        print(
            "[SCGM] effective_gradient_checkpointing=false (backbone frozen)",
            flush=True,
        )

    if args.beta1 is None:
        args.beta1 = args.beta
    if args.beta2 is None:
        args.beta2 = args.beta
    if args.beta3 is None:
        args.beta3 = args.beta

    if args.run_name:
        args.output_dir = os.path.join("output", "scgm_text", args.run_name)
    args.output_dir = str(resolve_output_dir("scgm_text", args.output_dir))
    args.projection = normalize_projection_name(args.projection, None)

    _resolve_test_corpus_args(args)


def _resolve_test_corpus_args(args: argparse.Namespace) -> None:
    """Remplit test_data_csv depuis test_corpus ou le registre."""
    import os

    from safer_core.test_corpus import resolve_test_corpus

    corpus = getattr(args, "test_corpus", None) or os.environ.get("TEST_CORPUS")
    if corpus:
        spec = resolve_test_corpus(corpus)
        args.test_data_csv = str(spec.data_csv)
        return
    if not getattr(args, "test_data_csv", None):
        spec = resolve_test_corpus(None)
        args.test_data_csv = str(spec.data_csv)


def get_effective_lmd(
    epoch: int,
    lmd: float,
    lmd_start: Optional[float] = None,
    lmd_warmup_epochs: int = 0,
) -> float:
    """Linear warm-up: epoch 1 -> lmd_start, epoch warmup -> lmd (see plan convention)."""
    target = float(lmd)
    if lmd_start is None or int(lmd_warmup_epochs) <= 0:
        return target
    start = float(lmd_start)
    warmup = int(lmd_warmup_epochs)
    progress = min(1.0, max(0.0, (int(epoch) - 1) / max(1, warmup - 1)))
    return start + progress * (target - start)


def labels_to_onehot(label_ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    batch_size = label_ids.shape[0]
    onehot = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=label_ids.device)
    onehot.scatter_(1, label_ids.view(-1, 1), 1.0)
    return onehot


def initialize_q_new(num_samples: int, num_subclasses: int, label_ids: np.ndarray) -> np.ndarray:
    q_new = np.zeros((num_samples, num_subclasses), dtype=np.float32)
    rng = np.random.default_rng(0)
    for index, label_id in enumerate(label_ids):
        start = (label_id * num_subclasses) // 4
        end = ((label_id + 1) * num_subclasses) // 4
        component = rng.integers(start, max(start + 1, end))
        q_new[index, component] = 1.0
    return q_new


def to_local_train_indices(
    selected_indices: torch.Tensor,
    train_loader,
    n_train: int,
) -> np.ndarray:
    indices = selected_indices.detach().cpu().numpy()
    if indices.size == 0:
        return indices
    if indices.max() < n_train and indices.min() >= 0:
        return indices
    subset = train_loader.dataset
    if hasattr(subset, "indices"):
        index_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(subset.indices)}
        return np.array([index_map[int(idx)] for idx in indices], dtype=np.int64)
    raise IndexError(
        f"Invalid train indices: min={indices.min()}, max={indices.max()}, n_train={n_train}"
    )


def run_estep(
    model: SCGMTextModel,
    train_loader,
    device: torch.device,
    tau: float,
    n_class: int,
    n_train: int,
    n_subclass: int,
    lmd: float,
    train_labels: np.ndarray,
    sample_prior: str = "uniform",
    *,
    progress_every: int = SCGM_PROGRESS_EVERY_DEFAULT,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-4,
    sinkhorn_tol_mode: str = "mean",
    sinkhorn_check_every: int = 10,
    sinkhorn_eps: float = 1e-12,
    sinkhorn_normalize_input: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    score_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    n_batches = len(train_loader)
    print(
        f"[SCGM] E-step forward start (n_train={n_train}, batches={n_batches}, prior={sample_prior})",
        flush=True,
    )
    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            _, label_ids, selected_indices = unpack_batch(batch)
            label_ids = label_ids.to(device)
            batch_y = labels_to_onehot(label_ids, n_class)
            features = forward_features(model, batch)
            score_for_sinkhorn, _, _ = model.compute_latent_sinkhorn_scores(features, batch_y, tau)
            score_parts.append(score_for_sinkhorn.detach().cpu().numpy())
            index_parts.append(to_local_train_indices(selected_indices, train_loader, n_train))
            _log_progress("E-step forward", batch_i, n_batches, every=progress_every)

    score_tr = np.concatenate(score_parts, axis=0)
    batch_idx = np.concatenate(index_parts, axis=0)
    if len(batch_idx) != n_train:
        raise ValueError(f"E-step size mismatch: got {len(batch_idx)} rows, expected {n_train}")

    print(
        f"[SCGM] E-step Sinkhorn assign (n={score_tr.shape[0]}, r={score_tr.shape[1]}, lmd={lmd}) …",
        flush=True,
    )
    _, argmax_q, sink_diag = sinkhorn_assign(
        score_tr,
        lmd,
        labels=train_labels[batch_idx],
        n_classes=n_class,
        sample_prior=sample_prior,
        log_marginals=True,
        max_iter=sinkhorn_max_iter,
        tol=sinkhorn_tol,
        tol_mode=sinkhorn_tol_mode,
        check_every=sinkhorn_check_every,
        eps=sinkhorn_eps,
        normalize_input=sinkhorn_normalize_input,
        verbose=True,
    )
    print("[SCGM] E-step done.", flush=True)
    q_new = np.zeros((n_train, n_subclass), dtype=np.float32)
    q_new[batch_idx, argmax_q] = 1.0
    q_diag = q_assignment_distribution(q_new)
    sink_diag.update(q_diag)
    return q_new, sink_diag


def _geometry_keys_from_row(geom: Dict[str, Any]) -> Dict[str, float]:
    """Extrait les clés ``GEOMETRY_METRIC_KEYS`` pour agrégation K-fold / tuning."""
    out: Dict[str, float] = {}
    for key in GEOMETRY_METRIC_KEYS:
        val = geom.get(key, float("nan"))
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            out[key] = float("nan")
    return out


def checkpoint_selection_score(
    val_metrics: Dict[str, float],
    metric_name: str,
    lambda_c1: float = 0.0,
) -> float:
    if metric_name == "eta2_macro_balanced_perc":
        val = float(val_metrics.get("val_eta2_macro_balanced_perc", float("nan")))
        if np.isnan(val):
            val = float(val_metrics.get("val_delta_macro_pct", float("nan")))
        if np.isnan(val):
            val = 100.0 * float(val_metrics.get("val_eta2_macro_balanced", float("nan")))
        return val if np.isfinite(val) else float("-inf")
    eta2 = float(val_metrics.get("val_eta2_macro_balanced", float("nan")))
    if np.isnan(eta2):
        return float("-inf")
    return eta2


def evaluate_split(
    model: SCGMTextModel,
    data_loader,
    device: torch.device,
    tau: float,
    n_class: int,
    prefix: str = "val",
    *,
    compute_classifier_diagnostics: bool = False,
    progress_tag: Optional[str] = None,
    progress_every: int = SCGM_PROGRESS_EVERY_DEFAULT,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    z_pred: List[int] = []
    embeddings: List[np.ndarray] = []
    prob_z_list: List[np.ndarray] = []
    prob_yz_list: List[np.ndarray] = []
    n_batches = len(data_loader)
    if progress_tag and progress_every > 0:
        print(f"[SCGM] {progress_tag} start ({n_batches} batches)", flush=True)

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader, start=1):
            batch = batch_to_device(batch, device)
            _, label_ids, _ = unpack_batch(batch)
            features = forward_features(model, batch)
            prob_y_x, prob_z_x, prob_y_z = model.pred(features, tau)
            preds = prob_y_x.argmax(dim=1).cpu().numpy()
            z_preds = prob_z_x.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            z_pred.extend(z_preds.tolist())
            y_true.extend(label_ids.detach().cpu().numpy().tolist())
            embeddings.append(features.detach().cpu().numpy())
            prob_z_list.append(prob_z_x.cpu().numpy())
            prob_yz_list.append(prob_y_z.cpu().numpy())
            if progress_tag:
                _log_progress(progress_tag, batch_i, n_batches, every=progress_every)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    z_pred_arr = np.asarray(z_pred, dtype=np.int64)
    embedding_arr = np.concatenate(embeddings, axis=0)
    prob_z = np.concatenate(prob_z_list, axis=0)
    prob_yz = np.concatenate(prob_yz_list, axis=0)

    macro_labels = np.array([ID2LABEL[int(i)] for i in y_true_arr])
    geom = build_geometry_metrics_row(
        embedding_arr,
        macro_labels,
        method=f"{prefix}_scgm",
        l2_normalize=True,
    )
    metrics: Dict[str, float] = {
        f"{prefix}_eta2_macro_balanced": float(geom["eta2_macro_balanced"]),
        f"{prefix}_eta2_weighted": float(geom["eta2_weighted"]),
        f"{prefix}_eta2_macro_balanced_perc": float(geom["eta2_macro_balanced_perc"]),
        f"{prefix}_entropy_pz": mean_entropy(prob_z),
        f"{prefix}_entropy_py_z": mean_entropy(prob_yz),
        "n_active_z": float(count_active_clusters(z_pred_arr)),
    }
    if compute_classifier_diagnostics:
        metrics[f"{prefix}_acc"] = accuracy(y_true_arr, y_pred_arr)
        metrics[f"{prefix}_macro_f1"] = macro_f1(y_true_arr, y_pred_arr)
        metrics[f"{prefix}_balanced_acc"] = balanced_accuracy(y_true_arr, y_pred_arr)
    return metrics, y_true_arr, y_pred_arr, embedding_arr, geom


def compute_train_subtype_metrics(
    model: SCGMTextModel,
    train_loader,
    dataset,
    train_idx: np.ndarray,
    device: torch.device,
    tau: float,
) -> Dict[str, float]:
    if "pred_subtype" not in dataset.metadata_df.columns:
        return {}
    model.eval()
    z_all: List[int] = []
    subtypes: List[str] = []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            _, _, selected_indices = unpack_batch(batch)
            features = forward_features(model, batch)
            _, prob_z_x, _ = model.pred(features, tau)
            z_all.extend(prob_z_x.argmax(dim=1).cpu().tolist())
            for idx in selected_indices.cpu().numpy():
                subtypes.append(str(dataset.metadata_df.iloc[int(idx)]["pred_subtype"]))

    z_arr = np.asarray(z_all, dtype=np.int64)
    sub_arr = np.asarray(subtypes)
    out = subtype_alignment_diagnostics(z_arr, sub_arr)
    out.update(homogeneity_purity_safe(sub_arr, z_arr))
    return {
        "train_nmi_subtype": out.get("nmi_subtype", float("nan")),
        "train_ari_subtype": out.get("ari_subtype", float("nan")),
        "train_homogeneity_subtype": out.get("homogeneity_subtype", float("nan")),
        "train_purity_subtype": out.get("purity_subtype", float("nan")),
    }


def save_checkpoint(
    path: str,
    model: SCGMTextModel,
    args: argparse.Namespace,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    payload = {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "label2id": LABEL2ID,
        "hiddim": int(model.hiddim),
        "backbone_dim": int(model.input_dim),
            "train_idx": train_idx,
            "val_idx": val_idx,
        "pipeline": "end2end_text",
    }
    torch.save(payload, path)


def ensure_best_checkpoint_file(checkpoints_dir: str) -> None:
    """Garantit best_model.pt (copie depuis last_model.pt si la sélection val n'a rien écrit)."""
    import shutil

    best = os.path.join(checkpoints_dir, "best_model.pt")
    last = os.path.join(checkpoints_dir, "last_model.pt")
    if os.path.isfile(best):
        return
    if os.path.isfile(last):
        shutil.copy2(last, best)
        print("[checkpoint] best_model.pt créé depuis last_model.pt", flush=True)


def load_resume(
    path: str,
    model: SCGMTextModel,
    args: argparse.Namespace,
) -> Optional[Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    saved = ckpt.get("args", {})
    skip = {"config", "resume_from_checkpoint", "smoke_epochs"}
    for key, value in saved.items():
        if key in skip:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    if saved.get("input_mode") == "precomputed_embeddings":
        raise ValueError(
            "Checkpoint precomputed_embeddings is no longer supported. Retrain with end2end SCGM."
        )
    return None


def _smoke_backbone_step(
    model: SCGMTextModel,
    train_loader,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    model.train()
    batch = next(iter(train_loader))
    batch = batch_to_device(batch, device)
    _, label_ids, local_indices = unpack_batch(batch)
    label_ids = label_ids.to(device)
    batch_y = labels_to_onehot(label_ids, args.n_class)
    n_train = len(train_loader.dataset.indices) if hasattr(train_loader.dataset, "indices") else len(train_loader.dataset)
    local = to_local_train_indices(local_indices, train_loader, n_train)
    q_dummy = torch.zeros(len(local), args.n_subclass, device=device)
    q_dummy[torch.arange(len(local)), torch.randint(0, args.n_subclass, (len(local),), device=device)] = 1.0

    before = snapshot_backbone_weights(
        model, all_params=not bool(getattr(args, "backbone_trainable", True))
    )
    features = forward_features(model, batch)
    loss, *_ = model.loss(
        features, q_dummy, batch_y, args.tau, args.alpha,
        beta1=args.beta1, beta2=args.beta2, beta3=args.beta3,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    change = measure_backbone_weight_change(model, before)
    verify_backbone_updated(model, args, before, change)


def run_training(
    args: argparse.Namespace,
    *,
    train_idx_override: Optional[np.ndarray] = None,
    val_idx_override: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    t_run_start = time.perf_counter()
    set_seed(args.seed)
    print(describe_fidelity_mode(args), flush=True)

    layout = layout_method_output("scgm_text", args.output_dir)
    args.output_dir = str(layout["root"])
    dirs = create_run_dirs(args.output_dir)
    dirs["checkpoints_dir"] = str(layout["checkpoints"])
    dirs["logs_dir"] = str(layout["logs"])
    ensure_dir(layout["checkpoints"])
    ensure_dir(layout["logs"])

    from transformers import AutoTokenizer

    dataset = TextRawDataset(
        data_csv=args.data_csv,
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
        text_col=args.text_col,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone_model_name_or_path, trust_remote_code=True
    )
    collate_fn = make_text_collate_fn(tokenizer, args.max_seq_length)
    if train_idx_override is not None and val_idx_override is not None:
        train_idx, val_idx = train_idx_override, val_idx_override
    elif getattr(args, "final_fit_full_data", False):
        n = len(dataset)
        train_idx = np.arange(n, dtype=np.int64)
        val_idx = np.array([], dtype=np.int64)
    else:
        train_idx, val_idx = split_by_group(dataset, val_ratio=args.val_ratio, seed=args.seed)
    pin_memory = args.device == "cuda"
    train_loader, val_loader = build_text_dataloaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda":
            print("CUDA indisponible : entraînement sur CPU.", flush=True)
        device = torch.device("cpu")
    print(f"Device effectif: {device}", flush=True)

    model = SCGMTextModel.from_args(args).to(device)
    print_end2end_startup(model)

    if args.resume_from_checkpoint:
        load_resume(args.resume_from_checkpoint, model, args)
        print(f"Resumed from {args.resume_from_checkpoint}", flush=True)

    optimizer = build_optimizer(model, args)
    print_trainable_parameters(model)
    assert_scgm_trainability(model, optimizer, args.backbone_trainable)

    if args.verify_backbone_update:
        _smoke_backbone_step(model, train_loader, optimizer, device, args)

    grad_accum = max(1, int(args.gradient_accumulation_steps))
    debug_grad_done = False

    train_labels = dataset.metadata_df.iloc[train_idx]["label_id"].to_numpy(dtype=np.int64)
    q_new = initialize_q_new(len(train_idx), args.n_subclass, train_labels)
    sinkhorn_diag: Dict[str, float] = {}

    config_payload = vars(args).copy()
    config_payload["label2id"] = LABEL2ID
    config_payload["id2label"] = ID2LABEL
    config_payload["backbone_dim"] = int(model.input_dim)
    config_payload["hiddim"] = int(model.hiddim)
    config_payload["pipeline"] = "end2end_text"
    config_payload["train_idx"] = train_idx.tolist()
    config_payload["val_idx"] = val_idx.tolist()
    config_payload["label_distribution"] = dataset.get_label_distribution()
    save_config_resolved(config_payload, layout["root"])
    save_json(config_payload, layout["configs"] / "config.json")
    save_json(LABEL2ID, layout["configs"] / "label2id.json")

    metric_fields = build_metric_fields(args)
    init_metrics_csv(dirs["train_log_csv"], metric_fields)
    legacy_fields = [
                "epoch",
                "train_loss",
                "loss_macro",
                "loss_latent",
    "val_eta2_macro_balanced",
    "val_eta2_weighted",
    "val_eta2_macro_balanced_perc",
    ]
    if args.compute_classifier_diagnostics:
        legacy_fields.extend(["val_acc", "val_macro_f1", "val_balanced_acc"])

    best_score = float("-inf")
    best_epoch = 0
    best_geometry: Dict[str, float] = {}
    last_eval_geom: Dict[str, Any] = {}
    with open(dirs["legacy_logs_csv"], "w", newline="", encoding="utf-8") as legacy_file:
        legacy_writer = csv.DictWriter(legacy_file, fieldnames=legacy_fields)
        legacy_writer.writeheader()

        progress_every = int(getattr(args, "progress_every", SCGM_PROGRESS_EVERY_DEFAULT))
        for epoch in range(1, args.epochs + 1):
            current_lr = step_scheduler(optimizer, args, epoch, args.epochs)
            print(f"\n[SCGM] ===== Epoch {epoch}/{args.epochs} (lr={current_lr:.6f}) =====", flush=True)

            if epoch % args.n_iter_estep == 1:
                lmd_eff = get_effective_lmd(
                    epoch,
                    args.lmd,
                    getattr(args, "lmd_start", None),
                    getattr(args, "lmd_warmup_epochs", 0),
                )
                print(
                    f"[SCGM Sinkhorn] epoch={epoch} lmd_effective={lmd_eff:.4f} "
                    f"lmd_target={float(args.lmd):.4f}",
                    flush=True,
                )
                q_new, sinkhorn_diag = run_estep(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    tau=args.tau,
                    n_class=args.n_class,
                    n_train=len(train_idx),
                    n_subclass=args.n_subclass,
                    lmd=lmd_eff,
                    train_labels=train_labels,
                    sample_prior=args.sinkhorn_sample_prior,
                    progress_every=progress_every,
                    sinkhorn_max_iter=args.sinkhorn_max_iter,
                    sinkhorn_tol=args.sinkhorn_tol,
                    sinkhorn_tol_mode=args.sinkhorn_tol_mode,
                    sinkhorn_check_every=args.sinkhorn_check_every,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_normalize_input=args.sinkhorn_normalize_input,
                )

            model.train()
            totals = {k: 0.0 for k in ("loss", "ls1", "ls2", "ls3", "ls_div1", "ls_div2", "ls_div3", "macro", "latent")}
            num_batches = 0
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            n_train_batches = len(train_loader)
            if progress_every > 0:
                print(f"[SCGM] M-step train start ({n_train_batches} batches)", flush=True)

            for batch_i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1):
                batch = batch_to_device(batch, device)
                _, label_ids, selected_indices = unpack_batch(batch)
                label_ids = label_ids.to(device)
                batch_y = labels_to_onehot(label_ids, args.n_class)
                local_indices = to_local_train_indices(selected_indices, train_loader, len(train_idx))
                batch_q = torch.tensor(q_new[local_indices], dtype=torch.float32, device=device)

                features = forward_features(model, batch)
                loss, ls1, ls2, ls3, ls_div1, ls_div2, ls_div3 = model.loss(
                    features,
                    batch_q,
                    batch_y,
                    args.tau,
                    args.alpha,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    beta3=args.beta3,
                )
                (loss / grad_accum).backward()
                micro_step += 1

                if micro_step % grad_accum == 0:
                    if args.debug_grad_norm and not debug_grad_done:
                        print_grad_norms(
                            model, expect_backbone_trainable=args.backbone_trainable
                        )
                        debug_grad_done = True
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                totals["loss"] += float(loss.detach().cpu())
                totals["ls1"] += float(ls1.detach().cpu())
                totals["ls2"] += float(ls2.detach().cpu())
                totals["ls3"] += float(ls3.detach().cpu())
                totals["ls_div1"] += float(ls_div1.detach().cpu() if torch.is_tensor(ls_div1) else ls_div1)
                totals["ls_div2"] += float(ls_div2.detach().cpu() if torch.is_tensor(ls_div2) else ls_div2)
                totals["ls_div3"] += float(ls_div3.detach().cpu() if torch.is_tensor(ls_div3) else ls_div3)
                totals["macro"] += float(ls3.detach().cpu())
                totals["latent"] += float((ls1 + ls2).detach().cpu())
                num_batches += 1
                _log_progress(f"M-step epoch {epoch}", batch_i, n_train_batches, every=progress_every)

            if micro_step % grad_accum != 0:
                if args.debug_grad_norm and not debug_grad_done:
                    print_grad_norms(
                        model, expect_backbone_trainable=args.backbone_trainable
                    )
                    debug_grad_done = True
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            nb = max(num_batches, 1)
            if progress_every > 0:
                print("[SCGM] M-step train done, eval train …", flush=True)
            train_metrics, _, _, _, train_geom = evaluate_split(
                model,
                train_loader,
                device,
                args.tau,
                args.n_class,
                prefix="train",
                compute_classifier_diagnostics=args.compute_classifier_diagnostics,
                progress_tag=f"eval train epoch {epoch}",
                progress_every=progress_every,
            )
            if args.compute_subtype_diagnostics:
                train_metrics.update(
                    compute_train_subtype_metrics(
                        model, train_loader, dataset, train_idx, device, args.tau
                    )
                )
            has_val = len(val_idx) > 0
            if has_val:
                if progress_every > 0:
                    print("[SCGM] eval val …", flush=True)
                val_metrics, _, _, _, val_geom = evaluate_split(
                    model,
                    val_loader,
                    device,
                    args.tau,
                    args.n_class,
                    prefix="val",
                    compute_classifier_diagnostics=args.compute_classifier_diagnostics,
                    progress_tag=f"eval val epoch {epoch}",
                    progress_every=progress_every,
                )
                eval_geom = val_geom
            else:
                val_metrics = {
                    "val_eta2_macro_balanced": train_metrics.get("train_eta2_macro_balanced"),
                    "val_eta2_weighted": train_metrics.get("train_eta2_weighted"),
                    "val_eta2_macro_balanced_perc": train_metrics.get("train_eta2_macro_balanced_perc"),
                }
                eval_geom = train_geom
            last_eval_geom = eval_geom

            row: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": totals["loss"] / nb,
                "ls1": totals["ls1"] / nb,
                "ls2": totals["ls2"] / nb,
                "ls3": totals["ls3"] / nb,
                "ls_div1": totals["ls_div1"] / nb,
                "ls_div2": totals["ls_div2"] / nb,
                "ls_div3": totals["ls_div3"] / nb,
                "loss_macro": totals["macro"] / nb,
                "loss_latent": totals["latent"] / nb,
                "lr": current_lr,
                "optimizer": args.optimizer,
                "scheduler": args.scheduler,
                "projection": args.projection,
                "fidelity_mode": getattr(args, "fidelity_mode", "strict_fidelity"),
                **train_metrics,
                **val_metrics,
                **sinkhorn_diag,
            }
            for key in metric_fields:
                row.setdefault(key, float("nan"))

            with open(dirs["train_log_csv"], "a", newline="", encoding="utf-8") as mf:
                csv.DictWriter(mf, fieldnames=metric_fields, extrasaction="ignore").writerow(row)
            append_jsonl(row, dirs["epoch_jsonl"])

            legacy_row = {
                "epoch": epoch,
                "train_loss": row["train_loss"],
                "loss_macro": row["loss_macro"],
                "loss_latent": row["loss_latent"],
                "val_eta2_macro_balanced": row.get("val_eta2_macro_balanced"),
                "val_eta2_weighted": row.get("val_eta2_weighted"),
                "val_eta2_macro_balanced_perc": row.get("val_eta2_macro_balanced_perc"),
            }
            if args.compute_classifier_diagnostics:
                legacy_row["val_acc"] = row.get("val_acc")
                legacy_row["val_macro_f1"] = row.get("val_macro_f1")
                legacy_row["val_balanced_acc"] = row.get("val_balanced_acc")
            legacy_writer.writerow(legacy_row)
            legacy_file.flush()

            print(
                f"Epoch {epoch}/{args.epochs} | lr={current_lr:.6f} | "
                f"loss={row['train_loss']:.4f} | ls1={row['ls1']:.4f} ls2={row['ls2']:.4f} ls3={row['ls3']:.4f} | "
                f"val_eta2={row.get('val_eta2_macro_balanced', float('nan')):.4f}",
                flush=True,
            )

            save_checkpoint(
                os.path.join(dirs["checkpoints_dir"], "last_model.pt"),
                model,
                args,
                train_idx,
                val_idx,
            )
            score = checkpoint_selection_score(
                val_metrics,
                args.best_checkpoint_metric,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_geometry = _geometry_keys_from_row(eval_geom)
                save_checkpoint(
                    os.path.join(dirs["checkpoints_dir"], "best_model.pt"),
                    model,
                    args,
                    train_idx,
                    val_idx,
                )

    if not best_geometry and last_eval_geom:
        best_geometry = _geometry_keys_from_row(last_eval_geom)

    ensure_best_checkpoint_file(dirs["checkpoints_dir"])

    config_payload["best_checkpoint_metric"] = args.best_checkpoint_metric
    config_payload["best_checkpoint_score"] = best_score
    config_payload["best_checkpoint_epoch"] = best_epoch
    save_config_resolved(config_payload, layout["root"])
    save_json(config_payload, layout["configs"] / "config.json")
    selection_score = best_geometry.get(PRIMARY_SELECTION_METRIC, float("nan"))
    if not np.isfinite(selection_score):
        selection_score = best_score if np.isfinite(best_score) else float("nan")
    return {
        **best_geometry,
        "best_checkpoint_score": best_score,
        "best_checkpoint_epoch": best_epoch,
        "selection_score": float(selection_score),
        "train_wall_time_sec": float(time.perf_counter() - t_run_start),
    }


def run_kfold(args: argparse.Namespace) -> None:
    from contrastive_methods.eval_geometry import compute_fold_ipr
    from safer_core.kfold_eval import group_kfold_splits, save_kfold_tables

    dataset = TextRawDataset(
        data_csv=args.data_csv,
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
        text_col=args.text_col,
    )
    groups = dataset.metadata_df[args.group_col].to_numpy()
    splits = group_kfold_splits(groups, args.kfold, args.seed)
    fold_rows: List[Dict[str, Any]] = []
    base_out = args.output_dir
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        fold_args = argparse.Namespace(**vars(args))
        fold_args.output_dir = os.path.join(base_out, "folds", f"fold_{fold_id}")
        print(f"[kfold] fold {fold_id} → {fold_args.output_dir}", flush=True)
        metrics = run_training(fold_args, train_idx_override=train_idx, val_idx_override=val_idx)
        val_meta = dataset.metadata_df.iloc[val_idx]
        method_geom = {k: metrics[k] for k in GEOMETRY_METRIC_KEYS if k in metrics}
        ipr = compute_fold_ipr(val_meta, args.label_col, method_geom)
        fold_rows.append({"fold_id": fold_id, **metrics, **ipr})
    layout = layout_method_output("scgm_text", base_out)
    save_kfold_tables(fold_rows, layout["metrics"])
    print(f"[kfold] Résumé val → {layout['metrics'] / 'kfold_summary.csv'}", flush=True)


def run_post_train_eval(args: argparse.Namespace) -> None:
    """Évalue BTP + test avec best_model.pt et exporte projections pour le notebook."""
    from scgm_text.eval_corpus import evaluate_and_save_btp_test

    layout = layout_method_output("scgm_text", args.output_dir)
    ckpt = layout["checkpoints"] / "best_model.pt"
    if not ckpt.is_file():
        print(f"[eval] Checkpoint absent : {ckpt}", flush=True)
        return
    paths = evaluate_and_save_btp_test(
        checkpoint_path=str(ckpt),
        output_root=str(layout["root"]),
        data_btp=args.data_csv,
        data_test=args.test_data_csv,
        test_corpus_id=getattr(args, "test_corpus", None),
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
        save_projections=True,
    )
    if paths.get("projections_test"):
        print(f"[eval] Projections test : {paths['projections_test']}", flush=True)
    elif paths.get("test"):
        print("[eval] Métriques test OK mais projections .npy non écrites (voir messages ci-dessus).", flush=True)


def main() -> None:
    args = parse_args()
    cli_config = args.config
    apply_config(args, cli_config)
    finalize_args(args)
    if args.smoke_epochs is not None:
        args.epochs = args.smoke_epochs
    if args.kfold and args.kfold > 1:
        run_kfold(args)
        final_args = argparse.Namespace(**vars(args))
        final_args.final_fit_full_data = True
        final_args.kfold = 0
        layout = layout_method_output("scgm_text", final_args.output_dir)
        final_args.output_dir = str(layout["root"])
        print("[scgm] Réentraînement final 100 % BTP…", flush=True)
        from safer_core.kfold_eval import record_final_fit_wall_time

        t_final = time.perf_counter()
        run_training(final_args)
        record_final_fit_wall_time(layout["metrics"], time.perf_counter() - t_final)
        run_post_train_eval(final_args)
    else:
        run_training(args)
        run_post_train_eval(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
