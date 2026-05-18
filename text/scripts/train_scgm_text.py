import argparse
import csv
import os
import sys
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
from scgm_text.dataset_text_embeddings import (
    ID2LABEL,
    LABEL2ID,
    TextEmbeddingDataset,
    build_dataloaders,
    split_by_group,
)
from scgm_text.dataset_text_raw import TextRawDataset, build_text_dataloaders
from scgm_text.distillation import (
    build_teacher,
    snapshot_teacher_from_student,
    teacher_logits,
)
from scgm_text.fidelity import (
    apply_precomputed_identity_defaults,
    apply_scgm_strict_defaults,
    apply_strict_finetune_identity_defaults,
    apply_text_pragmatic_defaults,
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
    assert_backbone_trainable_when_identity_text,
    measure_backbone_weight_change,
    print_trainable_parameters,
    snapshot_backbone_weights,
    verify_backbone_updated,
    warn_identity_frozen_backbone,
)
from scgm_text.sinkhorn_estep import sinkhorn_assign
from metrics.geometry import build_geometry_metrics_row
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
    "use_self_distillation",
    "train_entropy_pz",
    "train_entropy_py_z",
    "n_active_z",
    "z_usage_entropy",
    "sinkhorn_n_active_z",
    "sinkhorn_assignment_entropy",
    "train_eta2_macro_balanced",
    "train_eta2_weighted",
    "val_eta2_macro_balanced",
    "val_eta2_weighted",
    "val_delta_macro_pct",
    "rankme_global",
    "c1_global",
    "c10_global",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SCGM-G on text or precomputed embeddings.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--input_mode",
        type=str,
        default="precomputed_embeddings",
        choices=["text", "precomputed_embeddings"],
    )
    parser.add_argument(
        "--backbone_model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"])
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_last_n_layers", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--backbone_weight_decay", type=float, default=0.01)
    parser.add_argument("--head_weight_decay", type=float, default=None)
    parser.add_argument("--strict_finetune_identity", action="store_true")
    parser.add_argument("--precomputed_identity", action="store_true")
    parser.add_argument("--verify_backbone_update", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_csv", type=str, default="dataset/data_btp.csv")
    parser.add_argument("--emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_btp.csv")
    parser.add_argument("--output_dir", type=str, default="resultats/scgm_text")
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument("--pred_ok_col", type=str, default="pred_ok")
    parser.add_argument("--group_col", type=str, default="accident_id")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--hiddim", type=int, default=128)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--n_subclass", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lmd", type=float, default=25.0)
    parser.add_argument("--n_iter_estep", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--projection",
        type=str,
        default="identity",
        choices=["identity", "linear", "mlp"],
    )
    parser.add_argument("--with_mlp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scgm_strict_mode", action="store_true")
    parser.add_argument("--text_pragmatic_mode", action="store_true")
    parser.add_argument("--use_self_distillation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kd_t", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--beta3", type=float, default=None)
    parser.add_argument(
        "--teacher_mode",
        type=str,
        default="none",
        choices=["none", "ema", "previous_epoch"],
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--smoke_epochs", type=int, default=None)
    parser.add_argument(
        "--best_checkpoint_metric",
        type=str,
        default="delta_macro_pct",
        choices=["delta_macro_pct", "eta2_macro_balanced", "composite"],
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
        "--test_data_csv",
        type=str,
        default="dataset/test/data_metallurgie.csv",
    )
    parser.add_argument(
        "--test_emb_csv",
        type=str,
        default="embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv",
    )
    parser.add_argument(
        "--best_checkpoint_lambda",
        type=float,
        default=0.01,
        help="λ pour composite = eta2_macro_balanced - λ * c1_global.",
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
    return parser.parse_args()


def apply_config(args: argparse.Namespace, config_path: Optional[str]) -> None:
    if not config_path:
        return
    raw = load_yaml_config(config_path)
    flat = flatten_config_yaml(raw) if any(k in raw for k in ("model", "training", "data")) else raw
    for key, value in flat.items():
        key_norm = key.replace("-", "_")
        if key_norm == "n_folds":
            args.kfold = int(value)
            continue
        if hasattr(args, key_norm):
            setattr(args, key_norm, value)


def finalize_args(args: argparse.Namespace) -> None:
    preset_flags = sum(
        bool(x)
        for x in (
            args.scgm_strict_mode,
            args.text_pragmatic_mode,
            args.strict_finetune_identity,
            args.precomputed_identity,
        )
    )
    if preset_flags > 1:
        raise ValueError("Un seul preset d'entraînement à la fois.")
    if args.strict_finetune_identity:
        apply_strict_finetune_identity_defaults(args)
    elif args.precomputed_identity:
        apply_precomputed_identity_defaults(args)
    elif args.scgm_strict_mode:
        apply_scgm_strict_defaults(args)
    elif args.text_pragmatic_mode:
        apply_text_pragmatic_defaults(args)

    if args.head_lr is None:
        args.head_lr = float(args.lr)
    if args.head_weight_decay is None:
        args.head_weight_decay = float(args.weight_decay)

    if args.beta1 is None:
        args.beta1 = args.beta
    if args.beta2 is None:
        args.beta2 = args.beta
    if args.beta3 is None:
        args.beta3 = args.beta

    if args.use_self_distillation and args.teacher_mode == "none":
        args.teacher_mode = "ema"

    if args.run_name:
        args.output_dir = os.path.join("resultats", "scgm_text", args.run_name)
    args.output_dir = str(resolve_output_dir("scgm_text", args.output_dir))

    if getattr(args, "with_mlp", None) is not None:
        args.projection = normalize_projection_name(None, args.with_mlp)
    else:
        args.projection = normalize_projection_name(args.projection, None)

    if not getattr(args, "fidelity_mode", None):
        args.fidelity_mode = "custom"

    if args.input_mode == "text" and not args.backbone_model_name_or_path:
        raise ValueError("input_mode=text exige --backbone_model_name_or_path.")

    if args.verify_backbone_update is None:
        args.verify_backbone_update = (
            args.input_mode == "text"
            and args.projection == "identity"
            and not args.freeze_backbone
        )

    warn_identity_frozen_backbone(args)


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
) -> Tuple[np.ndarray, Dict[str, float]]:
    score_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            _, label_ids, selected_indices = unpack_batch(batch)
            label_ids = label_ids.to(device)
            batch_y = labels_to_onehot(label_ids, n_class)
            features = forward_features(model, batch)
            score_for_sinkhorn, _, _ = model.compute_latent_sinkhorn_scores(features, batch_y, tau)
            score_parts.append(score_for_sinkhorn.detach().cpu().numpy())
            index_parts.append(to_local_train_indices(selected_indices, train_loader, n_train))

    score_tr = np.concatenate(score_parts, axis=0)
    batch_idx = np.concatenate(index_parts, axis=0)
    if len(batch_idx) != n_train:
        raise ValueError(f"E-step size mismatch: got {len(batch_idx)} rows, expected {n_train}")

    _, argmax_q, sink_diag = sinkhorn_assign(score_tr, lmd)
    q_new = np.zeros((n_train, n_subclass), dtype=np.float32)
    q_new[batch_idx, argmax_q] = 1.0
    q_diag = q_assignment_distribution(q_new)
    sink_diag.update(q_diag)
    return q_new, sink_diag


def checkpoint_selection_score(
    val_metrics: Dict[str, float],
    metric_name: str,
    lambda_c1: float,
) -> float:
    if metric_name == "delta_macro_pct":
        val = float(val_metrics.get("val_delta_macro_pct", float("nan")))
        if np.isnan(val):
            val = 100.0 * float(val_metrics.get("val_eta2_macro_balanced", float("nan")))
        return val if np.isfinite(val) else float("-inf")
    eta2 = float(val_metrics.get("val_eta2_macro_balanced", float("nan")))
    if np.isnan(eta2):
        eta2 = float("-inf")
    if metric_name == "composite":
        c1 = float(val_metrics.get("c1_global", 0.0))
        if np.isnan(c1):
            c1 = 0.0
        return eta2 - float(lambda_c1) * c1
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
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    z_pred: List[int] = []
    embeddings: List[np.ndarray] = []
    prob_z_list: List[np.ndarray] = []
    prob_yz_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
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
        f"{prefix}_delta_macro_pct": float(geom["delta_macro_pct"]),
        f"{prefix}_entropy_pz": mean_entropy(prob_z),
        f"{prefix}_entropy_py_z": mean_entropy(prob_yz),
        "n_active_z": float(count_active_clusters(z_pred_arr)),
        "rankme_global": float(geom["rankme_global"]),
        "c1_global": float(geom["c1_global"]),
        "c10_global": float(geom["c10_global"]),
    }
    if compute_classifier_diagnostics:
        metrics[f"{prefix}_acc"] = accuracy(y_true_arr, y_pred_arr)
        metrics[f"{prefix}_macro_f1"] = macro_f1(y_true_arr, y_pred_arr)
        metrics[f"{prefix}_balanced_acc"] = balanced_accuracy(y_true_arr, y_pred_arr)
    return metrics, y_true_arr, y_pred_arr, embedding_arr


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
    input_dim: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ema_teacher=None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "label2id": LABEL2ID,
        "input_dim": input_dim,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    if ema_teacher is not None:
        payload["teacher_state_dict"] = ema_teacher.state_dict()
    torch.save(payload, path)


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
    skip = {"scgm_strict_mode", "text_pragmatic_mode", "config", "resume_from_checkpoint", "smoke_epochs"}
    for key, value in saved.items():
        if key in skip:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    return ckpt.get("teacher_state_dict")


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

    before = snapshot_backbone_weights(model)
    features = forward_features(model, batch)
    loss, *_ = model.loss(features, q_dummy, batch_y, args.tau, args.alpha, kd_t=args.kd_t)
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
    set_seed(args.seed)
    print(describe_fidelity_mode(args), flush=True)
    print(
        f"input_mode={args.input_mode} projection={args.projection} "
        f"freeze_backbone={args.freeze_backbone}",
        flush=True,
    )

    layout = layout_method_output("scgm_text", args.output_dir)
    args.output_dir = str(layout["root"])
    dirs = create_run_dirs(args.output_dir)
    dirs["checkpoints_dir"] = str(layout["checkpoints"])
    dirs["logs_dir"] = str(layout["logs"])
    ensure_dir(layout["checkpoints"])
    ensure_dir(layout["logs"])

    collate_fn = None
    if args.input_mode == "text":
        from transformers import AutoTokenizer

        dataset = TextRawDataset(
            data_csv=args.data_csv,
            label_col=args.label_col,
            pred_ok_col=args.pred_ok_col,
            group_col=args.group_col,
            text_col=args.text_col,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_name_or_path)
        collate_fn = make_text_collate_fn(tokenizer, args.max_seq_length)
        if train_idx_override is not None and val_idx_override is not None:
            train_idx, val_idx = train_idx_override, val_idx_override
        elif getattr(args, "final_fit_full_data", False):
            n = len(dataset)
            train_idx = np.arange(n, dtype=np.int64)
            val_idx = np.array([], dtype=np.int64)
        else:
            train_idx, val_idx = split_by_group(dataset, val_ratio=args.val_ratio, seed=args.seed)
        train_loader, val_loader = build_text_dataloaders(
            dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        input_dim = 0
    else:
        dataset = TextEmbeddingDataset(
            data_csv=args.data_csv,
            emb_csv=args.emb_csv,
            label_col=args.label_col,
            pred_ok_col=args.pred_ok_col,
            group_col=args.group_col,
        )
        if train_idx_override is not None and val_idx_override is not None:
            train_idx, val_idx = train_idx_override, val_idx_override
        elif getattr(args, "final_fit_full_data", False):
            n = len(dataset)
            train_idx = np.arange(n, dtype=np.int64)
            val_idx = np.array([], dtype=np.int64)
        else:
            train_idx, val_idx = split_by_group(dataset, val_ratio=args.val_ratio, seed=args.seed)
        train_loader, val_loader = build_dataloaders(
            dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        input_dim = int(dataset.get_input_dim())
        if args.projection == "identity":
            args.hiddim = input_dim

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda":
            print("CUDA indisponible : entraînement sur CPU.", flush=True)
        device = torch.device("cpu")
    print(f"Device effectif: {device}", flush=True)

    model = SCGMTextModel.from_args(args, input_dim=input_dim).to(device)
    if args.input_mode == "text":
        input_dim = int(model.hiddim)
        args.hiddim = input_dim

    ema_teacher = None
    if args.resume_from_checkpoint:
        teacher_sd = load_resume(args.resume_from_checkpoint, model, args)
        print(f"Resumed from {args.resume_from_checkpoint}", flush=True)
        if args.use_self_distillation:
            ema_teacher = build_teacher(model, args.teacher_mode, args.ema_decay)
            if teacher_sd and ema_teacher is not None:
                ema_teacher.load_state_dict(teacher_sd)

    optimizer = build_optimizer(model, args)
    print_trainable_parameters(model)
    assert_backbone_trainable_when_identity_text(model, args, optimizer)

    if args.verify_backbone_update and args.input_mode == "text":
        _smoke_backbone_step(model, train_loader, optimizer, device, args)

    if args.use_self_distillation and ema_teacher is None:
        ema_teacher = build_teacher(model, args.teacher_mode, args.ema_decay)
        if args.teacher_mode == "previous_epoch":
            snapshot_teacher_from_student(ema_teacher, model)

    train_labels = dataset.metadata_df.iloc[train_idx]["label_id"].to_numpy(dtype=np.int64)
    q_new = initialize_q_new(len(train_idx), args.n_subclass, train_labels)
    sinkhorn_diag: Dict[str, float] = {}

    config_payload = vars(args).copy()
    config_payload["label2id"] = LABEL2ID
    config_payload["id2label"] = ID2LABEL
    config_payload["input_dim"] = input_dim
    config_payload["hiddim"] = int(model.hiddim)
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
    "val_delta_macro_pct",
    "rankme_global",
        "c1_global",
        "c10_global",
    ]
    if args.compute_classifier_diagnostics:
        legacy_fields.extend(["val_acc", "val_macro_f1", "val_balanced_acc"])

    best_score = float("-inf")
    best_epoch = 0
    with open(dirs["legacy_logs_csv"], "w", newline="", encoding="utf-8") as legacy_file:
        legacy_writer = csv.DictWriter(legacy_file, fieldnames=legacy_fields)
        legacy_writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            if args.teacher_mode == "previous_epoch" and ema_teacher is not None:
                snapshot_teacher_from_student(ema_teacher, model)

            current_lr = step_scheduler(optimizer, args, epoch, args.epochs)

            if epoch % args.n_iter_estep == 1:
                q_new, sinkhorn_diag = run_estep(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    tau=args.tau,
                    n_class=args.n_class,
                    n_train=len(train_idx),
                    n_subclass=args.n_subclass,
                    lmd=args.lmd,
                )

            model.train()
            totals = {k: 0.0 for k in ("loss", "ls1", "ls2", "ls3", "ls_div1", "ls_div2", "ls_div3", "macro", "latent")}
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                batch = batch_to_device(batch, device)
                _, label_ids, selected_indices = unpack_batch(batch)
                label_ids = label_ids.to(device)
                batch_y = labels_to_onehot(label_ids, args.n_class)
                local_indices = to_local_train_indices(selected_indices, train_loader, len(train_idx))
                batch_q = torch.tensor(q_new[local_indices], dtype=torch.float32, device=device)

                features = forward_features(model, batch)
                logit_t1 = logit_t2 = logit_t3 = None
                if args.use_self_distillation and ema_teacher is not None:
                    logit_t1, logit_t2, logit_t3 = teacher_logits(
                        ema_teacher.teacher, features, batch_y, args.tau
                    )

                loss, ls1, ls2, ls3, ls_div1, ls_div2, ls_div3 = model.loss(
                    features,
                    batch_q,
                    batch_y,
                    args.tau,
                    args.alpha,
                    logit_t1=logit_t1,
                    logit_t2=logit_t2,
                    logit_t3=logit_t3,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    beta3=args.beta3,
                    kd_t=args.kd_t,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

            nb = max(num_batches, 1)
            train_metrics, _, _, _ = evaluate_split(
                model,
                train_loader,
                device,
                args.tau,
                args.n_class,
                prefix="train",
                compute_classifier_diagnostics=args.compute_classifier_diagnostics,
            )
            if args.compute_subtype_diagnostics:
                train_metrics.update(
                    compute_train_subtype_metrics(
                        model, train_loader, dataset, train_idx, device, args.tau
                    )
                )
            has_val = len(val_idx) > 0
            if has_val:
                val_metrics, _, _, _ = evaluate_split(
                    model,
                    val_loader,
                    device,
                    args.tau,
                    args.n_class,
                    prefix="val",
                    compute_classifier_diagnostics=args.compute_classifier_diagnostics,
                )
            else:
                val_metrics = {
                    "val_eta2_macro_balanced": train_metrics.get("train_eta2_macro_balanced"),
                    "val_eta2_weighted": train_metrics.get("train_eta2_weighted"),
                    "val_delta_macro_pct": train_metrics.get("train_delta_macro_pct"),
                    "rankme_global": train_metrics.get("rankme_global"),
                    "c1_global": train_metrics.get("c1_global"),
                    "c10_global": train_metrics.get("c10_global"),
                }

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
                "fidelity_mode": getattr(args, "fidelity_mode", "custom"),
                "use_self_distillation": bool(args.use_self_distillation),
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
                "val_delta_macro_pct": row.get("val_delta_macro_pct"),
                "rankme_global": row.get("rankme_global"),
                "c1_global": row.get("c1_global"),
                "c10_global": row.get("c10_global"),
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
                f"val_eta2={row.get('val_eta2_macro_balanced', float('nan')):.4f} | "
                f"rankme={row.get('rankme_global', float('nan')):.2f}",
                flush=True,
            )

            if ema_teacher is not None and args.teacher_mode == "ema":
                ema_teacher.update(model)

            save_checkpoint(
                os.path.join(dirs["checkpoints_dir"], "last_model.pt"),
                model,
                args,
                input_dim,
                train_idx,
                val_idx,
                ema_teacher,
            )
            score = checkpoint_selection_score(
                val_metrics,
                args.best_checkpoint_metric,
                args.best_checkpoint_lambda,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(
                    os.path.join(dirs["checkpoints_dir"], "best_model.pt"),
                    model,
                    args,
                    input_dim,
                    train_idx,
                    val_idx,
                    ema_teacher,
                )

    config_payload["best_checkpoint_metric"] = args.best_checkpoint_metric
    config_payload["best_checkpoint_lambda"] = args.best_checkpoint_lambda
    config_payload["best_checkpoint_score"] = best_score
    config_payload["best_checkpoint_epoch"] = best_epoch
    save_config_resolved(config_payload, layout["root"])
    save_json(config_payload, layout["configs"] / "config.json")
    return {
        "delta_macro_pct": best_score if args.best_checkpoint_metric == "delta_macro_pct" else float("nan"),
        "val_eta2_macro_balanced": float(config_payload.get("val_eta2_macro_balanced", float("nan"))),
        "best_checkpoint_score": best_score,
        "best_checkpoint_epoch": best_epoch,
    }


def run_kfold(args: argparse.Namespace) -> None:
    from safer_core.kfold_eval import group_kfold_splits, save_kfold_tables

    if args.input_mode == "text":
        dataset = TextRawDataset(
            data_csv=args.data_csv,
            label_col=args.label_col,
            pred_ok_col=args.pred_ok_col,
            group_col=args.group_col,
            text_col=args.text_col,
        )
    else:
        dataset = TextEmbeddingDataset(
            data_csv=args.data_csv,
            emb_csv=args.emb_csv,
            label_col=args.label_col,
            pred_ok_col=args.pred_ok_col,
            group_col=args.group_col,
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
        fold_rows.append({"fold_id": fold_id, **metrics})
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
        emb_btp=args.emb_csv,
        data_test=getattr(args, "test_data_csv", "dataset/test/data_metallurgie.csv"),
        emb_test=getattr(args, "test_emb_csv", "embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv"),
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
        run_training(final_args)
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
