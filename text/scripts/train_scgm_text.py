import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scgm_text.dataset_text_embeddings import (
    ID2LABEL,
    LABEL2ID,
    TextEmbeddingDataset,
    build_dataloaders,
    split_by_group,
)
from scgm_text.distillation import (
    build_teacher,
    snapshot_teacher_from_student,
    teacher_logits,
)
from scgm_text.fidelity import (
    apply_scgm_strict_defaults,
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
    pca_energy_c1_c10,
    q_assignment_distribution,
    rankme_effective_rank,
    subtype_alignment_diagnostics,
)
from scgm_text.optimizers import build_optimizer
from scgm_text.projection import normalize_projection_name
from scgm_text.schedulers import step_scheduler
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.sinkhorn_estep import sinkhorn_assign
from scgm_text.utils_io import ensure_dir, load_yaml_config, save_json, set_seed

METRIC_FIELDS = [
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
    "train_acc",
    "train_macro_f1",
    "train_nmi_subtype",
    "train_ari_subtype",
    "train_homogeneity_subtype",
    "train_purity_subtype",
    "train_entropy_pz",
    "train_entropy_py_z",
    "n_active_z",
    "z_usage_entropy",
    "sinkhorn_n_active_z",
    "sinkhorn_assignment_entropy",
    "val_acc",
    "val_macro_f1",
    "val_balanced_acc",
    "rankme_global",
    "c1_global",
    "c10_global",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SCGM-G on fixed text embeddings.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_csv", type=str, default="dataset/data_btp.csv")
    parser.add_argument("--emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_btp.csv")
    parser.add_argument("--output_dir", type=str, default="runs/scgm_text_qwen06")
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
    return parser.parse_args()


def apply_config(args: argparse.Namespace, config_path: Optional[str]) -> None:
    if not config_path:
        return
    raw = load_yaml_config(config_path)
    flat = flatten_config_yaml(raw) if any(k in raw for k in ("model", "training", "data")) else raw
    for key, value in flat.items():
        key_norm = key.replace("-", "_")
        if hasattr(args, key_norm):
            setattr(args, key_norm, value)


def finalize_args(args: argparse.Namespace) -> None:
    if args.scgm_strict_mode and args.text_pragmatic_mode:
        raise ValueError("Cannot use --scgm_strict_mode and --text_pragmatic_mode together.")
    if args.scgm_strict_mode:
        apply_scgm_strict_defaults(args)
    elif args.text_pragmatic_mode:
        apply_text_pragmatic_defaults(args)

    if args.beta1 is None:
        args.beta1 = args.beta
    if args.beta2 is None:
        args.beta2 = args.beta
    if args.beta3 is None:
        args.beta3 = args.beta

    if args.use_self_distillation and args.teacher_mode == "none":
        args.teacher_mode = "ema"

    if args.run_name:
        args.output_dir = os.path.join("runs", "scgm_text", args.run_name)

    if getattr(args, "with_mlp", None) is not None:
        args.projection = normalize_projection_name(None, args.with_mlp)
    else:
        args.projection = normalize_projection_name(args.projection, None)

    if not getattr(args, "fidelity_mode", None):
        args.fidelity_mode = "custom"


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
    model: SCGMEmbeddingNet,
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
        for embeddings, label_ids, selected_indices in train_loader:
            embeddings = embeddings.to(device)
            label_ids = label_ids.to(device)
            batch_y = labels_to_onehot(label_ids, n_class)
            features = model(embeddings)
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


def evaluate_split(
    model: SCGMEmbeddingNet,
    data_loader,
    device: torch.device,
    tau: float,
    n_class: int,
    prefix: str = "val",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    z_pred: List[int] = []
    embeddings: List[np.ndarray] = []
    prob_z_list: List[np.ndarray] = []
    prob_yz_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch_embeddings, label_ids, _ in data_loader:
            batch_embeddings = batch_embeddings.to(device)
            features = model(batch_embeddings)
            prob_y_x, prob_z_x, prob_y_z = model.pred(features, tau)
            preds = prob_y_x.argmax(dim=1).cpu().numpy()
            z_preds = prob_z_x.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            z_pred.extend(z_preds.tolist())
            y_true.extend(label_ids.numpy().tolist())
            embeddings.append(features.detach().cpu().numpy())
            prob_z_list.append(prob_z_x.cpu().numpy())
            prob_yz_list.append(prob_y_z.cpu().numpy())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    z_pred_arr = np.asarray(z_pred, dtype=np.int64)
    embedding_arr = np.concatenate(embeddings, axis=0)
    prob_z = np.concatenate(prob_z_list, axis=0)
    prob_yz = np.concatenate(prob_yz_list, axis=0)

    metrics = {
        f"{prefix}_acc": accuracy(y_true_arr, y_pred_arr),
        f"{prefix}_macro_f1": macro_f1(y_true_arr, y_pred_arr),
        f"{prefix}_balanced_acc": balanced_accuracy(y_true_arr, y_pred_arr),
        f"{prefix}_entropy_pz": mean_entropy(prob_z),
        f"{prefix}_entropy_py_z": mean_entropy(prob_yz),
        "n_active_z": float(count_active_clusters(z_pred_arr)),
    }
    c1, c10 = pca_energy_c1_c10(embedding_arr)
    metrics["rankme_global"] = rankme_effective_rank(embedding_arr)
    metrics["c1_global"] = c1
    metrics["c10_global"] = c10
    return metrics, y_true_arr, y_pred_arr, embedding_arr


def compute_train_subtype_metrics(
    model: SCGMEmbeddingNet,
    train_loader,
    dataset: TextEmbeddingDataset,
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
        for batch_embeddings, _, selected_indices in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            features = model(batch_embeddings)
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
    model: SCGMEmbeddingNet,
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
    model: SCGMEmbeddingNet,
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


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    print(describe_fidelity_mode(args), flush=True)

    dirs = create_run_dirs(args.output_dir)
    ensure_dir(args.output_dir)

    dataset = TextEmbeddingDataset(
        data_csv=args.data_csv,
        emb_csv=args.emb_csv,
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
    )
    train_idx, val_idx = split_by_group(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_loader, val_loader = build_dataloaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda":
            print("CUDA indisponible : entraînement sur CPU.", flush=True)
        device = torch.device("cpu")
    print(f"Device effectif: {device}", flush=True)

    if args.projection == "identity":
        args.hiddim = int(dataset.get_input_dim())

    model = SCGMEmbeddingNet(
        input_dim=dataset.get_input_dim(),
        hiddim=args.hiddim,
        num_classes=args.n_class,
        num_subclasses=args.n_subclass,
        projection=args.projection,
        kd_t=args.kd_t,
    ).to(device)

    ema_teacher = None
    if args.resume_from_checkpoint:
        teacher_sd = load_resume(args.resume_from_checkpoint, model, args)
        print(f"Resumed from {args.resume_from_checkpoint}", flush=True)
        if args.use_self_distillation:
            ema_teacher = build_teacher(model, args.teacher_mode, args.ema_decay)
            if teacher_sd and ema_teacher is not None:
                ema_teacher.load_state_dict(teacher_sd)

    optimizer = build_optimizer(model, args)

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
    config_payload["input_dim"] = dataset.get_input_dim()
    config_payload["train_idx"] = train_idx.tolist()
    config_payload["val_idx"] = val_idx.tolist()
    config_payload["label_distribution"] = dataset.get_label_distribution()
    save_json(config_payload, os.path.join(args.output_dir, "config.json"))
    save_json(LABEL2ID, os.path.join(args.output_dir, "label2id.json"))

    init_metrics_csv(dirs["train_log_csv"], METRIC_FIELDS)
    legacy_fields = [
        "epoch",
        "train_loss",
        "loss_macro",
        "loss_latent",
        "val_acc",
        "val_macro_f1",
        "val_balanced_acc",
        "rankme_global",
        "c1_global",
        "c10_global",
    ]

    best_f1 = -1.0
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

            for embeddings, label_ids, selected_indices in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                embeddings = embeddings.to(device)
                label_ids = label_ids.to(device)
                batch_y = labels_to_onehot(label_ids, args.n_class)
                local_indices = to_local_train_indices(selected_indices, train_loader, len(train_idx))
                batch_q = torch.tensor(q_new[local_indices], dtype=torch.float32, device=device)

                features = model(embeddings)
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
                model, train_loader, device, args.tau, args.n_class, prefix="train"
            )
            train_metrics.update(compute_train_subtype_metrics(model, train_loader, dataset, train_idx, device, args.tau))
            val_metrics, _, _, _ = evaluate_split(model, val_loader, device, args.tau, args.n_class, prefix="val")

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
            for key in METRIC_FIELDS:
                row.setdefault(key, float("nan"))

            with open(dirs["train_log_csv"], "a", newline="", encoding="utf-8") as mf:
                csv.DictWriter(mf, fieldnames=METRIC_FIELDS, extrasaction="ignore").writerow(row)
            append_jsonl(row, dirs["epoch_jsonl"])

            legacy_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "loss_macro": row["loss_macro"],
                    "loss_latent": row["loss_latent"],
                    "val_acc": row.get("val_acc"),
                    "val_macro_f1": row.get("val_macro_f1"),
                    "val_balanced_acc": row.get("val_balanced_acc"),
                    "rankme_global": row.get("rankme_global"),
                    "c1_global": row.get("c1_global"),
                    "c10_global": row.get("c10_global"),
                }
            )
            legacy_file.flush()

            print(
                f"Epoch {epoch}/{args.epochs} | lr={current_lr:.6f} | "
                f"loss={row['train_loss']:.4f} | ls1={row['ls1']:.4f} ls2={row['ls2']:.4f} ls3={row['ls3']:.4f} | "
                f"train_acc={row.get('train_acc', float('nan')):.4f} | "
                f"val_macro_f1={row.get('val_macro_f1', float('nan')):.4f}",
                flush=True,
            )

            if ema_teacher is not None and args.teacher_mode == "ema":
                ema_teacher.update(model)

            save_checkpoint(
                os.path.join(args.output_dir, "last_model.pt"),
                model,
                args,
                dataset.get_input_dim(),
                train_idx,
                val_idx,
                ema_teacher,
            )
            if val_metrics.get("val_macro_f1", -1.0) > best_f1:
                best_f1 = val_metrics["val_macro_f1"]
                save_checkpoint(
                    os.path.join(args.output_dir, "best_model.pt"),
                    model,
                    args,
                    dataset.get_input_dim(),
                    train_idx,
                    val_idx,
                    ema_teacher,
                )


def main() -> None:
    args = parse_args()
    cli_config = args.config
    apply_config(args, cli_config)
    finalize_args(args)
    if args.smoke_epochs is not None:
        args.epochs = args.smoke_epochs
    run_training(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
