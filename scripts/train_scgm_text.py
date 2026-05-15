import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

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
from scgm_text.metrics import (
    accuracy,
    balanced_accuracy,
    c1_c10_by_macro,
    compute_confusion_matrix,
    macro_f1,
    pca_energy_c1_c10,
    rankme_by_macro,
    rankme_effective_rank,
)
from scgm_text.projection import normalize_projection_name
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.utils_io import ensure_dir, load_yaml_config, save_json, set_seed
from sinkhornknopp import optimize_l_sk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SCGM-G on fixed text embeddings.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_csv", type=str, default="dataset/data_btp.csv")
    parser.add_argument("--emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_btp.csv")
    parser.add_argument("--output_dir", type=str, default="runs/scgm_text_qwen06")
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument("--pred_ok_col", type=str, default="pred_ok")
    parser.add_argument("--group_col", type=str, default="accident_id")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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
        help="identity = backbone natif (hiddim forcé à la dimension des embeddings).",
    )
    parser.add_argument(
        "--with_mlp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Déprécié : équivalent --projection mlp ou linear selon le booléen.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--smoke_epochs", type=int, default=None, help="Override epochs for quick smoke tests.")
    return parser.parse_args()


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
) -> np.ndarray:
    prob_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for embeddings, label_ids, selected_indices in train_loader:
            embeddings = embeddings.to(device)
            label_ids = label_ids.to(device)
            batch_y = labels_to_onehot(label_ids, n_class)
            features = model(embeddings)
            batch_prob_y_x, _, _ = model.forward_to_prob(features, batch_y, tau)
            prob_parts.append(batch_prob_y_x.detach().cpu().numpy())
            index_parts.append(to_local_train_indices(selected_indices, train_loader, n_train))

    prob_tr = np.concatenate(prob_parts, axis=0)
    batch_idx = np.concatenate(index_parts, axis=0)
    if len(batch_idx) != n_train:
        raise ValueError(f"E-step size mismatch: got {len(batch_idx)} rows, expected {n_train}")
    _, argmax_q = optimize_l_sk(prob_tr, lmd)
    q_new = np.zeros((n_train, n_subclass), dtype=np.float32)
    q_new[batch_idx, argmax_q] = 1.0
    return q_new


def evaluate_split(
    model: SCGMEmbeddingNet,
    data_loader,
    device: torch.device,
    tau: float,
    n_class: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for batch_embeddings, label_ids, _ in data_loader:
            batch_embeddings = batch_embeddings.to(device)
            features = model(batch_embeddings)
            prob_y_x, _, _ = model.pred(features, tau)
            preds = prob_y_x.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(label_ids.numpy().tolist())
            embeddings.append(features.detach().cpu().numpy())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    embedding_arr = np.concatenate(embeddings, axis=0)
    metrics = {
        "val_acc": accuracy(y_true_arr, y_pred_arr),
        "val_macro_f1": macro_f1(y_true_arr, y_pred_arr),
        "val_balanced_acc": balanced_accuracy(y_true_arr, y_pred_arr),
    }
    c1, c10 = pca_energy_c1_c10(embedding_arr)
    metrics["rankme_global"] = rankme_effective_rank(embedding_arr)
    metrics["c1_global"] = c1
    metrics["c10_global"] = c10
    return metrics, y_true_arr, y_pred_arr, embedding_arr


def save_checkpoint(
    path: str,
    model: SCGMEmbeddingNet,
    args: argparse.Namespace,
    input_dim: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "label2id": LABEL2ID,
            "input_dim": input_dim,
            "train_idx": train_idx,
            "val_idx": val_idx,
        },
        path,
    )


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
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
    if getattr(args, "with_mlp", None) is not None:
        args.projection = normalize_projection_name(None, args.with_mlp)
    else:
        args.projection = normalize_projection_name(args.projection, None)
    if args.projection == "identity":
        args.hiddim = int(dataset.get_input_dim())
    model = SCGMEmbeddingNet(
        input_dim=dataset.get_input_dim(),
        hiddim=args.hiddim,
        num_classes=args.n_class,
        num_subclasses=args.n_subclass,
        projection=args.projection,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_labels = dataset.metadata_df.iloc[train_idx]["label_id"].to_numpy(dtype=np.int64)
    q_new = initialize_q_new(len(train_idx), args.n_subclass, train_labels)

    config_payload = vars(args).copy()
    config_payload["label2id"] = LABEL2ID
    config_payload["id2label"] = ID2LABEL
    config_payload["input_dim"] = dataset.get_input_dim()
    config_payload["train_idx"] = train_idx.tolist()
    config_payload["val_idx"] = val_idx.tolist()
    config_payload["label_distribution"] = dataset.get_label_distribution()
    save_json(config_payload, os.path.join(args.output_dir, "config.json"))
    save_json(LABEL2ID, os.path.join(args.output_dir, "label2id.json"))

    log_path = os.path.join(args.output_dir, "logs.csv")
    best_f1 = -1.0
    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
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
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            if epoch % args.n_iter_estep == 1:
                q_new = run_estep(
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
            total_loss = 0.0
            total_macro = 0.0
            total_latent = 0.0
            num_batches = 0
            for embeddings, label_ids, selected_indices in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                embeddings = embeddings.to(device)
                label_ids = label_ids.to(device)
                batch_y = labels_to_onehot(label_ids, args.n_class)
                local_indices = to_local_train_indices(selected_indices, train_loader, len(train_idx))
                batch_q = torch.tensor(q_new[local_indices], dtype=torch.float32, device=device)

                features = model(embeddings)
                loss, ls1, ls2, ls3, _, _, _ = model.loss(
                    features,
                    batch_q,
                    batch_y,
                    args.tau,
                    args.alpha,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_macro += float(ls3.detach().cpu())
                total_latent += float((ls1 + ls2).detach().cpu())
                num_batches += 1

            val_metrics, _, _, _ = evaluate_split(
                model=model,
                data_loader=val_loader,
                device=device,
                tau=args.tau,
                n_class=args.n_class,
            )
            row = {
                "epoch": epoch,
                "train_loss": total_loss / max(num_batches, 1),
                "loss_macro": total_macro / max(num_batches, 1),
                "loss_latent": total_latent / max(num_batches, 1),
                **val_metrics,
            }
            writer.writerow(row)
            log_file.flush()
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_macro_f1={row['val_macro_f1']:.4f} | "
                f"val_balanced_acc={row['val_balanced_acc']:.4f}",
                flush=True,
            )

            save_checkpoint(
                os.path.join(args.output_dir, "last_model.pt"),
                model,
                args,
                dataset.get_input_dim(),
                train_idx,
                val_idx,
            )
            if val_metrics["val_macro_f1"] > best_f1:
                best_f1 = val_metrics["val_macro_f1"]
                save_checkpoint(
                    os.path.join(args.output_dir, "best_model.pt"),
                    model,
                    args,
                    dataset.get_input_dim(),
                    train_idx,
                    val_idx,
                )


def main() -> None:
    args = parse_args()
    if args.config:
        config = load_yaml_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
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
