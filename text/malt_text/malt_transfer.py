import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from malt_text.malt_dataset import MALTTargetDataset, build_target_dataloader
from malt_text.malt_losses import MALTLossComputer
from malt_text.malt_metrics import (
    active_cluster_count,
    anchor_drift_metrics,
    cluster_entropy,
    diagnostic_classification_metrics,
    geometry_metrics,
    mean_py_given_z_entropy,
    probability_summary,
)
from malt_text.malt_model import MALTTargetModel
from malt_text.utils import (
    copy_projector_state,
    init_target_mu_y,
    load_source_scgm,
    select_device,
    set_global_seed,
)
from scgm_text.dataset_text_embeddings import ID2LABEL
from scgm_text.projection import projection_from_checkpoint_args
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.utils_io import ensure_dir, save_json, save_numpy


def compute_p0_target(
    source_model: SCGMEmbeddingNet,
    data_loader: DataLoader,
    device: torch.device,
    tau_macro: float,
) -> Tuple[np.ndarray, np.ndarray]:
    projected: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for embeddings, _ in data_loader:
            embeddings = embeddings.to(device)
            features = source_model(embeddings)
            mu_y = source_model.mu_y
            logits = torch.nn.functional.normalize(features, p=2, dim=1) @ torch.nn.functional.normalize(
                mu_y, p=2, dim=1
            ).transpose(0, 1)
            prob = torch.softmax(logits / tau_macro, dim=1)
            projected.append(features.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())
    return np.concatenate(projected, axis=0), np.concatenate(probs, axis=0)


def initialize_nu_global(
    projected_source: np.ndarray,
    num_subclasses: int,
    hiddim: int,
    seed: int,
) -> np.ndarray:
    if projected_source.shape[0] < num_subclasses:
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(num_subclasses, hiddim)).astype(np.float32)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=num_subclasses,
            random_state=seed,
            batch_size=min(4096, projected_source.shape[0]),
            n_init="auto",
        )
        centers = kmeans.fit(projected_source).cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return centers / norms


def save_p0_artifacts(
    output_dir: str,
    metadata_df: pd.DataFrame,
    p0: np.ndarray,
) -> None:
    save_numpy(p0, os.path.join(output_dir, "p0_y_target.npy"))
    frame = metadata_df.copy()
    for macro_id, macro_name in ID2LABEL.items():
        frame[f"p0_{macro_name}"] = p0[:, macro_id]
    frame["p0_macro_id"] = p0.argmax(axis=1)
    frame["p0_macro_name"] = [ID2LABEL[int(value)] for value in frame["p0_macro_id"]]
    frame["p0_confidence"] = p0.max(axis=1)
    frame.to_csv(os.path.join(output_dir, "source_macro_pseudo_labels.csv"), index=False)


def evaluate_epoch(
    model: MALTTargetModel,
    data_loader: DataLoader,
    p0_full: np.ndarray,
    mu_source: torch.Tensor,
    device: torch.device,
    loss_computer: MALTLossComputer,
    tau_z: float,
    tau_yz: float,
    tau_macro: float,
    diagnostic_label_ids: Optional[np.ndarray],
) -> Dict[str, float]:
    model.eval()
    losses = {
        "loss_total": 0.0,
        "loss_softmacro": 0.0,
        "loss_latent": 0.0,
        "loss_anchor": 0.0,
        "loss_div": 0.0,
    }
    count = 0
    projected: List[np.ndarray] = []
    pt_probs: List[np.ndarray] = []
    z_hat: List[np.ndarray] = []
    p0_preds: List[np.ndarray] = []
    pt_preds: List[np.ndarray] = []
    with torch.no_grad():
        for embeddings, indices in data_loader:
            embeddings = embeddings.to(device)
            batch_p0 = torch.from_numpy(p0_full[indices.numpy()]).to(device=device, dtype=torch.float32)
            features = model(embeddings)
            prob_z_x = model.latent_probs(features, tau_z)
            prob_y_z = model.macro_given_latent(tau_yz)
            pt = model.marginal_macro(features, tau_z, tau_yz)
            batch_loss = loss_computer.forward(
                p0=batch_p0,
                prob_z_x=prob_z_x,
                prob_y_z=prob_y_z,
                pt=pt,
                mu_target=model.mu_y,
                mu_source=mu_source,
                nu=model.nu,
            )
            batch_size = embeddings.shape[0]
            count += batch_size
            for key, value in (
                ("loss_total", batch_loss.loss_total),
                ("loss_softmacro", batch_loss.loss_softmacro),
                ("loss_latent", batch_loss.loss_latent),
                ("loss_anchor", batch_loss.loss_anchor),
                ("loss_div", batch_loss.loss_div),
            ):
                losses[key] += float(value.item()) * batch_size
            projected.append(features.detach().cpu().numpy())
            pt_probs.append(pt.detach().cpu().numpy())
            z_hat.append(prob_z_x.argmax(dim=1).detach().cpu().numpy())
            p0_preds.append(batch_p0.argmax(dim=1).detach().cpu().numpy())
            pt_preds.append(pt.argmax(dim=1).detach().cpu().numpy())

    metrics = {key: value / max(count, 1) for key, value in losses.items()}
    projected_arr = np.concatenate(projected, axis=0)
    pt_arr = np.concatenate(pt_probs, axis=0)
    z_arr = np.concatenate(z_hat, axis=0)
    p0_summary = probability_summary(p0_full)
    pt_summary = probability_summary(pt_arr)
    metrics.update(
        {
            "mean_entropy_p0": p0_summary["mean_entropy"],
            "mean_entropy_pt": pt_summary["mean_entropy"],
            "mean_max_p0": p0_summary["mean_max"],
            "mean_max_pt": pt_summary["mean_max"],
        }
    )
    drift = anchor_drift_metrics(
        mu_source.detach().cpu().numpy(),
        model.mu_y.detach().cpu().numpy(),
    )
    metrics.update(drift)
    geom = geometry_metrics(projected_arr)
    metrics["rankme_target"] = geom["rankme"]
    metrics["c1_target"] = geom["c1"]
    metrics["c10_target"] = geom["c10"]
    metrics["n_active_clusters"] = float(active_cluster_count(z_arr, model.num_subclasses))
    metrics["cluster_entropy"] = cluster_entropy(z_arr, model.num_subclasses)
    metrics["mean_p_y_given_z_entropy"] = mean_py_given_z_entropy(
        model.macro_given_latent(tau_yz).detach().cpu().numpy()
    )
    if diagnostic_label_ids is not None:
        p0_diag = diagnostic_classification_metrics(
            diagnostic_label_ids,
            np.concatenate(p0_preds, axis=0),
        )
        pt_diag = diagnostic_classification_metrics(
            diagnostic_label_ids,
            np.concatenate(pt_preds, axis=0),
        )
        for key, value in p0_diag.items():
            metrics[f"p0_diag_{key}"] = value
        for key, value in pt_diag.items():
            metrics[f"pt_diag_{key}"] = value
    return metrics


def train_one_epoch(
    model: MALTTargetModel,
    data_loader: DataLoader,
    p0_full: np.ndarray,
    mu_source: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_computer: MALTLossComputer,
    tau_z: float,
    tau_yz: float,
) -> Dict[str, float]:
    model.train()
    losses = {
        "loss_total": 0.0,
        "loss_softmacro": 0.0,
        "loss_latent": 0.0,
        "loss_anchor": 0.0,
        "loss_div": 0.0,
    }
    count = 0
    for embeddings, indices in tqdm(data_loader, desc="train", leave=False):
        embeddings = embeddings.to(device)
        batch_p0 = torch.from_numpy(p0_full[indices.numpy()]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        features = model(embeddings)
        prob_z_x = model.latent_probs(features, tau_z)
        prob_y_z = model.macro_given_latent(tau_yz)
        pt = model.marginal_macro(features, tau_z, tau_yz)
        batch_loss = loss_computer.forward(
            p0=batch_p0,
            prob_z_x=prob_z_x,
            prob_y_z=prob_y_z,
            pt=pt,
            mu_target=model.mu_y,
            mu_source=mu_source,
            nu=model.nu,
        )
        if torch.isnan(batch_loss.loss_total):
            continue
        batch_loss.loss_total.backward()
        optimizer.step()
        batch_size = embeddings.shape[0]
        count += batch_size
        for key, value in (
            ("loss_total", batch_loss.loss_total),
            ("loss_softmacro", batch_loss.loss_softmacro),
            ("loss_latent", batch_loss.loss_latent),
            ("loss_anchor", batch_loss.loss_anchor),
            ("loss_div", batch_loss.loss_div),
        ):
            losses[key] += float(value.item()) * batch_size
    return {key: value / max(count, 1) for key, value in losses.items()}


def save_malt_checkpoint(
    path: str,
    model: MALTTargetModel,
    args: Any,
    input_dim: int,
    label2id: Dict[str, int],
    source_checkpoint: str,
    mu_source: np.ndarray,
    p0: np.ndarray,
    resolved_paths: Dict[str, str],
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args) if hasattr(args, "__dict__") else dict(args),
            "input_dim": input_dim,
            "label2id": label2id,
            "source_checkpoint": source_checkpoint,
            "mu_y_source": mu_source,
            "p0_y_target": p0,
            "resolved_paths": resolved_paths,
        },
        path,
    )


def run_malt_training(args) -> None:
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)
    device = select_device(args.device)

    source_model, source_args, label2id, input_dim = load_source_scgm(args.source_checkpoint, device)
    dataset = MALTTargetDataset(
        data_csv=args.resolved_target_data_csv,
        emb_csv=args.resolved_target_emb_csv,
        filter_pred_ok=args.filter_pred_ok,
        expected_input_dim=input_dim,
    )
    print(
        f"[MALT] samples={len(dataset)} epochs={args.epochs} device={device} out={args.output_dir}",
        flush=True,
    )
    data_loader = build_target_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
    )
    eval_loader = build_target_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
    )

    print("[MALT] calcul des pseudo-labels macro p0 (projection source)…", flush=True)
    projected_source, p0 = compute_p0_target(
        source_model=source_model,
        data_loader=eval_loader,
        device=device,
        tau_macro=args.tau_macro,
    )
    save_p0_artifacts(args.output_dir, dataset.get_metadata_df(), p0)

    proj = projection_from_checkpoint_args(source_args)
    target_model = MALTTargetModel(
        input_dim=input_dim,
        hiddim=int(source_args.get("hiddim", 128)),
        num_classes=4,
        num_subclasses=args.n_subclass,
        projection=proj,
        freeze_projector=args.freeze_projector,
    ).to(device)
    copy_projector_state(source_model, target_model)
    init_target_mu_y(source_model, target_model)
    nu_init = initialize_nu_global(
        projected_source=projected_source,
        num_subclasses=args.n_subclass,
        hiddim=target_model.hiddim,
        seed=args.seed,
    )
    with torch.no_grad():
        target_model.nu.copy_(torch.from_numpy(nu_init))

    mu_source = source_model.mu_y.detach().to(device)
    loss_computer = MALTLossComputer(
        beta_latent=args.beta_latent,
        beta_anchor=args.beta_anchor,
        beta_div=args.beta_div,
        tau_div=args.tau_div,
        confidence_threshold=args.confidence_threshold,
        latent_loss_mode=args.latent_loss_mode,
        use_sinkhorn=args.use_sinkhorn,
        sinkhorn_lmd=args.sinkhorn_lmd,
        disable_softmacro=args.disable_softmacro,
        disable_latent=args.disable_latent,
        disable_anchor=args.disable_anchor,
        disable_div=args.disable_div,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in target_model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log_path = os.path.join(args.output_dir, "logs.csv")
    fieldnames: Optional[List[str]] = None
    best_loss = float("inf")
    diagnostic_label_ids = dataset.get_diagnostic_label_ids()
    resolved_paths = {
        "target_data_csv": args.resolved_target_data_csv,
        "target_emb_csv": args.resolved_target_emb_csv,
    }

    for epoch in tqdm(
        range(1, args.epochs + 1),
        desc="MALT",
        unit="epoch",
        leave=True,
    ):
        train_metrics = train_one_epoch(
            model=target_model,
            data_loader=data_loader,
            p0_full=p0,
            mu_source=mu_source,
            optimizer=optimizer,
            device=device,
            loss_computer=loss_computer,
            tau_z=args.tau_z,
            tau_yz=args.tau_yz,
        )
        eval_metrics = evaluate_epoch(
            model=target_model,
            data_loader=eval_loader,
            p0_full=p0,
            mu_source=mu_source,
            device=device,
            loss_computer=loss_computer,
            tau_z=args.tau_z,
            tau_yz=args.tau_yz,
            tau_macro=args.tau_macro,
            diagnostic_label_ids=diagnostic_label_ids,
        )
        row = {"epoch": epoch, **train_metrics, **{f"val_{key}": value for key, value in eval_metrics.items()}}
        if fieldnames is None:
            fieldnames = list(row.keys())
            with open(log_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)

        print(
            f"[MALT] epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss_total']:.4f} val_loss={eval_metrics['loss_total']:.4f}",
            flush=True,
        )

        save_malt_checkpoint(
            path=os.path.join(args.output_dir, "last_model.pt"),
            model=target_model,
            args=args,
            input_dim=input_dim,
            label2id=label2id,
            source_checkpoint=args.source_checkpoint,
            mu_source=mu_source.detach().cpu().numpy(),
            p0=p0,
            resolved_paths=resolved_paths,
        )
        if eval_metrics["loss_total"] < best_loss:
            best_loss = eval_metrics["loss_total"]
            save_malt_checkpoint(
                path=os.path.join(args.output_dir, "best_model.pt"),
                model=target_model,
                args=args,
                input_dim=input_dim,
                label2id=label2id,
                source_checkpoint=args.source_checkpoint,
                mu_source=mu_source.detach().cpu().numpy(),
                p0=p0,
                resolved_paths=resolved_paths,
            )

    id2label = {int(value): key for key, value in label2id.items()}
    save_json(label2id, os.path.join(args.output_dir, "label2id.json"))
    save_json(id2label, os.path.join(args.output_dir, "id2label.json"))
    save_json(vars(args), os.path.join(args.output_dir, "config.json"))
    print("[MALT] terminé.", flush=True)
