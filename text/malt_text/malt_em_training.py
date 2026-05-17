"""MALT-EM training loop: full-dataset E-step + M-step with fixed q."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from malt_text.malt_dataset import MALTTargetDataset, build_target_dataloader
from malt_text.malt_em_estep import compute_malt_em_scores, run_malt_em_estep
from malt_text.malt_em_losses import MALTEMLossComputer
from malt_text.malt_em_model import MALTEMTargetModel
from malt_text.malt_metrics import (
    active_cluster_count,
    anchor_drift_metrics,
    cluster_entropy,
    diagnostic_classification_metrics,
    geometry_metrics,
    mean_py_given_z_entropy,
    probability_summary,
)
from malt_text.malt_transfer import (
    compute_p0_target,
    initialize_nu_global,
    save_malt_checkpoint,
    save_p0_artifacts,
)
from malt_text.utils import (
    copy_projector_state,
    init_target_mu_y,
    load_source_scgm,
    select_device,
    set_global_seed,
)
from scgm_text.dataset_text_embeddings import ID2LABEL
from scgm_text.optimizers import build_optimizer
from scgm_text.projection import projection_from_checkpoint_args
from scgm_text.schedulers import step_scheduler
from scgm_text.sinkhorn_estep import sinkhorn_assign
from scgm_text.utils_io import ensure_dir, save_json, save_numpy

MALT_EM_BANNER = """\
Running MALT-EM STRICT.
This mode implements a SCGM-like EM loop on the target corpus.
The observed coarse label y is replaced by transferred soft macro responsibilities p0(y|x).
E-step: full-dataset Sinkhorn over latent motif scores.
M-step: update target projector, macro anchors and latent motifs with q fixed.
"""


def _batch_indices(batch: Any) -> np.ndarray:
    if isinstance(batch, dict):
        idx = batch["index"]
    else:
        _, idx = batch
    return idx.detach().cpu().numpy()


def initialize_q_all(
    mode: str,
    n_samples: int,
    n_subclass: int,
    n_class: int,
    p0_all: np.ndarray,
    prob_z_all: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    sinkhorn_lmd: float = 25.0,
    seed: int = 42,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    rng = np.random.default_rng(seed)
    if mode == "random":
        q = np.zeros((n_samples, n_subclass), dtype=np.float32)
        for i in range(n_samples):
            k = int(rng.integers(0, n_subclass))
            q[i, k] = 1.0
        return q
    if mode == "p0_block":
        q = np.zeros((n_samples, n_subclass), dtype=np.float32)
        label_ids = p0_all.argmax(axis=1)
        for i, label_id in enumerate(label_ids):
            start = (int(label_id) * n_subclass) // n_class
            end = ((int(label_id) + 1) * n_subclass) // n_class
            component = int(rng.integers(start, max(start + 1, end)))
            q[i, component] = 1.0
        return q
    if mode == "kmeans" and prob_z_all is not None:
        z_hat = prob_z_all.argmax(axis=1)
        q = np.zeros((n_samples, n_subclass), dtype=np.float32)
        q[np.arange(n_samples), z_hat] = 1.0
        return q
    if mode in ("source_scores", "scores") and scores is not None:
        q_soft, z_hat, _ = sinkhorn_assign(scores, sinkhorn_lmd)
        q = np.zeros((n_samples, n_subclass), dtype=np.float32)
        q[np.arange(n_samples), z_hat.astype(np.int64)] = 1.0
        return q
    if prob_z_all is not None:
        z_hat = prob_z_all.argmax(axis=1)
        q = np.zeros((n_samples, n_subclass), dtype=np.float32)
        q[np.arange(n_samples), z_hat] = 1.0
        return q
    raise ValueError(f"Cannot initialize q with mode={mode!r}")


def _collect_prob_z_all(
    model: MALTEMTargetModel,
    data_loader: DataLoader,
    device: torch.device,
    tau_z: float,
    tau_yz: float,
    n_total: int,
) -> np.ndarray:
    k = model.num_subclasses
    prob_z_all = np.zeros((n_total, k), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                embeddings = batch["embedding"].to(device)
                indices = _batch_indices(batch)
            else:
                embeddings, idx_t = batch
                embeddings = embeddings.to(device)
                indices = idx_t.detach().cpu().numpy()
            features = model(embeddings)
            prob_z = model.latent_probs(features, tau_z).detach().cpu().numpy()
            prob_z_all[indices] = prob_z
    return prob_z_all


@torch.no_grad()
def evaluate_malt_em_epoch(
    model: MALTEMTargetModel,
    data_loader: DataLoader,
    p0_all: np.ndarray,
    q_all: np.ndarray,
    mu_source: torch.Tensor,
    device: torch.device,
    loss_computer: MALTEMLossComputer,
    tau_z: float,
    tau_yz: float,
    diagnostic_label_ids: Optional[np.ndarray],
) -> Dict[str, float]:
    model.eval()
    keys = [
        "loss_total",
        "loss_em",
        "loss_z",
        "loss_yz",
        "loss_anchor",
        "loss_div",
        "loss_macro",
        "loss_balance",
    ]
    totals = {k: 0.0 for k in keys}
    count = 0
    projected: List[np.ndarray] = []
    pt_probs: List[np.ndarray] = []
    z_hat: List[np.ndarray] = []
    p0_preds: List[np.ndarray] = []
    pt_preds: List[np.ndarray] = []

    for batch in data_loader:
        if isinstance(batch, dict):
            embeddings = batch["embedding"].to(device)
            indices = _batch_indices(batch)
        else:
            embeddings, idx_t = batch
            embeddings = embeddings.to(device)
            indices = idx_t.detach().cpu().numpy()

        batch_p0 = torch.from_numpy(p0_all[indices]).to(device=device, dtype=torch.float32)
        batch_q = torch.from_numpy(q_all[indices]).to(device=device, dtype=torch.float32)
        probs = model.compute_all_probs(embeddings, tau_z=tau_z, tau_yz=tau_yz)
        batch_loss = loss_computer.forward(
            p0=batch_p0,
            q=batch_q,
            prob_z_x=probs["prob_z_x"],
            prob_y_z=probs["prob_y_z"],
            prob_y_x=probs["prob_y_x"],
            mu_target=model.mu_y,
            mu_source=mu_source,
            nu=model.nu,
        )
        bs = embeddings.shape[0]
        count += bs
        for key in keys:
            totals[key] += float(getattr(batch_loss, key).item()) * bs
        projected.append(probs["features"].detach().cpu().numpy())
        pt_probs.append(probs["prob_y_x"].detach().cpu().numpy())
        z_hat.append(q_all[indices].argmax(axis=1))
        p0_preds.append(batch_p0.argmax(dim=1).detach().cpu().numpy())
        pt_preds.append(probs["prob_y_x"].argmax(dim=1).detach().cpu().numpy())

    metrics = {k: v / max(count, 1) for k, v in totals.items()}
    projected_arr = np.concatenate(projected, axis=0)
    pt_arr = np.concatenate(pt_probs, axis=0)
    z_arr = np.concatenate(z_hat, axis=0)
    p0_summary = probability_summary(p0_all)
    pt_summary = probability_summary(pt_arr)
    metrics["mean_entropy_p0"] = p0_summary["mean_entropy"]
    metrics["mean_entropy_pt"] = pt_summary["mean_entropy"]
    metrics["mean_max_p0"] = p0_summary["mean_max"]
    metrics["mean_max_pt"] = pt_summary["mean_max"]
    metrics.update(anchor_drift_metrics(mu_source.detach().cpu().numpy(), model.mu_y.detach().cpu().numpy()))
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
        for prefix, preds in (("p0_diag", np.concatenate(p0_preds)), ("pt_diag", np.concatenate(pt_preds))):
            diag = diagnostic_classification_metrics(diagnostic_label_ids, preds)
            for k, v in diag.items():
                metrics[f"{prefix}_{k}"] = v
    return metrics


def train_mstep_epoch(
    model: MALTEMTargetModel,
    data_loader: DataLoader,
    p0_all: np.ndarray,
    q_all: np.ndarray,
    mu_source: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_computer: MALTEMLossComputer,
    tau_z: float,
    tau_yz: float,
) -> Dict[str, float]:
    model.train()
    keys = [
        "loss_total",
        "loss_em",
        "loss_z",
        "loss_yz",
        "loss_anchor",
        "loss_div",
        "loss_macro",
        "loss_balance",
    ]
    totals = {k: 0.0 for k in keys}
    count = 0
    for batch in tqdm(data_loader, desc="mstep", leave=False):
        if isinstance(batch, dict):
            embeddings = batch["embedding"].to(device)
            indices = _batch_indices(batch)
        else:
            embeddings, idx_t = batch
            embeddings = embeddings.to(device)
            indices = idx_t.detach().cpu().numpy()

        batch_p0 = torch.from_numpy(p0_all[indices]).to(device=device, dtype=torch.float32)
        batch_q = torch.from_numpy(q_all[indices]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        probs = model.compute_all_probs(embeddings, tau_z=tau_z, tau_yz=tau_yz)
        batch_loss = loss_computer.forward(
            p0=batch_p0,
            q=batch_q,
            prob_z_x=probs["prob_z_x"],
            prob_y_z=probs["prob_y_z"],
            prob_y_x=probs["prob_y_x"],
            mu_target=model.mu_y,
            mu_source=mu_source,
            nu=model.nu,
        )
        if torch.isnan(batch_loss.loss_total):
            continue
        batch_loss.loss_total.backward()
        optimizer.step()
        bs = embeddings.shape[0]
        count += bs
        for key in keys:
            totals[key] += float(getattr(batch_loss, key).item()) * bs
    return {k: v / max(count, 1) for k, v in totals.items()}


def _q_change_rate(q_prev: Optional[np.ndarray], q_new: np.ndarray) -> float:
    if q_prev is None:
        return 1.0
    z_prev = q_prev.argmax(axis=1)
    z_new = q_new.argmax(axis=1)
    return float(np.mean(z_prev != z_new))


def save_malt_em_checkpoint(
    path: str,
    model: MALTEMTargetModel,
    args: Any,
    input_dim: int,
    label2id: Dict[str, int],
    source_checkpoint: str,
    mu_source: np.ndarray,
    p0: np.ndarray,
    q_all: np.ndarray,
    resolved_paths: Dict[str, str],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
        "input_dim": input_dim,
        "label2id": label2id,
        "source_checkpoint": source_checkpoint,
        "mu_y_source": mu_source,
        "p0_y_target": p0,
        "q_em": q_all,
        "resolved_paths": resolved_paths,
        "malt_mode": "em_strict",
    }
    torch.save(payload, path)


def run_malt_em_training(args: Any) -> None:
    print(MALT_EM_BANNER, flush=True)
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)
    metrics_dir = os.path.join(args.output_dir, "metrics")
    ensure_dir(metrics_dir)
    device = select_device(args.device)

    source_model, source_args, label2id, input_dim = load_source_scgm(args.source_checkpoint, device)
    n_class = int(getattr(args, "num_classes", 4))
    dataset = MALTTargetDataset(
        data_csv=args.resolved_target_data_csv,
        emb_csv=args.resolved_target_emb_csv,
        filter_pred_ok=getattr(args, "filter_pred_ok", False),
        expected_input_dim=input_dim,
        return_index=True,
    )
    n_samples = len(dataset)
    print(
        f"[MALT-EM] samples={n_samples} epochs={args.epochs} device={device} out={args.output_dir}",
        flush=True,
    )

    eval_loader = build_target_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
        return_index=True,
    )
    train_loader = build_target_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
        return_index=True,
    )

    print("[MALT-EM] computing transferred macro p0(y|x)…", flush=True)
    projected_source, p0 = compute_p0_target(
        source_model=source_model,
        data_loader=eval_loader,
        device=device,
        tau_macro=args.tau_macro,
    )
    save_p0_artifacts(args.output_dir, dataset.get_metadata_df(), p0)

    proj = getattr(args, "projection", None) or projection_from_checkpoint_args(source_args)
    if getattr(args, "copy_source_projector", True):
        proj = projection_from_checkpoint_args(source_args)

    target_model = MALTEMTargetModel(
        input_dim=input_dim,
        hiddim=int(getattr(args, "hiddim", source_args.get("hiddim", 128))),
        num_classes=n_class,
        num_subclasses=args.n_subclass,
        projection=proj,
        freeze_projector=getattr(args, "freeze_projector", False),
    ).to(device)

    if getattr(args, "copy_source_projector", True):
        copy_projector_state(source_model, target_model)
    if str(getattr(args, "init_mu_target", "source")).lower() == "source":
        init_target_mu_y(source_model, target_model)
    nu_init = initialize_nu_global(
        projected_source=projected_source,
        num_subclasses=args.n_subclass,
        hiddim=target_model.hiddim,
        seed=args.seed,
    )
    with torch.no_grad():
        target_model.nu.copy_(torch.from_numpy(nu_init).to(device))

    mu_source = source_model.mu_y.detach().to(device)
    loss_computer = MALTEMLossComputer(
        beta_anchor=args.beta_anchor,
        beta_div=args.beta_div,
        beta_macro=args.beta_macro,
        beta_balance=getattr(args, "beta_balance", 0.0),
        tau_div=args.tau_div,
        confidence_threshold=getattr(args, "confidence_threshold", 0.0),
        macro_weight_mode=getattr(args, "macro_weight_mode", "max_prob"),
        disable_anchor=getattr(args, "disable_anchor", False),
        disable_div=getattr(args, "disable_div", False),
        disable_macro=getattr(args, "disable_macro", False),
        disable_balance=getattr(args, "disable_balance", False),
    )
    optimizer = build_optimizer(target_model, args)

    prob_z_init = _collect_prob_z_all(
        target_model, eval_loader, device, args.tau_z, args.tau_yz, n_samples
    )
    scores_init = None
    init_mode = getattr(args, "init_q_mode", "source_scores")
    if init_mode in ("source_scores", "scores"):
        p0_t = torch.from_numpy(p0).to(device=device, dtype=torch.float32)
        prob_z_t = torch.from_numpy(prob_z_init).to(device=device, dtype=torch.float32)
        prob_y_z_t = target_model.macro_given_latent(args.tau_yz)
        scores_init, _ = compute_malt_em_scores(prob_z_t, prob_y_z_t, p0_t)
        scores_init = scores_init.detach().cpu().numpy()

    q_all = initialize_q_all(
        mode=init_mode,
        n_samples=n_samples,
        n_subclass=args.n_subclass,
        n_class=n_class,
        p0_all=p0,
        prob_z_all=prob_z_init,
        scores=scores_init,
        sinkhorn_lmd=args.sinkhorn_lmd,
        seed=args.seed,
    )
    q_prev: Optional[np.ndarray] = None
    z_prev: Optional[np.ndarray] = None

    log_path = os.path.join(metrics_dir, "train_log.csv")
    jsonl_path = os.path.join(metrics_dir, "epoch_metrics.jsonl")
    legacy_log = os.path.join(args.output_dir, "logs.csv")
    fieldnames: Optional[List[str]] = None
    best_loss = float("inf")
    diagnostic_label_ids = dataset.get_diagnostic_label_ids()
    resolved_paths = {
        "target_data_csv": args.resolved_target_data_csv,
        "target_emb_csv": args.resolved_target_emb_csv,
    }
    n_iter_estep = int(getattr(args, "n_iter_estep", 5))
    em_q_mode = getattr(args, "em_q_mode", "hard")

    for epoch in tqdm(range(1, args.epochs + 1), desc="MALT-EM", unit="epoch"):
        run_estep = epoch == 1 or (epoch % n_iter_estep == 0)
        estep_diag: Dict[str, float] = {}
        if run_estep:
            q_all, z_hat, estep_diag = run_malt_em_estep(
                model=target_model,
                data_loader=eval_loader,
                p0_all=p0,
                tau_z=args.tau_z,
                tau_yz=args.tau_yz,
                sinkhorn_lmd=args.sinkhorn_lmd,
                device=device,
                q_mode=em_q_mode,
            )
            if getattr(args, "save_q_every_estep", True):
                save_numpy(q_all, os.path.join(args.output_dir, f"q_epoch_{epoch:04d}.npy"))
                save_numpy(z_hat, os.path.join(args.output_dir, f"z_hat_epoch_{epoch:04d}.npy"))
            estep_diag["q_change_rate"] = _q_change_rate(q_prev, q_all)
            estep_diag["z_change_rate"] = (
                float(np.mean(z_prev != z_hat)) if z_prev is not None else 1.0
            )
            q_prev = q_all.copy()
            z_prev = z_hat.copy()

        current_lr = step_scheduler(optimizer, args, epoch, args.epochs)
        train_metrics = train_mstep_epoch(
            model=target_model,
            data_loader=train_loader,
            p0_all=p0,
            q_all=q_all,
            mu_source=mu_source,
            optimizer=optimizer,
            device=device,
            loss_computer=loss_computer,
            tau_z=args.tau_z,
            tau_yz=args.tau_yz,
        )
        eval_metrics = evaluate_malt_em_epoch(
            model=target_model,
            data_loader=eval_loader,
            p0_all=p0,
            q_all=q_all,
            mu_source=mu_source,
            device=device,
            loss_computer=loss_computer,
            tau_z=args.tau_z,
            tau_yz=args.tau_yz,
            diagnostic_label_ids=diagnostic_label_ids,
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            **train_metrics,
            **{f"val_{k}": v for k, v in eval_metrics.items()},
            **{f"estep_{k}": v for k, v in estep_diag.items()},
        }
        if fieldnames is None:
            fieldnames = list(row.keys())
            for path in (log_path, legacy_log):
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        with open(legacy_log, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        with open(jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        print(
            f"[MALT-EM] epoch {epoch}/{args.epochs} lr={current_lr:.6f} "
            f"loss_em={train_metrics['loss_em']:.4f} val_loss={eval_metrics['loss_total']:.4f}",
            flush=True,
        )

        save_malt_em_checkpoint(
            path=os.path.join(args.output_dir, "last_model.pt"),
            model=target_model,
            args=args,
            input_dim=input_dim,
            label2id=label2id,
            source_checkpoint=args.source_checkpoint,
            mu_source=mu_source.detach().cpu().numpy(),
            p0=p0,
            q_all=q_all,
            resolved_paths=resolved_paths,
        )
        if eval_metrics["loss_total"] < best_loss:
            best_loss = eval_metrics["loss_total"]
            save_malt_em_checkpoint(
                path=os.path.join(args.output_dir, "best_model.pt"),
                model=target_model,
                args=args,
                input_dim=input_dim,
                label2id=label2id,
                source_checkpoint=args.source_checkpoint,
                mu_source=mu_source.detach().cpu().numpy(),
                p0=p0,
                q_all=q_all,
                resolved_paths=resolved_paths,
            )
            save_numpy(q_all, os.path.join(args.output_dir, "q_em_final.npy"))
            save_numpy(q_all.argmax(axis=1), os.path.join(args.output_dir, "z_hat_em_final.npy"))

    id2label = {int(v): k for k, v in label2id.items()}
    save_json(label2id, os.path.join(args.output_dir, "label2id.json"))
    save_json(id2label, os.path.join(args.output_dir, "id2label.json"))
    cfg = vars(args) if hasattr(args, "__dict__") else dict(args)
    save_json(cfg, os.path.join(args.output_dir, "config.json"))
    try:
        import yaml

        with open(os.path.join(args.output_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass
    print("[MALT-EM] done.", flush=True)

