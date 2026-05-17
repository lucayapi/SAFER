import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _as_repo_path(repo_root: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


from malt_text.malt_dataset import MALTTargetDataset, build_target_dataloader
from malt_text.malt_metrics import anchor_drift_metrics
from malt_text.malt_em_model import MALTEMTargetModel
from malt_text.malt_topic_export import export_malt_topic_tables
from malt_text.malt_transfer import compute_p0_target
from malt_text.utils import (
    load_source_scgm,
    resolve_existing_path,
    resolve_target_embedding_csv,
    select_device,
)
from scgm_text.dataset_text_embeddings import ID2LABEL
from scgm_text.projection import projection_from_checkpoint_args
from scgm_text.utils_io import ensure_dir, save_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MALT target outputs.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source_checkpoint", type=str, default="resultats/scgm_text/checkpoints/best_model.pt")
    parser.add_argument("--target_data_csv", type=str, default="dataset/test/data_metallurgie.csv")
    parser.add_argument("--target_data_csv_alt", type=str, default="dataset/test/data_metallurgie.csv")
    parser.add_argument("--target_emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_mettalurgie.csv")
    parser.add_argument("--target_emb_csv_alt", type=str, default="embeddings/Qwen3-Embedding-0.6B_metallurgie.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def run_export(args: argparse.Namespace) -> None:
    args = argparse.Namespace(**vars(args))
    args.output_dir = _as_repo_path(ROOT_DIR, args.output_dir)
    args.checkpoint = _as_repo_path(ROOT_DIR, args.checkpoint)
    args.source_checkpoint = _as_repo_path(ROOT_DIR, args.source_checkpoint)
    ensure_dir(args.output_dir)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    input_dim = int(checkpoint.get("input_dim", checkpoint_args.get("input_dim", 0)))

    target_data_csv = resolve_existing_path(
        _as_repo_path(ROOT_DIR, args.target_data_csv),
        _as_repo_path(ROOT_DIR, args.target_data_csv_alt),
        "target data CSV",
    )
    target_emb_csv = resolve_target_embedding_csv(
        _as_repo_path(ROOT_DIR, args.target_emb_csv),
        _as_repo_path(ROOT_DIR, args.target_emb_csv_alt),
        _as_repo_path(ROOT_DIR, "embeddings"),
    )
    dataset = MALTTargetDataset(
        data_csv=target_data_csv,
        emb_csv=target_emb_csv,
        filter_pred_ok=checkpoint_args.get("filter_pred_ok", False),
        expected_input_dim=input_dim,
    )
    data_loader = build_target_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    source_model, source_args, _, _ = load_source_scgm(args.source_checkpoint, device)
    proj = projection_from_checkpoint_args(source_args)
    target_model = MALTEMTargetModel(
        input_dim=input_dim,
        hiddim=int(source_args.get("hiddim", 128)),
        num_classes=4,
        num_subclasses=int(checkpoint_args.get("n_subclass", 32)),
        projection=proj,
        freeze_projector=bool(checkpoint_args.get("freeze_projector", False)),
    )
    target_model.load_state_dict(checkpoint["state_dict"])
    target_model.to(device)
    target_model.eval()

    tau_macro = float(checkpoint_args.get("tau_macro", 0.1))
    tau_z = float(checkpoint_args.get("tau_z", 0.1))
    tau_yz = float(checkpoint_args.get("tau_yz", 0.1))

    projected_source, p0 = compute_p0_target(source_model, data_loader, device, tau_macro)
    projected_adapted = []
    pt_list = []
    pz_list = []
    z_hat_parts = []
    z_conf_parts = []
    with torch.no_grad():
        for embeddings, _ in data_loader:
            embeddings = embeddings.to(device)
            probs = target_model.compute_all_probs(embeddings, tau_z=tau_z, tau_yz=tau_yz)
            features = probs["features"]
            prob_z_x = probs["prob_z_x"]
            pt = probs["prob_y_x"]
            projected_adapted.append(features.detach().cpu().numpy())
            pt_list.append(pt.detach().cpu().numpy())
            pz_list.append(prob_z_x.detach().cpu().numpy())
            z_hat_parts.append(prob_z_x.argmax(dim=1).detach().cpu().numpy())
            z_conf_parts.append(prob_z_x.max(dim=1).values.detach().cpu().numpy())

    projected_adapted_arr = np.concatenate(projected_adapted, axis=0)
    pt_arr = np.concatenate(pt_list, axis=0)
    pz_arr = np.concatenate(pz_list, axis=0)
    z_hat_arr = np.concatenate(z_hat_parts, axis=0)
    z_conf_arr = np.concatenate(z_conf_parts, axis=0)

    q_em = checkpoint.get("q_em")
    if q_em is None:
        q_path = os.path.join(os.path.dirname(args.checkpoint), "q_em_final.npy")
        if os.path.isfile(q_path):
            q_em = np.load(q_path)
    if q_em is not None:
        q_em = np.asarray(q_em, dtype=np.float32)
        if q_em.shape[0] == len(dataset):
            z_hat_arr = q_em.argmax(axis=1).astype(np.int64)
            q_z_hat_arr = q_em.max(axis=1)
        else:
            q_z_hat_arr = z_conf_arr
    else:
        q_z_hat_arr = z_conf_arr
    prob_y_z = target_model.macro_given_latent(tau_yz).detach().cpu().numpy()
    mu_source = source_model.mu_y.detach().cpu().numpy()
    mu_target = target_model.mu_y.detach().cpu().numpy()
    nu_target = target_model.nu.detach().cpu().numpy()

    save_numpy(projected_source, os.path.join(args.output_dir, "target_projected_source.npy"))
    save_numpy(projected_adapted_arr, os.path.join(args.output_dir, "target_projected_adapted.npy"))
    save_numpy(p0, os.path.join(args.output_dir, "p0_y_target.npy"))
    save_numpy(pt_arr, os.path.join(args.output_dir, "pt_y_target.npy"))
    save_numpy(pz_arr, os.path.join(args.output_dir, "pt_z_target.npy"))
    save_numpy(prob_y_z, os.path.join(args.output_dir, "pt_y_given_z.npy"))
    save_numpy(mu_source, os.path.join(args.output_dir, "mu_y_source.npy"))
    save_numpy(mu_target, os.path.join(args.output_dir, "mu_y_target.npy"))
    save_numpy(nu_target, os.path.join(args.output_dir, "nu_target.npy"))
    if q_em is not None:
        save_numpy(q_em, os.path.join(args.output_dir, "q_em_final.npy"))
        save_numpy(z_hat_arr, os.path.join(args.output_dir, "z_hat_em_final.npy"))

    p0_hat = p0.argmax(axis=1)
    pt_hat = pt_arr.argmax(axis=1)
    transition = pd.crosstab(
        pd.Series([ID2LABEL[int(i)] for i in p0_hat], name="p0_hat"),
        pd.Series([ID2LABEL[int(i)] for i in pt_hat], name="pt_hat"),
    )
    transition.to_csv(os.path.join(args.output_dir, "macro_transition_p0_to_pt.csv"))

    drift = anchor_drift_metrics(mu_source, mu_target)
    pd.DataFrame([drift]).to_csv(os.path.join(args.output_dir, "anchor_drift.csv"), index=False)

    metadata = dataset.get_metadata_df()
    enriched = metadata.copy()
    for macro_id, macro_name in ID2LABEL.items():
        enriched[f"p0_{macro_name}"] = p0[:, macro_id]
        enriched[f"pt_{macro_name}"] = pt_arr[:, macro_id]
    enriched["p0_hat"] = p0_hat
    enriched["pt_hat"] = pt_hat
    enriched["p0_macro_id"] = p0_hat
    enriched["p0_macro_name"] = [ID2LABEL[int(value)] for value in p0_hat]
    enriched["p0_confidence"] = p0.max(axis=1)
    enriched["pt_macro_id"] = pt_hat
    enriched["pt_macro_name"] = [ID2LABEL[int(value)] for value in pt_hat]
    enriched["pt_confidence"] = pt_arr.max(axis=1)
    enriched["z_hat"] = z_hat_arr
    enriched["z_confidence"] = z_conf_arr
    enriched["q_z_hat"] = q_z_hat_arr
    enriched["z_dominant_macro"] = [ID2LABEL[int(prob_y_z[z].argmax())] for z in z_hat_arr]
    for macro_id, macro_name in ID2LABEL.items():
        enriched[f"p_{macro_name}_given_z"] = prob_y_z[z_hat_arr, macro_id]
    enriched.to_csv(os.path.join(args.output_dir, "metadata_with_malt_predictions.csv"), index=False)

    z_assignments = enriched[
        [
            "doc_id",
            "accident_id",
            "fact_id",
            "z_hat",
            "z_confidence",
            "z_dominant_macro",
            "p0_macro_name",
            "pt_macro_name",
        ]
    ].copy()
    z_assignments.to_csv(os.path.join(args.output_dir, "z_assignments_target.csv"), index=False)

    export_malt_topic_tables(
        metadata_df=metadata,
        projected_embeddings=projected_adapted_arr,
        nu=nu_target,
        z_hat=z_hat_arr,
        prob_y_z=prob_y_z,
        p0=p0,
        pt=pt_arr,
        output_dir=args.output_dir,
    )


def main() -> None:
    run_export(parse_args())


if __name__ == "__main__":
    main()
