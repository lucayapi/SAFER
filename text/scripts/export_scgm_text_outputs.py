import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scgm_text.dataset_text_embeddings import ID2LABEL, TextEmbeddingDataset
from scgm_text.projection import projection_from_checkpoint_args
from scgm_text.scgm_embedding_model import SCGMEmbeddingNet
from scgm_text.topic_export import export_topic_tables
from scgm_text.utils_io import ensure_dir, save_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SCGM text model outputs.")
    parser.add_argument("--data_csv", type=str, default="dataset/data_btp.csv")
    parser.add_argument("--emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_btp.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument("--pred_ok_col", type=str, default="pred_ok")
    parser.add_argument("--group_col", type=str, default="accident_id")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def labels_to_onehot(label_ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    onehot = torch.zeros(label_ids.shape[0], num_classes, dtype=torch.float32, device=label_ids.device)
    onehot.scatter_(1, label_ids.view(-1, 1), 1.0)
    return onehot


def run_export(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    dataset = TextEmbeddingDataset(
        data_csv=args.data_csv,
        emb_csv=args.emb_csv,
        label_col=args.label_col,
        pred_ok_col=args.pred_ok_col,
        group_col=args.group_col,
    )

    proj = projection_from_checkpoint_args(checkpoint_args)
    model = SCGMEmbeddingNet(
        input_dim=checkpoint.get("input_dim", dataset.get_input_dim()),
        hiddim=checkpoint_args.get("hiddim", 128),
        num_classes=checkpoint_args.get("n_class", 4),
        num_subclasses=checkpoint_args.get("n_subclass", 32),
        projection=proj,
    )
    model.load_state_dict(checkpoint["state_dict"])

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    tau = checkpoint_args.get("tau", 0.1)
    n_class = checkpoint_args.get("n_class", 4)
    metadata_df = dataset.get_metadata_df()

    projected = []
    prob_y_x_list = []
    prob_z_x_list = []
    z_hat = []
    macro_pred = []
    max_prob_y = []
    max_prob_z = []

    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            end = min(start + args.batch_size, len(dataset))
            batch_embeddings = []
            batch_labels = []
            for index in range(start, end):
                embedding, label_id, _ = dataset[index]
                batch_embeddings.append(embedding)
                batch_labels.append(label_id)
            embeddings = torch.stack(batch_embeddings).to(device)
            label_ids = torch.stack(batch_labels).to(device)
            batch_y = labels_to_onehot(label_ids, n_class)
            features = model(embeddings)
            prob_y_x, prob_z_x, prob_y_z = model.pred(features, tau)

            projected.append(features.cpu().numpy())
            prob_y_x_list.append(prob_y_x.cpu().numpy())
            prob_z_x_list.append(prob_z_x.cpu().numpy())
            z_hat.append(prob_z_x.argmax(dim=1).cpu().numpy())
            macro_pred.append(prob_y_x.argmax(dim=1).cpu().numpy())
            max_prob_y.append(prob_y_x.max(dim=1).values.cpu().numpy())
            max_prob_z.append(prob_z_x.max(dim=1).values.cpu().numpy())

    projected_embeddings = np.concatenate(projected, axis=0)
    prob_y_x = np.concatenate(prob_y_x_list, axis=0)
    prob_z_x = np.concatenate(prob_z_x_list, axis=0)
    z_hat_arr = np.concatenate(z_hat, axis=0)
    macro_pred_arr = np.concatenate(macro_pred, axis=0)
    max_prob_y_arr = np.concatenate(max_prob_y, axis=0)
    max_prob_z_arr = np.concatenate(max_prob_z, axis=0)

    mu_y = model.mu_y.detach().cpu()
    mu_z = model.mu_z.detach().cpu()
    mu_y_norm = F.normalize(mu_y, p=2, dim=1)
    mu_z_norm = F.normalize(mu_z, p=2, dim=1)
    prob_y_z = torch.exp(mu_z_norm @ mu_y_norm.T)
    prob_y_z = prob_y_z / prob_y_z.sum(dim=1, keepdim=True)
    prob_y_z = prob_y_z.numpy()
    save_numpy(mu_y.numpy(), os.path.join(args.output_dir, "mu_y.npy"))
    save_numpy(mu_z.numpy(), os.path.join(args.output_dir, "mu_z.npy"))
    save_numpy(projected_embeddings, os.path.join(args.output_dir, "projected_embeddings.npy"))
    save_numpy(prob_y_x, os.path.join(args.output_dir, "prob_y_x.npy"))
    save_numpy(prob_z_x, os.path.join(args.output_dir, "prob_z_x.npy"))
    save_numpy(prob_y_z, os.path.join(args.output_dir, "prob_y_z.npy"))

    z_assignments = metadata_df[["doc_id", "accident_id", "fact_id", args.label_col]].copy()
    z_assignments["z_hat"] = z_hat_arr
    z_assignments["max_prob_z"] = max_prob_z_arr
    z_assignments.to_csv(os.path.join(args.output_dir, "z_assignments.csv"), index=False)

    macro_predictions = metadata_df[["doc_id", args.label_col]].copy()
    macro_predictions["pred_macro_id"] = macro_pred_arr
    macro_predictions["pred_macro_name"] = [ID2LABEL[int(value)] for value in macro_pred_arr]
    macro_predictions["max_prob_y"] = max_prob_y_arr
    macro_predictions.to_csv(os.path.join(args.output_dir, "macro_predictions.csv"), index=False)

    enriched = metadata_df.copy()
    enriched["pred_macro_id"] = macro_pred_arr
    enriched["pred_macro_name"] = [ID2LABEL[int(value)] for value in macro_pred_arr]
    enriched["z_hat"] = z_hat_arr
    enriched["max_prob_y"] = max_prob_y_arr
    enriched["max_prob_z"] = max_prob_z_arr
    enriched.to_csv(os.path.join(args.output_dir, "metadata_with_predictions.csv"), index=False)

    export_topic_tables(
        metadata_df=metadata_df,
        projected_embeddings=projected_embeddings,
        mu_z=mu_z.numpy(),
        z_hat=z_hat_arr,
        output_dir=args.output_dir,
        sentence_col="sentence",
        label_col=args.label_col,
    )


def main() -> None:
    run_export(parse_args())


if __name__ == "__main__":
    main()
