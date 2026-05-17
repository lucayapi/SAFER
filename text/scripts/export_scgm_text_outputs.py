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

from scgm_text.batch_utils import batch_to_device, forward_features
from scgm_text.checkpoint_io import load_scgm_checkpoint
from scgm_text.collate import make_text_collate_fn
from scgm_text.dataset_text_embeddings import ID2LABEL, TextEmbeddingDataset
from scgm_text.dataset_text_raw import TextRawDataset
from scgm_text.topic_export import export_topic_tables
from safer_core.paths import layout_method_output
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
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=256)
    return parser.parse_args()


def labels_to_onehot(label_ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    onehot = torch.zeros(label_ids.shape[0], num_classes, dtype=torch.float32, device=label_ids.device)
    onehot.scatter_(1, label_ids.view(-1, 1), 1.0)
    return onehot


def run_export(args: argparse.Namespace) -> None:
    layout = layout_method_output("scgm_text", args.output_dir)
    args.output_dir = str(layout["root"])
    emb_dir = layout["embeddings"]
    assign_dir = layout["assignments"]
    topics_dir = layout["topics"]
    for d in (emb_dir, assign_dir, topics_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, checkpoint_args, _ = load_scgm_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()

    input_mode = checkpoint_args.get("input_mode", "precomputed_embeddings")
    tau = checkpoint_args.get("tau", 0.1)
    n_class = checkpoint_args.get("n_class", 4)

    if input_mode == "text":
        dataset = TextRawDataset(
            data_csv=args.data_csv,
            label_col=args.label_col,
            pred_ok_col=args.pred_ok_col,
            group_col=args.group_col,
            text_col=args.text_col,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_args.get("backbone_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B")
        )
        collate_fn = make_text_collate_fn(tokenizer, args.max_seq_length)
        batch_size = min(args.batch_size, 32)
    else:
        dataset = TextEmbeddingDataset(
            data_csv=args.data_csv,
            emb_csv=args.emb_csv,
            label_col=args.label_col,
            pred_ok_col=args.pred_ok_col,
            group_col=args.group_col,
        )
        collate_fn = None
        batch_size = args.batch_size

    metadata_df = dataset.get_metadata_df()

    projected = []
    raw_embeddings = []
    prob_y_x_list = []
    prob_z_x_list = []
    z_hat = []
    macro_pred = []
    max_prob_y = []
    max_prob_z = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            if input_mode == "text":
                items = [dataset[index] for index in range(start, end)]
                batch = collate_fn(items)
                batch = batch_to_device(batch, device)
                label_ids = batch["label_ids"]
                raw_embeddings.append(model.encode(batch).cpu().numpy())
                features = forward_features(model, batch)
            else:
                batch_embeddings = []
                batch_labels = []
                for index in range(start, end):
                    embedding, label_id, _ = dataset[index]
                    batch_embeddings.append(embedding)
                    batch_labels.append(label_id)
                embeddings = torch.stack(batch_embeddings).to(device)
                label_ids = torch.stack(batch_labels)
                raw_embeddings.append(embeddings.cpu().numpy())
                features = model(embeddings)

            batch_y = labels_to_onehot(label_ids.to(device), n_class)
            prob_y_x, prob_z_x, prob_y_z = model.pred(features, tau)

            projected.append(features.cpu().numpy())
            prob_y_x_list.append(prob_y_x.cpu().numpy())
            prob_z_x_list.append(prob_z_x.cpu().numpy())
            z_hat.append(prob_z_x.argmax(dim=1).cpu().numpy())
            macro_pred.append(prob_y_x.argmax(dim=1).cpu().numpy())
            max_prob_y.append(prob_y_x.max(dim=1).values.cpu().numpy())
            max_prob_z.append(prob_z_x.max(dim=1).values.cpu().numpy())

    projected_embeddings = np.concatenate(projected, axis=0)
    raw_embeddings_arr = np.concatenate(raw_embeddings, axis=0)
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
    save_numpy(mu_y.numpy(), str(emb_dir / "mu_y.npy"))
    save_numpy(mu_z.numpy(), str(emb_dir / "mu_z.npy"))
    save_numpy(raw_embeddings_arr, str(emb_dir / "raw_embeddings.npy"))
    save_numpy(projected_embeddings, str(emb_dir / "projected_embeddings.npy"))
    save_numpy(prob_y_x, str(emb_dir / "prob_y_x.npy"))
    save_numpy(prob_z_x, str(emb_dir / "prob_z_x.npy"))
    save_numpy(prob_y_z, str(emb_dir / "prob_y_z.npy"))

    z_assignments = metadata_df[["doc_id", "accident_id", "fact_id", args.label_col]].copy()
    z_assignments["z_hat"] = z_hat_arr
    z_assignments["max_prob_z"] = max_prob_z_arr
    z_assignments.to_csv(assign_dir / "z_assignments.csv", index=False)

    macro_predictions = metadata_df[["doc_id", args.label_col]].copy()
    macro_predictions["pred_macro_id"] = macro_pred_arr
    macro_predictions["pred_macro_name"] = [ID2LABEL[int(value)] for value in macro_pred_arr]
    macro_predictions["max_prob_y"] = max_prob_y_arr
    macro_predictions.to_csv(assign_dir / "macro_predictions.csv", index=False)

    enriched = metadata_df.copy()
    enriched["pred_macro_id"] = macro_pred_arr
    enriched["pred_macro_name"] = [ID2LABEL[int(value)] for value in macro_pred_arr]
    enriched["z_hat"] = z_hat_arr
    enriched["max_prob_y"] = max_prob_y_arr
    enriched["max_prob_z"] = max_prob_z_arr
    enriched.to_csv(emb_dir / "metadata_with_predictions.csv", index=False)

    export_topic_tables(
        metadata_df=metadata_df,
        projected_embeddings=projected_embeddings,
        mu_z=mu_z.numpy(),
        z_hat=z_hat_arr,
        output_dir=str(topics_dir),
        sentence_col="sentence",
        label_col=args.label_col,
    )


def main() -> None:
    run_export(parse_args())


if __name__ == "__main__":
    main()
