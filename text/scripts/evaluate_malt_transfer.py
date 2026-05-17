import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from malt_text.malt_metrics import anchor_drift_metrics, probability_summary
from metrics.embedding_geometry_separation import METRICS_TABLE_COLUMNS, build_geometry_metrics_row
from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
from scgm_text.utils_io import ensure_dir, save_json


def load_raw_embeddings(metadata: pd.DataFrame, emb_csv: str) -> np.ndarray:
    slim = metadata.drop(columns=[c for c in metadata.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_columns = merge_metadata_with_embeddings(slim, emb_csv)
    if len(merged) != len(metadata):
        raise ValueError(
            f"Embedding merge row mismatch: metadata={len(metadata)}, merged={len(merged)}"
        )
    return merged[dim_columns].to_numpy(dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported MALT transfer outputs (eta2 geometry).")
    parser.add_argument("--exports_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument("--emb_csv", type=str, default=None)
    parser.add_argument("--method_raw", type=str, default="Embedding brut")
    return parser.parse_args()


def transition_matrix(p0_ids: np.ndarray, pt_ids: np.ndarray) -> pd.DataFrame:
    labels = ["A0", "A1", "B", "C"]
    matrix = np.zeros((4, 4), dtype=np.int64)
    for p0_id, pt_id in zip(p0_ids, pt_ids):
        matrix[int(p0_id), int(pt_id)] += 1
    return pd.DataFrame(matrix, index=labels, columns=labels)


def run_evaluate(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    exports_dir = args.exports_dir
    metadata = pd.read_csv(os.path.join(exports_dir, "metadata_with_malt_predictions.csv"))
    projected_source = np.load(os.path.join(exports_dir, "target_projected_source.npy"))
    projected_adapted = np.load(os.path.join(exports_dir, "target_projected_adapted.npy"))
    p0 = np.load(os.path.join(exports_dir, "p0_y_target.npy"))
    pt = np.load(os.path.join(exports_dir, "pt_y_target.npy"))
    prob_y_z = np.load(os.path.join(exports_dir, "pt_y_given_z.npy"))
    mu_source = np.load(os.path.join(exports_dir, "mu_y_source.npy"))
    mu_target = np.load(os.path.join(exports_dir, "mu_y_target.npy"))
    z_hat = metadata["z_hat"].to_numpy(dtype=np.int64)
    macro_labels = metadata[args.label_col].to_numpy()

    malt_diagnostics = {
        "p0_distribution": probability_summary(p0),
        "pt_distribution": probability_summary(pt),
        "change_rate": float((metadata["p0_macro_id"].to_numpy() != metadata["pt_macro_id"].to_numpy()).mean()),
        "anchor_drift": anchor_drift_metrics(mu_source, mu_target),
    }

    transition = transition_matrix(
        metadata["p0_macro_id"].to_numpy(dtype=np.int64),
        metadata["pt_macro_id"].to_numpy(dtype=np.int64),
    )
    transition.to_csv(os.path.join(args.output_dir, "macro_transition_matrix.csv"))
    transition.to_csv(os.path.join(args.output_dir, "transition_matrix.csv"))
    pd.DataFrame([malt_diagnostics["anchor_drift"]]).to_csv(
        os.path.join(args.output_dir, "anchor_drift.csv"), index=False
    )

    q_path = os.path.join(exports_dir, "q_em_final.npy")
    if os.path.isfile(q_path):
        q_em = np.load(q_path)
        malt_diagnostics["q_entropy_mean"] = float(
            -np.mean(np.sum(q_em * np.log(np.clip(q_em, 1e-12, None)), axis=1))
        )
        malt_diagnostics["em_active_clusters"] = int(np.unique(q_em.argmax(axis=1)).size)

    run_root = os.path.dirname(exports_dir.rstrip("/\\"))
    for log_name in ("metrics/train_log.csv", "logs.csv"):
        log_path = os.path.join(run_root, log_name)
        if os.path.isfile(log_path):
            log_df = pd.read_csv(log_path)
            for col in ("loss_em", "loss_z", "loss_yz", "estep_q_change_rate", "estep_z_change_rate"):
                if col in log_df.columns:
                    malt_diagnostics[f"last_{col}"] = float(log_df[col].iloc[-1])
            break

    from scgm_text.dataset_text_embeddings import ID2LABEL

    z_affiliation = []
    for z_id in range(prob_y_z.shape[0]):
        dominant_macro = ID2LABEL[int(np.argmax(prob_y_z[z_id]))]
        z_affiliation.append(
            {
                "z_id": z_id,
                "dominant_macro": dominant_macro,
                "p_A0": float(prob_y_z[z_id, 0]),
                "p_A1": float(prob_y_z[z_id, 1]),
                "p_B": float(prob_y_z[z_id, 2]),
                "p_C": float(prob_y_z[z_id, 3]),
                "n_units": int((z_hat == z_id).sum()),
            }
        )
    pd.DataFrame(z_affiliation).to_csv(os.path.join(args.output_dir, "z_macro_affiliation.csv"), index=False)

    metrics_rows = []

    raw_path = os.path.join(exports_dir, "raw_embeddings.npy")
    if os.path.isfile(raw_path):
        raw = np.load(raw_path)
        metrics_rows.append(build_geometry_metrics_row(raw, macro_labels, method=args.method_raw))
    elif args.emb_csv:
        raw = load_raw_embeddings(metadata, args.emb_csv)
        metrics_rows.append(build_geometry_metrics_row(raw, macro_labels, method=args.method_raw))

    metrics_rows.append(
        build_geometry_metrics_row(projected_source, macro_labels, method="MALT_source")
    )
    metrics_rows.append(
        build_geometry_metrics_row(projected_adapted, macro_labels, method="MALT_adapted")
    )

    metrics_table = pd.DataFrame(metrics_rows, columns=METRICS_TABLE_COLUMNS)
    metrics_table.to_csv(os.path.join(args.output_dir, "metrics_table.csv"), index=False)
    metrics_table.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)

    save_json(
        {"metrics_table": metrics_rows, "malt_diagnostics": malt_diagnostics},
        os.path.join(args.output_dir, "metrics_summary.json"),
    )


def main() -> None:
    run_evaluate(parse_args())


if __name__ == "__main__":
    main()
