import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from malt_text.malt_metrics import (
    anchor_drift_metrics,
    geometry_metrics,
    probability_summary,
)
from scgm_text.dataset_text_embeddings import ID2LABEL, LABEL2ID
from scgm_text.metrics import (
    balanced_accuracy,
    calinski_harabasz_score_safe,
    compute_confusion_matrix,
    davies_bouldin_score_safe,
    macro_f1,
    silhouette_score_safe,
)
from scgm_text.utils_io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported MALT transfer outputs.")
    parser.add_argument("--exports_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument(
        "--subtype_col",
        type=str,
        default="pred_subtype",
        help="Column for ARI vs z_hat (diagnostic; often pseudo-labels).",
    )
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

    summary = {
        "p0_distribution": probability_summary(p0),
        "pt_distribution": probability_summary(pt),
        "change_rate": float((metadata["p0_macro_id"].to_numpy() != metadata["pt_macro_id"].to_numpy()).mean()),
    }
    summary["geometry_source"] = geometry_metrics(projected_source)
    summary["geometry_adapted"] = geometry_metrics(projected_adapted)
    summary["anchor_drift"] = anchor_drift_metrics(mu_source, mu_target)
    summary["clustering"] = {
        "silhouette_z": silhouette_score_safe(projected_adapted, z_hat),
        "davies_bouldin_z": davies_bouldin_score_safe(projected_adapted, z_hat),
        "calinski_harabasz_z": calinski_harabasz_score_safe(projected_adapted, z_hat),
    }

    transition = transition_matrix(
        metadata["p0_macro_id"].to_numpy(dtype=np.int64),
        metadata["pt_macro_id"].to_numpy(dtype=np.int64),
    )
    transition.to_csv(os.path.join(args.output_dir, "macro_transition_matrix.csv"))

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

    # ARI global (vue « micro » : tous les segments avec sous-type renseigné, un seul score sur l'ensemble)
    subtype_col = (args.subtype_col or "").strip()
    if subtype_col and subtype_col in metadata.columns:
        st = metadata[subtype_col].astype(str).str.strip()
        valid_subtype = st.notna() & (st != "") & (st.str.lower() != "nan")
        n_valid = int(valid_subtype.sum())
        if n_valid >= 2:
            z_valid = z_hat[valid_subtype.to_numpy()]
            enc = LabelEncoder()
            y_subtype = enc.fit_transform(st.loc[valid_subtype])
            n_z = len(np.unique(z_valid))
            n_st = len(np.unique(y_subtype))
            if n_z >= 2 and n_st >= 2:
                ari_micro = float(adjusted_rand_score(y_subtype, z_valid))
                summary["ari_z_vs_pred_subtype_micro"] = ari_micro
                summary["ari_subtype_n_segments"] = n_valid
                summary["ari_subtype_n_unique_subtype"] = int(n_st)
                summary["ari_subtype_n_unique_z"] = int(n_z)
                pd.DataFrame(
                    [
                        {
                            "metric": "ari_z_vs_pred_subtype_micro",
                            "value": ari_micro,
                            "n_segments": n_valid,
                            "n_unique_pred_subtype": int(n_st),
                            "n_unique_z_hat": int(n_z),
                            "subtype_col": subtype_col,
                        }
                    ]
                ).to_csv(os.path.join(args.output_dir, "ari_subtype_alignment.csv"), index=False)

    if args.label_col in metadata.columns:
        valid = metadata[args.label_col].notna() & metadata[args.label_col].isin(LABEL2ID.keys())
        if valid.any():
            y_true = metadata.loc[valid, args.label_col].map(LABEL2ID).to_numpy(dtype=np.int64)
            p0_pred = metadata.loc[valid, "p0_macro_id"].to_numpy(dtype=np.int64)
            pt_pred = metadata.loc[valid, "pt_macro_id"].to_numpy(dtype=np.int64)
            summary["diagnostic_p0"] = {
                "macro_f1": macro_f1(y_true, p0_pred),
                "balanced_accuracy": balanced_accuracy(y_true, p0_pred),
            }
            summary["diagnostic_pt"] = {
                "macro_f1": macro_f1(y_true, pt_pred),
                "balanced_accuracy": balanced_accuracy(y_true, pt_pred),
            }
            labels = [0, 1, 2, 3]
            pd.DataFrame(
                compute_confusion_matrix(y_true, p0_pred, labels=labels),
                index=["A0", "A1", "B", "C"],
                columns=["A0", "A1", "B", "C"],
            ).to_csv(os.path.join(args.output_dir, "confusion_matrix_p0.csv"))
            pd.DataFrame(
                compute_confusion_matrix(y_true, pt_pred, labels=labels),
                index=["A0", "A1", "B", "C"],
                columns=["A0", "A1", "B", "C"],
            ).to_csv(os.path.join(args.output_dir, "confusion_matrix_pt.csv"))

    save_json(summary, os.path.join(args.output_dir, "metrics_summary.json"))
    pd.json_normalize(summary, sep="_").to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    pd.DataFrame([summary["geometry_source"]]).to_csv(os.path.join(args.output_dir, "geometry_summary.csv"), index=False)
    pd.DataFrame([summary["clustering"]]).to_csv(os.path.join(args.output_dir, "clustering_metrics.csv"), index=False)
    pd.DataFrame([summary["anchor_drift"]]).to_csv(os.path.join(args.output_dir, "anchor_drift.csv"), index=False)


def main() -> None:
    run_evaluate(parse_args())


if __name__ == "__main__":
    main()
