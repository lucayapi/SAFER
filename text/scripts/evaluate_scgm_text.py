import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scgm_text.metrics import (
    accuracy,
    balanced_accuracy,
    c1_c10_by_macro,
    calinski_harabasz_score_safe,
    compute_confusion_matrix,
    davies_bouldin_score_safe,
    macro_f1,
    pca_energy_c1_c10,
    rankme_by_macro,
    rankme_effective_rank,
    silhouette_score_safe,
    subtype_alignment_diagnostics,
)
from scgm_text.utils_io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported SCGM text outputs.")
    parser.add_argument("--exports_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="pred_label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    metadata = pd.read_csv(os.path.join(args.exports_dir, "metadata_with_predictions.csv"))
    projected = np.load(os.path.join(args.exports_dir, "projected_embeddings.npy"))
    z_hat = metadata["z_hat"].to_numpy(dtype=np.int64)

    label_map = {"A0": 0, "A1": 1, "B": 2, "C": 3}
    y_true = metadata[args.label_col].map(label_map).to_numpy(dtype=np.int64)
    y_pred = metadata["pred_macro_id"].to_numpy(dtype=np.int64)

    macro_metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
    }
    confusion = compute_confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    pd.DataFrame(
        confusion,
        index=["A0", "A1", "B", "C"],
        columns=["A0", "A1", "B", "C"],
    ).to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))

    c1_global, c10_global = pca_energy_c1_c10(projected)
    geometry_global = {
        "rankme_global": rankme_effective_rank(projected),
        "c1_global": c1_global,
        "c10_global": c10_global,
    }
    geometry_by_macro = []
    for macro_name, macro_id in label_map.items():
        rankme_macro = rankme_by_macro(projected, y_true).get(macro_id, float("nan"))
        c1_macro, c10_macro = c1_c10_by_macro(projected, y_true).get(macro_id, (float("nan"), float("nan")))
        geometry_by_macro.append(
            {
                "macro": macro_name,
                "rankme": rankme_macro,
                "c1": c1_macro,
                "c10": c10_macro,
            }
        )
    pd.DataFrame(geometry_by_macro).to_csv(os.path.join(args.output_dir, "geometry_by_macro.csv"), index=False)

    clustering_rows = [
        {
            "scope": "global",
            "silhouette": silhouette_score_safe(projected, z_hat),
            "davies_bouldin": davies_bouldin_score_safe(projected, z_hat),
            "calinski_harabasz": calinski_harabasz_score_safe(projected, z_hat),
        }
    ]
    for macro_name, macro_id in label_map.items():
        mask = y_true == macro_id
        if mask.sum() < 3:
            continue
        clustering_rows.append(
            {
                "scope": macro_name,
                "silhouette": silhouette_score_safe(projected[mask], z_hat[mask]),
                "davies_bouldin": davies_bouldin_score_safe(projected[mask], z_hat[mask]),
                "calinski_harabasz": calinski_harabasz_score_safe(projected[mask], z_hat[mask]),
            }
        )
    pd.DataFrame(clustering_rows).to_csv(os.path.join(args.output_dir, "clustering_metrics.csv"), index=False)

    diagnostics = {}
    if "pred_subtype" in metadata.columns:
        diagnostics = subtype_alignment_diagnostics(z_hat, metadata["pred_subtype"].astype(str).to_numpy())
        diagnostics["subtype_diagnostic_note"] = (
            "pred_subtype is exploratory only and is not expert ground truth."
        )

    summary = {**macro_metrics, **geometry_global, **diagnostics}
    save_json(summary, os.path.join(args.output_dir, "metrics_summary.json"))
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)


if __name__ == "__main__":
    main()
