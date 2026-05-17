import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from metrics.geometry import METRICS_TABLE_COLUMNS, build_geometry_metrics_row
from safer_core.io import save_metrics_geometry
from safer_core.paths import layout_method_output, resolve_output_dir
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
    parser = argparse.ArgumentParser(description="Evaluate exported SCGM text outputs (eta2 geometry).")
    parser.add_argument("--exports_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="pred_label")
    parser.add_argument("--emb_csv", type=str, default=None, help="Encoder embedding CSV for raw row.")
    parser.add_argument("--method_raw", type=str, default="Embedding brut")
    parser.add_argument("--method_scgm", type=str, default="SCGM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    metadata = pd.read_csv(os.path.join(args.exports_dir, "metadata_with_predictions.csv"))
    projected = np.load(os.path.join(args.exports_dir, "projected_embeddings.npy"))
    macro_labels = metadata[args.label_col].to_numpy()

    rows = []

    raw_path = os.path.join(args.exports_dir, "raw_embeddings.npy")
    if os.path.isfile(raw_path):
        raw = np.load(raw_path)
        if raw.shape[0] != len(metadata):
            raise ValueError("raw_embeddings.npy row count does not match metadata.")
    elif args.emb_csv:
        raw = load_raw_embeddings(metadata, args.emb_csv)
    else:
        raw = None

    if raw is not None:
        rows.append(
            build_geometry_metrics_row(raw, macro_labels, method=args.method_raw)
        )

    rows.append(
        build_geometry_metrics_row(projected, macro_labels, method=args.method_scgm)
    )

    table = pd.DataFrame(rows, columns=METRICS_TABLE_COLUMNS)
    layout = layout_method_output("scgm_text", resolve_output_dir("scgm_text", args.output_dir))
    metrics_dir = layout["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        save_metrics_geometry(row, metrics_dir, stem=f"metrics_geometry_{row.get('method', 'row')}".replace(" ", "_"))
    table.to_csv(metrics_dir / "metrics_table.csv", index=False)
    table.to_csv(metrics_dir / "metrics_geometry.csv", index=False)
    save_json({"metrics_table": rows}, metrics_dir / "metrics_summary.json")
    ensure_dir(args.output_dir)
    table.to_csv(os.path.join(args.output_dir, "metrics_table.csv"), index=False)


if __name__ == "__main__":
    main()
