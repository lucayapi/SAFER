"""Post-traitement standardisé après entraînement contrastif legacy."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics.geometry import build_geometry_metrics_row
from safer_core.data_loading import load_metadata_with_embeddings
from safer_core.io import flatten_method_config, load_yaml, save_config_resolved, save_metrics_geometry
from safer_core.paths import layout_method_output
import safer_core.paths as paths_mod

METHOD_DISPLAY = {
    "batch_triplet": "Batch Triplet",
    "softtriple": "SoftTriple",
    "supcon": "SupCon",
}

METHOD_TO_LEGACY = {
    "batch_triplet": ("batchTripplet", "ftemb_script.py"),
    "softtriple": ("Softriple", "ftemb_script_softriple.py"),
    "supcon": ("Supcon", "ftemb_script_supcon.py"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-traitement résultats contrastifs → resultats/<method>/")
    p.add_argument("--method", type=str, required=True, choices=list(METHOD_TO_LEGACY.keys()))
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def _has_dim_cols(path: Path) -> bool:
    cols = pd.read_csv(path, nrows=0).columns
    return any(str(c).startswith("dim_") for c in cols)


def _find_embedding_csvs(root: Path) -> List[Path]:
    found: List[Path] = []
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.csv")):
        if path.stat().st_size > 0 and _has_dim_cols(path):
            found.append(path)
    return found


def _resolve_embeddings_source(method_name: str, results_root: Path, cfg: Dict[str, Any]) -> Path:
    label = str(cfg.get("label_col", "pred_label"))
    candidates: List[Path] = []

    final_dir = results_root / "best_model_full_data"
    candidates.extend(_find_embedding_csvs(final_dir))

    summary = results_root / "grid_search_summary.csv"
    if summary.is_file():
        df = pd.read_csv(summary)
        if len(df) and "combo_id" in df.columns:
            combo_id = str(df.iloc[0]["combo_id"])
            legacy_folder, _ = METHOD_TO_LEGACY[method_name]
            legacy_grid = (
                paths_mod.TEXT_ROOT
                / "legacy"
                / "contrastive_method_v0"
                / legacy_folder
                / "fnembeddings_grid"
            )
            pattern = f"embeddings__{label}__{combo_id}.csv"
            if legacy_grid.is_dir():
                for p in legacy_grid.glob(pattern):
                    candidates.append(p)

    candidates.extend(_find_embedding_csvs(results_root))

    legacy_folder, _ = METHOD_TO_LEGACY[method_name]
    legacy_grid = (
        paths_mod.TEXT_ROOT / "legacy" / "contrastive_method_v0" / legacy_folder / "fnembeddings_grid"
    )
    if legacy_grid.is_dir():
        for p in sorted(legacy_grid.glob(f"embeddings__{label}__*.csv"), reverse=True):
            candidates.append(p)

    seen = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen and p.is_file():
            seen.add(key)
            unique.append(p)

    if not unique:
        raise FileNotFoundError(
            f"Aucun CSV dim_* trouvé pour {method_name} sous {results_root} ou fnembeddings_grid legacy."
        )
    return unique[0]


def postprocess_contrastive_method(method_name: str, config_path: Optional[str] = None) -> Path:
    cfg_path = paths_mod.TEXT_ROOT / (config_path or f"configs/methods/{method_name}.yaml")
    raw = load_yaml(cfg_path)
    cfg = flatten_method_config(raw)
    layout = layout_method_output(method_name, cfg.get("output_dir", f"resultats/{method_name}"))
    results_root = Path(layout["root"])
    emb_dir = layout["embeddings"]
    metrics_dir = layout["metrics"]
    emb_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    source = _resolve_embeddings_source(method_name, results_root, cfg)
    dest = emb_dir / "final_embeddings.csv"
    shutil.copy2(source, dest)

    data_csv = paths_mod.TEXT_ROOT / cfg.get("dataset_path", "dataset/data_btp.csv")
    merged, dim_cols = load_metadata_with_embeddings(
        str(data_csv),
        str(dest),
        label_col=str(cfg.get("label_col", "pred_label")),
    )
    emb = merged[dim_cols].to_numpy(dtype=float)
    labels = merged[str(cfg.get("label_col", "pred_label"))].to_numpy()

    display = METHOD_DISPLAY.get(method_name, method_name)
    row = build_geometry_metrics_row(emb, labels, method=display, l2_normalize=True)
    save_metrics_geometry(row, metrics_dir)

    resolved: Dict[str, Any] = dict(cfg)
    resolved.update(
        {
            "method_name": method_name,
            "embeddings_source": str(source),
            "embeddings_dest": str(dest),
            "grid_summary": str(results_root / "grid_search_summary.csv")
            if (results_root / "grid_search_summary.csv").is_file()
            else None,
            "postprocessed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_config_resolved(resolved, results_root)
    print(f"[postprocess] {method_name}: embeddings → {dest}")
    print(f"[postprocess] metrics → {metrics_dir / 'metrics_geometry.csv'}")
    return dest


def main() -> None:
    args = parse_args()
    postprocess_contrastive_method(args.method, args.config)


if __name__ == "__main__":
    main()
