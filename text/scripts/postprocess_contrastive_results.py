"""Post-traitement optionnel : recalcul métriques si embeddings déjà exportés."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from contrastive_methods.config import config_to_resolved_dict, load_contrastive_config
from contrastive_methods.metrics import compute_and_save_geometry_metrics
from safer_core.io import save_config_resolved
from safer_core.paths import layout_method_output
import safer_core.paths as paths_mod

METHODS = ("batch_triplet", "softtriple", "supcon")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recalcule metrics_geometry depuis final_embeddings.csv")
    p.add_argument("--method", type=str, required=True, choices=METHODS)
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def _has_dim_cols(path: Path) -> bool:
    cols = pd.read_csv(path, nrows=0).columns
    return any(str(c).startswith("dim_") for c in cols)


def _resolve_embeddings(results_root: Path) -> Path:
    final = results_root / "embeddings" / "final_embeddings.csv"
    if final.is_file() and _has_dim_cols(final):
        return final
    candidates = [
        p
        for p in sorted(results_root.rglob("*.csv"))
        if p.is_file() and p.stat().st_size > 0 and _has_dim_cols(p)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Aucun CSV dim_* sous {results_root}. Lancez scripts/train_<method>.py d'abord."
        )
    return candidates[0]


def postprocess_contrastive_method(method_name: str, config_path: Optional[str] = None) -> Path:
    cfg = load_contrastive_config(method_name, config_path)
    layout = layout_method_output(method_name, cfg.resolved_output_dir)
    results_root = Path(layout["root"])
    metrics_dir = Path(layout["metrics"])
    emb_path = _resolve_embeddings(results_root)
    dest = results_root / "embeddings" / "final_embeddings.csv"
    if emb_path.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(emb_path.read_bytes())

    compute_and_save_geometry_metrics(dest, cfg, metrics_dir)

    resolved: Dict[str, Any] = config_to_resolved_dict(cfg)
    resolved.update(
        {
            "embeddings_dest": str(dest),
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
