"""CLI for the parallel UMAP-HDBSCAN theme-discovery stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scenario_pipeline import run_theme_discovery


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run recurrent-accident theme discovery with Pareto screening "
            "and geometric knee-point selection."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset registered in config.yaml (default: caou).",
    )
    parser.add_argument(
        "--reestimate",
        action="store_true",
        help="Ignore discovery caches and recompute candidates, resampling and seeds.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "metrics", "select", "seed", "evaluate"),
        default="all",
        help=(
            "all = metrics+select+seed; metrics = DBCV/S_R only; "
            "select = Pareto + geometric knee + materialize; "
            "seed = UMAP seed sensitivity. "
            "evaluate is deprecated (alias for select)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Existing run directory for --stage select/seed, or an explicit output directory.",
    )
    args = parser.parse_args()
    output_dir = run_theme_discovery(
        args.config.resolve(),
        dataset_id=args.dataset,
        reestimate=args.reestimate,
        stage=args.stage,
        run_dir=args.run_dir,
    )
    print(f"Theme-discovery run written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
