"""CLI for the parallel UMAP-HDBSCAN theme-discovery stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scenario_pipeline import run_theme_discovery


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run parallel D_U/S_R recurrent-accident theme discovery."
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
        "--debug",
        action="store_true",
        help="Use the short resampling count from the configuration.",
    )
    parser.add_argument(
        "--reestimate",
        action="store_true",
        help="Ignore discovery caches and recompute all candidates and resamples.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "metrics", "pareto"),
        default="all",
        help="Run all stages, metrics/stability only, or Pareto selection only.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Existing run directory for --stage pareto, or an explicit output directory.",
    )
    args = parser.parse_args()
    output_dir = run_theme_discovery(
        args.config.resolve(),
        debug=True if args.debug else None,
        dataset_id=args.dataset,
        reestimate=args.reestimate,
        stage=args.stage,
        run_dir=args.run_dir,
    )
    print(f"Theme-discovery run written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
