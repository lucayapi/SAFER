"""Command-line entry point for the recurrent-scenario audit pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from scenario_pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recurrent accident scenario protocol.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset", help="Dataset registered in config.yaml (default: caou).")
    parser.add_argument("--debug", action="store_true", help="Use the short repetition count from config.")
    args = parser.parse_args()
    output_dir = run_pipeline(args.config.resolve(), debug=True if args.debug else None, dataset_id=args.dataset)
    print(f"Run written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
