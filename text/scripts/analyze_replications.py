"""CLI d'analyse locale des sorties rapatriées du Mésocentre."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replication.bootstrap import analyze_replications
from replication.runner import load_replication_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap apparié accident_id des réplications.")
    parser.add_argument("--config", default="output/replication_recipes/replication_config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths = analyze_replications(load_replication_config(args.config), destination=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
