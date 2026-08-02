"""Tuning Batch Triplet par architecture et régression logistique."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_methods.architecture_tuning import run_architecture_tuning


def main() -> None:
    raise SystemExit(run_architecture_tuning("batch_triplet", sys.argv[1:]))


if __name__ == "__main__":
    main()
