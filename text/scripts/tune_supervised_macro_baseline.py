"""Lance le tuning de la baseline supervisée sklearn (notebook 07b)."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from macro_transfer.supervised_baseline_tuning import run_supervised_baseline_tuning_cli


def main() -> None:
    raise SystemExit(run_supervised_baseline_tuning_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
