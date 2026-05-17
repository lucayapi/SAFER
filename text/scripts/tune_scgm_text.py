"""Lance le tuning SCGM (grille + K-fold)."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scgm_text.tuning import run_scgm_tuning


def main() -> None:
    raise SystemExit(run_scgm_tuning())


if __name__ == "__main__":
    main()
