"""Lance le tuning supervised_macro_ft (grille + K-fold CV)."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from supervised_macro_ft.tuning import run_supervised_macro_ft_tuning


def main() -> None:
    raise SystemExit(run_supervised_macro_ft_tuning(sys.argv[1:]))


if __name__ == "__main__":
    main()
