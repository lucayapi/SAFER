"""Lance le tuning supervised_macro_geo_ft (grille CE + λ·L_geo + K-fold CV)."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from supervised_macro_ft.tuning import run_supervised_macro_ft_tuning

DEFAULT_GRID = "configs/tuning/supervised_macro_geo_ft_grid.yaml"


def main() -> None:
    argv = sys.argv[1:]
    if not any(a.startswith("--grid-config") for a in argv):
        argv = ["--grid-config", DEFAULT_GRID, *argv]
    raise SystemExit(run_supervised_macro_ft_tuning(argv))


if __name__ == "__main__":
    main()
