"""Entraînement SoftTriple → resultats/softtriple/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_methods.train import run_contrastive_method

if __name__ == "__main__":
    raise SystemExit(run_contrastive_method("softtriple", sys.argv[1:]))
