"""Grid search Batch Triplet (sélection δ_macro = 100×η²)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_methods.tuning import run_tuning

if __name__ == "__main__":
    raise SystemExit(run_tuning("batch_triplet", sys.argv[1:]))
