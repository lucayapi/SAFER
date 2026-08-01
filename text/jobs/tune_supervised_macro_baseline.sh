#!/usr/bin/env bash
# Tuning baseline sklearn (LR / RF / XGB) — GroupKFold BTP puis OOD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/tune_supervised_macro_baseline.py \
  --config configs/tuning/supervised_macro_baseline_grid.yaml
