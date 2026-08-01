#!/bin/bash
# Soumet les 4 jobs de tuning (grilles configs/tuning/*_grid.yaml).
# Lancer depuis text/jobs/ : bash submit_tuning_all.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

sbatch tune_scgm_text.sh
sbatch tune_batch_triplet.sh
sbatch tune_softtriple.sh
sbatch tune_supcon.sh
sbatch tune_supervised_macro_baseline.sh
echo "Jobs tuning soumis. Suivi : squeue -u \$USER"
echo "Variables communes : TEST_CORPUS, MAX_COMBOS, SKIP_FINAL_FIT, SEED, GRID_CONFIG"
echo "Baseline sklearn : GRID_CONFIG=configs/tuning/supervised_macro_baseline_grid.yaml"
