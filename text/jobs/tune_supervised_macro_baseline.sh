#!/bin/bash
# Tuning baseline sklearn (LR / RF / XGB) — grille YAML → best HP → preds train + OOD.
#
# Usage :
#   cd ~/SAFER/text && sbatch jobs/tune_supervised_macro_baseline.sh
#
# Variables optionnelles :
#   GRID_CONFIG=configs/tuning/supervised_macro_baseline_grid.yaml
#
# Sorties principales :
#   output_test/metallurgie/supervised_baseline/tuning/
#     grid_summary.csv, results_summary.csv, best_combo.json, best_hyperparams.json
#   output_test/btp/supervised_baseline_tuned/transfer/
#     source_macro_predictions.csv (+ models/<model>/source_*)
#   output_test/<corpus>/supervised_baseline_tuned/transfer/
#     target_macro_predictions.csv, all_models_test_metrics.csv, models/<model>/…

#SBATCH --job-name=tune_macro_base
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

GRID_CONFIG="${GRID_CONFIG:-configs/tuning/supervised_macro_baseline_grid.yaml}"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "[tune_macro_base] GRID_CONFIG=${GRID_CONFIG}"
echo "[tune_macro_base] sklearn CPU-bound (GPU alloué pour slot cluster, non requis par le code)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python scripts/tune_supervised_macro_baseline.py --config "${GRID_CONFIG}"

echo "[tune_macro_base] terminé $(date -Iseconds)"
echo "Voir : output_test/metallurgie/supervised_baseline/tuning/results_summary.csv"
