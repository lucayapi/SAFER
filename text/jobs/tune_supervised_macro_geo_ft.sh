#!/bin/bash
# Grille supervised_macro_geo_ft (K-fold CV, CE + λ·L_geo) + fit final du meilleur combo.
#
# Usage : cd ~/SAFER/text && sbatch jobs/tune_supervised_macro_geo_ft.sh
# Variables :
#   GRID_CONFIG=configs/tuning/supervised_macro_geo_ft_grid.yaml
#   TEST_CORPUS=metallurgie
#   MAX_COMBOS=4            # limite la grille (smoke test)
#   SKIP_FINAL_FIT=1        # CV seulement
#   SEED=42
#
# Sorties : output/supervised_macro_geo_ft/tuning/grid_summary.csv, best_combo.json
#           puis output/supervised_macro_geo_ft/checkpoints/best_model (si SKIP_FINAL_FIT=0)

#SBATCH --job-name=tune_macro_geo_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi
# shellcheck source=jobs/_tune_common.sh
source "${TEXT_JOBS_DIR}/_tune_common.sh"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"

export PYTHONUNBUFFERED=1

tune_job_run tune_supervised_macro_geo_ft.py configs/tuning/supervised_macro_geo_ft_grid.yaml
