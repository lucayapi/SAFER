#!/bin/bash
# Grille SupCon (K-fold) + fit final 100 % BTP du meilleur combo.
#
# Usage : cd ~/SAFER/text && sbatch jobs/tune_supcon.sh
# Variables :
#   GRID_CONFIG=configs/tuning/supcon_grid.yaml
#   TEST_CORPUS=metallurgie
#   MAX_COMBOS=8
#   SKIP_FINAL_FIT=1
#   SEED=42
#
# Sorties : output/supcon/tuning/ puis output/supcon/checkpoints/best_model

#SBATCH --job-name=tune_supcon
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
  # shellcheck source=jobs/_bootstrap.sh
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  # shellcheck source=jobs/_bootstrap.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi
# shellcheck source=jobs/_tune_common.sh
source "${TEXT_JOBS_DIR}/_tune_common.sh"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"

tune_job_run tune_supcon.py configs/tuning/supcon_grid.yaml
