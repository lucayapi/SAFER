#!/bin/bash
# SupCon : 8 architectures x 4 paramètres LR, paramètres contrastifs fixes.
# Usage : VARIANTS="full_yes full_no" REFIT=false sbatch jobs/tune_supcon_macro_ft.sh

#SBATCH --job-name=tune_supcon_macro_ft
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
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
source "${TEXT_JOBS_DIR}/_tune_common.sh"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
tune_architecture_job_run tune_supcon_macro_ft.py configs/tuning/supcon_macro_ft_grid.yaml
