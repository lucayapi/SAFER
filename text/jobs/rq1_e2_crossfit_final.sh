#!/usr/bin/env bash
# Fit each final M3 on its source A, then extract V and target-test features.
#SBATCH --job-name=rq1-e2x-final
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_e2_crossfit_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_e2_crossfit_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_e2_crossfit_env.sh"
fi
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
python -u "${RQ1_E2X_SCRIPT}" final --bundle "${RQ1_FINAL_BUNDLE}" \
  --out "${RQ1_E2X_ROOT}" --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --target-index "${SLURM_ARRAY_TASK_ID}"
