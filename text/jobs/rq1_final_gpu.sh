#!/usr/bin/env bash
# Six rotations LODO, at most two GPU tasks at once.
#SBATCH --job-name=rq1-final-lodo
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
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_final_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_final_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_final_env.sh"
fi
: "${SLURM_ARRAY_TASK_ID:?Use the Slurm array}"
python -u scripts/rq1_final_lodo.py fit \
  --bundle "${RQ1_FINAL_BUNDLE}" --run-root "${RQ1_FINAL_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" --task-index "${SLURM_ARRAY_TASK_ID}"

