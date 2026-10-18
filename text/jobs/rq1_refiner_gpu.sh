#!/usr/bin/env bash
#SBATCH --job-name=rq1-refiner-lodo
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
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_refiner_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_refiner_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_refiner_env.sh"
fi
: "${SLURM_ARRAY_TASK_ID:?Launch with the Slurm array}"
python -u scripts/rq1_refiner.py fit \
  --bundle "${RQ1_FINAL_BUNDLE}" --m3-root "${RQ1_REFINER_M3_ROOT}" \
  --out "${RQ1_REFINER_OUT}" --task-index "${SLURM_ARRAY_TASK_ID}"
