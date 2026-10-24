#!/usr/bin/env bash
#SBATCH --job-name=rq1-e2d-gpu
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-17%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_e2_disjoint_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_e2_disjoint_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_e2_disjoint_env.sh"
fi
python -u "${RQ1_E2D_SCRIPT}" run --bundle "${RQ1_FINAL_BUNDLE}" \
  --out "${RQ1_E2D_ROOT}" --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --task-index "${SLURM_ARRAY_TASK_ID:?}"
