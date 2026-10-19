#!/usr/bin/env bash
# One-step official LoRA smoke, adapter reload, token/offset and probability checks.
#SBATCH --job-name=rq1-e2x-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_e2_crossfit_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_e2_crossfit_env.sh" smoke
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_e2_crossfit_env.sh" smoke
fi
python -u "${RQ1_E2X_SCRIPT}" smoke --bundle "${RQ1_FINAL_BUNDLE}" \
  --out "${RQ1_E2X_ROOT}" --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
