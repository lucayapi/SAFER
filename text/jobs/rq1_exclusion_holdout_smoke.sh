#!/usr/bin/env bash
#SBATCH --job-name=rq1-exh-smoke
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
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_exclusion_holdout_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_exclusion_holdout_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_exclusion_holdout_env.sh"
fi
python -u "${RQ1_EXCLUSION_HOLDOUT_SCRIPT}" smoke --bundle "${RQ1_FINAL_BUNDLE}" \
  --out "${RQ1_EXCLUSION_HOLDOUT_ROOT}" --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
