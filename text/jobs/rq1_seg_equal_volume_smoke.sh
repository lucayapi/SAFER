#!/usr/bin/env bash
#SBATCH --job-name=rq1-eq-smoke
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
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_seg_equal_volume_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_seg_equal_volume_env.sh" smoke
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_seg_equal_volume_env.sh" smoke
fi
python -u scripts/rq1_seg_equal_volume.py smoke \
  --source-bundle "${RQ1_EQUAL_SOURCE_BUNDLE}" \
  --reference "${RQ1_EQUAL_REFERENCE}" --run-root "${RQ1_EQUAL_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
