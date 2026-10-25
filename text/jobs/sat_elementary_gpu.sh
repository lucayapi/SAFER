#!/usr/bin/env bash
# Six LODO targets, one training seed; at most three concurrent GPU tasks.
# From SAFER/: sbatch text/jobs/sat_elementary_gpu.sh
#SBATCH --job-name=sat-elementary
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_env.sh"
sat_elementary_setup
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
python -u scripts/sat_elementary_lodo.py fit \
  --bundle "${SAT_LODO_BUNDLE}" --prepared "${SAT_ELEMENTARY_PREPARED}" \
  --run-root "${SAT_LODO_RUN_ROOT}" --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
