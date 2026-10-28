#!/usr/bin/env bash
#SBATCH --job-name=role-e2e-reduced
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-53%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_narrative_e2e_retrain_env.sh"
mkdir -p "${ROLE_QWEN3_RUN_ROOT}/.locks"
exec 9>"${ROLE_QWEN3_RUN_ROOT}/.locks/e2e_${SLURM_ARRAY_TASK_ID:?}.lock"
flock -x 9
python -u scripts/role_narrative_e2e_retrain.py train --task-index "${SLURM_ARRAY_TASK_ID}"
