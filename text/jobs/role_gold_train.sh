#!/usr/bin/env bash
#SBATCH --job-name=role-gold-train
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-53%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_gold_env.sh"
default_run_name=safer_role_gold_v3
if [[ "${ROLE_GOLD_EXPERIMENT:-}" == *_equalized ]]; then default_run_name=role_gold_equalized; fi
run_root="${ROLE_GOLD_RUN_ROOT:-${SCRATCH:-$PWD/..}/${default_run_name}}"
mkdir -p "${run_root}/${ROLE_GOLD_EXPERIMENT:?}/.locks"
exec 9>"${run_root}/${ROLE_GOLD_EXPERIMENT}/.locks/train_${SLURM_ARRAY_TASK_ID:?}.lock"
flock -x 9
python -u scripts/role_gold_campaign.py train --experiment "${ROLE_GOLD_EXPERIMENT:?}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
