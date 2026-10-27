#!/usr/bin/env bash
#SBATCH --job-name=role-ce-triplet-train
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-35%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_qwen3_lodo_env.sh"
run_root="${ROLE_QWEN3_RUN_ROOT:-${SCRATCH:-${SLURM_SUBMIT_DIR}}/safer_role_qwen3_narrative_lodo_3epoch_v1}"
mkdir -p "${run_root}/missing_ce_batch_triplet_v1/.locks"
exec 9>"${run_root}/missing_ce_batch_triplet_v1/.locks/train_${SLURM_ARRAY_TASK_ID:?}.lock"
flock -x 9
python -u scripts/role_qwen3_lodo_missing.py train --task-index "${SLURM_ARRAY_TASK_ID}"
