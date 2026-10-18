#!/usr/bin/env bash
# Six independent LODO fits; checkpoint and threshold selected on source validation.
# From SAFER/: sbatch text/jobs/sat_lodo_gold_v3_checkpoint_gpu.sh
#SBATCH --job-name=sat-v3-ckpt
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
fi
export SAT_LODO_RUN_ROOT="${SCRATCH:-$HOME}/safer_sat_lodo/runs/sat12l_sm_gold_v3_rq1_checkpoint_selection_v1"
sat_gold_v3_setup
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
# An explicit sbatch restart discards only this array task's previous target
# outputs, including incomplete checkpoints. The smoke marker and other
# target runs remain untouched.
python -u scripts/reset_sat_checkpoint_target.py \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
python -u scripts/sat_lodo_checkpoint_select.py fit \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
