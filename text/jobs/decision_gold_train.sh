#!/usr/bin/env bash
#SBATCH --job-name=decision-gold-fit
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?}"
if [[ -d "${submit}/text/jobs" ]]; then env_script="${submit}/text/jobs/_decision_gold_env.sh"
else env_script="${submit}/jobs/_decision_gold_env.sh"; fi
source "${env_script}"
python -u scripts/decision_gold_models.py run --decision "${DECISION_GOLD_DECISION:?}" --task-index "${SLURM_ARRAY_TASK_ID:?}"
