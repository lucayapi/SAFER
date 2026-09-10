#!/bin/bash
# Une tâche Slurm = une méthode gelée × une seed d'entraînement.
# Soumis uniquement par submit_replications.sh (job array).
#SBATCH --job-name=safer-repl
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/replication_config.yaml}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID absent: utiliser submit_replications.sh}"
echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local} TASK=${TASK_INDEX}"
python -u scripts/run_replication.py --config "${CONFIG}" --task-index "${TASK_INDEX}"
