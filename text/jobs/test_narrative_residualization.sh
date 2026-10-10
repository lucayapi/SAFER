#!/bin/bash
# 4 independent CPU array tasks, all may run concurrently: two encoders x two sources.
# Submit from text/: sbatch jobs/test_narrative_residualization.sh

#SBATCH --job-name=narrative-resid
#SBATCH --partition=normal
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=30
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

TASKS=("qwen3 btp" "qwen3 metallurgie" "multilingual_e5_large btp" "multilingual_e5_large metallurgie")
read -r BACKBONE SOURCE <<< "${TASKS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT="${OUTPUT:-output/narrative_residualization}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-30}"
THREADS=$((ALLOCATED_CPUS - 1))
if (( THREADS < 1 )); then
  THREADS=1
fi

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local} ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}"
echo "Backbone=${BACKBONE}; source=${SOURCE}; allocated_cpus=${ALLOCATED_CPUS}; compute_threads=${THREADS}; output=${OUTPUT}"

python -u scripts/test_narrative_residualization.py \
  --backbone "${BACKBONE}" \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --threads "${THREADS}" \
  --outer-folds 3 \
  --inner-folds 3 \
  --selection-folds 3 \
  --bootstrap 1000
