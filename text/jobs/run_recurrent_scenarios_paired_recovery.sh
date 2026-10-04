#!/usr/bin/env bash
# Submit from text/. Default: 500 replicates in 50 array tasks, ten at a time.
# For a pilot: N_REPLICATES=10 OUTPUT_DIR=.../paired_recovery_pilot \
#   sbatch --array=0 jobs/run_recurrent_scenarios_paired_recovery.sh
# Full run: sbatch jobs/run_recurrent_scenarios_paired_recovery.sh
#SBATCH --job-name=paired_recovery
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=0-49%10
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

DATASET="${DATASET:-btp_carpentry_and_joinery}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_audit/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/paired_recovery}"
N_REPLICATES="${N_REPLICATES:-500}"
REPLICATES_PER_TASK="${REPLICATES_PER_TASK:-10}"
FRACTION="${FRACTION:-0.8}"
SEED="${SEED:-2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$((TASK_ID * REPLICATES_PER_TASK))

echo "Paired recovery: dataset=${DATASET}, task=${TASK_ID}, start=${START}, count=${REPLICATES_PER_TASK}, total=${N_REPLICATES}"
python -u recurrent_scenarios/paired_recovery.py run \
  --config "${CONFIG_PATH:-recurrent_scenarios/config.yaml}" \
  --dataset "${DATASET}" --run-dir "${RUN_DIR}" --output "${OUTPUT_DIR}" \
  --replicates "${N_REPLICATES}" --fraction "${FRACTION}" \
  --seed "${SEED}" --start "${START}" --count "${REPLICATES_PER_TASK}"
