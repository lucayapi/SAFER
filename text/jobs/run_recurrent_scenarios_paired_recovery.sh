#!/usr/bin/env bash
# Submit from text/. Default: 500 one-replicate array tasks, ten at a time.
# For a pilot: N_REPLICATES=10 OUTPUT_DIR=.../paired_recovery_pilot \
#   sbatch --array=0 jobs/run_recurrent_scenarios_paired_recovery.sh
# Full run: sbatch jobs/run_recurrent_scenarios_paired_recovery.sh
#SBATCH --job-name=paired_recovery
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --array=0-499%10
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
CONFIG_PATH="${CONFIG_PATH:-recurrent_scenarios/config.yaml}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_audit/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/paired_recovery}"
mapfile -t PAIR_CONFIG < <(python -u recurrent_scenarios/paired_recovery.py settings \
  --config "${CONFIG_PATH}" --dataset "${DATASET}" --run-dir "${RUN_DIR}")
N_REPLICATES="${N_REPLICATES:-${PAIR_CONFIG[0]}}"
FRACTION="${FRACTION:-${PAIR_CONFIG[1]}}"
SEED="${SEED:-${PAIR_CONFIG[2]}}"
REPLICATES_PER_TASK="${REPLICATES_PER_TASK:-${PAIR_CONFIG[3]}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$((TASK_ID * REPLICATES_PER_TASK))

echo "Paired recovery: dataset=${DATASET}, task=${TASK_ID}, start=${START}, count=${REPLICATES_PER_TASK}, total=${N_REPLICATES}"
python -u recurrent_scenarios/paired_recovery.py run \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" --run-dir "${RUN_DIR}" --output "${OUTPUT_DIR}" \
  --replicates "${N_REPLICATES}" --fraction "${FRACTION}" \
  --seed "${SEED}" --start "${START}" --count "${REPLICATES_PER_TASK}"
