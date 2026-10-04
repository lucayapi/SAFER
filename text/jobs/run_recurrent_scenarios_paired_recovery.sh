#!/usr/bin/env bash
# Submit one Slurm job from text/. It runs all paired replicates internally,
# using the CPUs allocated to the job.
#SBATCH --job-name=paired_recovery
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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
ARGS=(recurrent_scenarios/paired_recovery.py run
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}" --run-dir "${RUN_DIR}" --output "${OUTPUT_DIR}")
if [[ -n "${N_REPLICATES:-}" ]]; then ARGS+=(--replicates "${N_REPLICATES}"); fi
if [[ -n "${FRACTION:-}" ]]; then ARGS+=(--fraction "${FRACTION}"); fi
if [[ -n "${SEED:-}" ]]; then ARGS+=(--seed "${SEED}"); fi
if [[ -n "${N_WORKERS:-}" ]]; then ARGS+=(--workers "${N_WORKERS}"); fi

echo "Paired recovery: one job, ${SLURM_CPUS_PER_TASK:-1} allocated CPUs"
python -u "${ARGS[@]}"

python -u recurrent_scenarios/paired_recovery.py summarise \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" --run-dir "${RUN_DIR}" --output "${OUTPUT_DIR}"
