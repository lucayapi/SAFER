#!/usr/bin/env bash
# One job, policies processed successively, replicates parallelised within each.
#SBATCH --job-name=paired_size_sensitivity
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DATASET="${DATASET:-btp_carpentry_and_joinery}"
CONFIG_PATH="${CONFIG_PATH:-recurrent_scenarios/config.yaml}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_audit/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/paired_parameter_sensitivity}"
ARGS=(recurrent_scenarios/paired_sensitivity.py --config "$CONFIG_PATH"
      --dataset "$DATASET" --run-dir "$RUN_DIR" --output "$OUTPUT_DIR")
if [[ -n "${N_REPLICATES:-}" ]]; then ARGS+=(--replicates "$N_REPLICATES"); fi
if [[ -n "${N_WORKERS:-}" ]]; then ARGS+=(--workers "$N_WORKERS"); fi
python -u "${ARGS[@]}"
