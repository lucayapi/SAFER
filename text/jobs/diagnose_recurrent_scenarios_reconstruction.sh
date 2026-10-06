#!/usr/bin/env bash
# Small diagnostic: compare archived UMAP with serial and forked refits.
# Run from text/: sbatch jobs/diagnose_recurrent_scenarios_reconstruction.sh
#SBATCH --job-name=diagnose_scenario_refit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/diagnostic_reconstruction}"
ROLE="${ROLE:-C}"

python -u recurrent_scenarios/diagnose_reconstruction.py \
  --config "${CONFIG_PATH}" --dataset "${DATASET}" \
  --run-dir "${RUN_DIR}" --output "${OUTPUT_DIR}" \
  --role "${ROLE}" --workers 3
