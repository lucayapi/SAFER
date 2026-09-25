#!/usr/bin/env bash
# Exact global BN + empirical recurrent scenarios. Run from text/:
# DATASET=caou sbatch jobs/run_recurrent_scenarios_bn_exact.sh
# DATASET=caou REESTIMATE=1 sbatch jobs/run_recurrent_scenarios_bn_exact.sh

#SBATCH --job-name=exact_bn
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

DATASET="${DATASET:?Set DATASET, for example caou}"
CONFIG_PATH="${CONFIG_PATH:-recurrent_scenarios/config.yaml}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_audit/${DATASET}}"
ARGS=(recurrent_scenarios/run_exact_bn_analysis.py --config "${CONFIG_PATH}" --dataset "${DATASET}" --run-dir "${RUN_DIR}")
if [[ "${REESTIMATE:-0}" == "1" ]]; then
  ARGS+=(--reestimate)
fi
echo "Dataset=${DATASET}"
echo "RunDir=${RUN_DIR}"
echo "Reestimate=${REESTIMATE:-0}"
echo "Launching: python -u ${ARGS[*]}"
python -u "${ARGS[@]}"
