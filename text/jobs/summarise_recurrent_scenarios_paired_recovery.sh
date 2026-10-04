#!/usr/bin/env bash
# Submit after the array finishes; see paired_recovery_README.md.
#SBATCH --job-name=paired_summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
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

DATASET="${DATASET:-btp_carpentry_and_joinery}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_audit/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/paired_recovery}"

python -u recurrent_scenarios/paired_recovery.py summarise \
  --config "${CONFIG_PATH:-recurrent_scenarios/config.yaml}" \
  --dataset "${DATASET}" --run-dir "${RUN_DIR}" --output "${OUTPUT_DIR}"
