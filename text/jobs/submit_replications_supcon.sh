#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/submit_replications_model.sh" ]]; then
  SUBMIT_SCRIPT="${SLURM_SUBMIT_DIR}/jobs/submit_replications_model.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/submit_replications_model.sh" ]]; then
  SUBMIT_SCRIPT="${SLURM_SUBMIT_DIR}/submit_replications_model.sh"
else
  SUBMIT_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_replications_model.sh"
fi
MODEL=supcon_full_yes bash "${SUBMIT_SCRIPT}"
