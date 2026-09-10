#!/bin/bash
# Soumet toutes les réplications définies dans la configuration unique.
# Usage depuis text/: bash jobs/submit_replications.sh
# Pour une autre recette : CONFIG=... bash jobs/submit_replications.sh

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/replication_config.yaml}"
TASK_COUNT="$(python scripts/run_replication.py --config "${CONFIG}" --print-task-count)"
if [[ "${TASK_COUNT}" -lt 1 ]]; then
  echo "Aucune tâche de réplication définie." >&2
  exit 2
fi
echo "Soumission de ${TASK_COUNT} tâches (modèles × seeds) avec ${CONFIG}"
sbatch --array="0-$((TASK_COUNT - 1))" "${TEXT_JOBS_DIR}/replicate_experiment.sh"
