#!/bin/bash
# Soumet l'analyse après les arrays indiqués dans DEPENDENCY.
# Exemple : DEPENDENCY=8069670:8069671:8069672 bash jobs/submit_replication_analysis.sh

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/replication_config.yaml}"
DEPENDENCY="${DEPENDENCY:-}"

if [[ -n "${DEPENDENCY}" ]]; then
  echo "Analyse soumise après succès de : ${DEPENDENCY}"
  sbatch --dependency="afterok:${DEPENDENCY}" --export=ALL,CONFIG="${CONFIG}" \
    "${TEXT_JOBS_DIR}/analyze_replications.sh"
else
  echo "Aucune dépendance : l'analyse démarre dès que Slurm la planifie."
  echo "Pour attendre les arrays : DEPENDENCY=<id_soft>:<id_supcon>:<id_ce> bash $0" >&2
  sbatch --export=ALL,CONFIG="${CONFIG}" "${TEXT_JOBS_DIR}/analyze_replications.sh"
fi
