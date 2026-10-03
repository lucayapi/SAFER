#!/bin/bash
# Soumet les 40 exécutions CamemBERT (8 conditions x 5 graines).
# À lancer après les sélections source-only et après mise à jour de la recette :
#   cd ~/SAFER/text
#   bash jobs/submit_camembert_replications.sh
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/camembert_source_factorial.yaml}"
TASK_COUNT="$(python scripts/run_replication.py --config "${CONFIG}" --print-task-count)"
if [[ "${TASK_COUNT}" != "40" ]]; then
  echo "Expected 40 CamemBERT runs, got ${TASK_COUNT}. Check n_seeds and models in ${CONFIG}." >&2
  exit 2
fi
MAX_PARALLEL="$(python scripts/run_replication.py --config "${CONFIG}" --print-max-parallel-seeds)"
echo "Submitting ${TASK_COUNT} CamemBERT runs from ${CONFIG} (${MAX_PARALLEL} concurrent seed jobs)."
sbatch --export=ALL,CONFIG="${CONFIG}" \
  --array="0-$((TASK_COUNT - 1))%${MAX_PARALLEL}" \
  "${TEXT_JOBS_DIR}/replicate_experiment.sh"
