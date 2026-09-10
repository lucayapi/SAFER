#!/bin/bash
# Soumet les seeds d'une seule méthode, pour limiter le stockage GPU/HPC.
# Usage : MODEL=softtriple_full_yes bash jobs/submit_replications_model.sh

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/replication_config.yaml}"
MODEL="${MODEL:?Définir MODEL, par exemple softtriple_full_yes}"
SEED_COUNT="$(python scripts/run_replication.py --config "${CONFIG}" --print-seed-count)"
MAX_PARALLEL_SEEDS="$(python scripts/run_replication.py --config "${CONFIG}" --print-max-parallel-seeds)"
if [[ "${SEED_COUNT}" -lt 1 ]]; then
  echo "Aucune seed définie." >&2
  exit 2
fi
echo "Soumission de ${SEED_COUNT} seeds pour ${MODEL}, au plus ${MAX_PARALLEL_SEEDS} simultanée(s); nettoyage automatique activé après chaque seed complète."
sbatch --array="0-$((SEED_COUNT - 1))%${MAX_PARALLEL_SEEDS}" --export=ALL,CONFIG="${CONFIG}",MODEL="${MODEL}" "${TEXT_JOBS_DIR}/replicate_model_experiment.sh"
