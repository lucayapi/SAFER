#!/bin/bash
# Agrège les 15 réplications terminées et calcule les IC bootstrap OOD.
# Soumettre de préférence avec --dependency=afterok:<arrays des 3 méthodes>.
#SBATCH --job-name=safer-repl-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
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

CONFIG="${CONFIG:-output/replication_recipes/replication_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "Analyse des réplications : ${CONFIG}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  python -u scripts/analyze_replications.py --config "${CONFIG}" --output-dir "${OUTPUT_DIR}"
else
  python -u scripts/analyze_replications.py --config "${CONFIG}"
fi
