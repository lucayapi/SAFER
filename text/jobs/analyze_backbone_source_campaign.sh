#!/bin/bash
# Agrège les 80 répétitions backbone × domaine source et produit les tableaux,
# intervalles bootstrap et figures. À lancer depuis text/ avec :
#   sbatch jobs/analyze_backbone_source_campaign.sh

#SBATCH --job-name=backbone-source-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/backbone_source_factorial.yaml}"
OUTPUT="${OUTPUT:-output/backbone_source_factorial_analysis}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "Config=${CONFIG}"
echo "Output=${OUTPUT}; bootstrap resamples=${N_BOOTSTRAP}; workers=${N_WORKERS}"
python -u scripts/analyze_backbone_source_campaign.py \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-workers "${N_WORKERS}"
