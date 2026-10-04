#!/bin/bash
# Aggregate completed CamemBERT replications and compute target-level,
# role-level, and paired bootstrap summaries. Submit from text/ with:
#   sbatch jobs/analyze_camembert_source_campaign.sh

#SBATCH --job-name=camembert-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-output/replication_recipes/camembert_source_factorial.yaml}"
OUTPUT="${OUTPUT:-output/camembert_source_factorial_analysis}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-3}}"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "Config=${CONFIG}"
echo "Output=${OUTPUT}; bootstrap resamples=${N_BOOTSTRAP}; workers=${N_WORKERS}"
python -u scripts/analyze_backbone_source_campaign.py \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-workers "${N_WORKERS}"
