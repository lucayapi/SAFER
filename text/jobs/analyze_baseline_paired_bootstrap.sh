#!/bin/bash
# Compare les deux baselines déjà évaluées, sans réentraînement.
# Produit les IC bootstrap appariés pour Frozen embeddings + LR - TF-IDF + LR.
#SBATCH --job-name=safer-baseline-paired
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

TFIDF_ROOT="${TFIDF_ROOT:-output/tfidf_baseline}"
FROZEN_ROOT="${FROZEN_ROOT:-output_test}"
OUTPUT_DIR="${OUTPUT_DIR:-output/baseline_paired_bootstrap}"
N_RESAMPLES="${N_RESAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2027}"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "Paired baseline bootstrap: Frozen embeddings + LR - TF-IDF + LR"
python -u scripts/analyze_baseline_paired_bootstrap.py \
  --tfidf-root "${TFIDF_ROOT}" \
  --frozen-root "${FROZEN_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-resamples "${N_RESAMPLES}" \
  --seed "${BOOTSTRAP_SEED}"
