#!/bin/bash
# Export embeddings Qwen bruts + évaluation classification (BTP + corpus OOD).
# Sorties : output/raw_embedding/ (BTP + métallurgie + caou via test_corpora)
#
# Usage :
#   cd ~/SAFER/text && bash jobs/export_raw_embeddings_eval.sh
#   TEST_CORPORA=metallurgie,caou bash jobs/export_raw_embeddings_eval.sh

#SBATCH --job-name=raw_emb_eval
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  # shellcheck source=jobs/_bootstrap.sh
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  # shellcheck source=jobs/_bootstrap.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

export TEST_CORPORA="${TEST_CORPORA:-metallurgie,caou}"
SKIP_NPY="${SKIP_NPY:-0}"

RAW_ARGS=()
if [[ "${SKIP_NPY}" == "1" ]]; then
  RAW_ARGS+=(--skip_npy)
fi
echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "TEST_CORPORA=${TEST_CORPORA}"

echo "[raw_embedding] Export + classification (BTP + OOD)…"
if ((${#RAW_ARGS[@]} > 0)); then
  python scripts/export_raw_embeddings.py \
    --config configs/methods/raw_embedding.yaml \
    "${RAW_ARGS[@]}"
else
  python scripts/export_raw_embeddings.py \
    --config configs/methods/raw_embedding.yaml
fi

echo "[raw_embedding] terminé $(date -Iseconds)"
