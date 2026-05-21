#!/bin/bash
# Transfert macro + topics intra-macro sur corpus de test (SCGM puis SoftTriple).
#
# Usage :
#   cd ~/SAFER/text && bash jobs/run_macro_transfer.sh
# Variables :
#   CORPUS=metallurgie          # configs/test_corpora.yaml
#   CONFIG=configs/macro_transfer.yaml
#   METHOD=scgm_text|softtriple|both   # both = les deux encodeurs (défaut)
# Exemples :
#   METHOD=scgm_text bash jobs/run_macro_transfer.sh
#   METHOD=softtriple CORPUS=metallurgie sbatch jobs/run_macro_transfer.sh
#
# Prérequis : train_scgm_text.sh + train_softtriple.sh

#SBATCH --job-name=macro_transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-configs/macro_transfer.yaml}"
CORPUS="${CORPUS:-metallurgie}"
METHOD="${METHOD:-both}"

export TEST_CORPUS="${CORPUS}"

run_one() {
  local m="$1"
  local extra=(--corpus "${CORPUS}")
  if [[ -n "${CHECKPOINT:-}" ]]; then
    extra+=(--checkpoint "${CHECKPOINT}")
  fi
  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    extra+=(--output-dir "${OUTPUT_DIR}")
  fi
  echo "[macro_transfer] METHOD=${m} CORPUS=${CORPUS} $(date -Iseconds)"
  python scripts/run_macro_transfer_discovery.py \
    --method "${m}" \
    --config "${CONFIG}" \
    "${extra[@]}"
}

case "${METHOD}" in
  scgm_text|softtriple)
    run_one "${METHOD}"
    ;;
  both|*)
    run_one scgm_text
    run_one softtriple
    ;;
esac

echo "[macro_transfer] terminé CORPUS=${CORPUS} $(date -Iseconds)"
