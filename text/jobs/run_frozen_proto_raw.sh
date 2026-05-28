#!/bin/bash
# Baseline Frozen Source Prototypes - variante raw embedding.
#
# Usage:
#   cd ~/SAFER/text && bash jobs/run_frozen_proto_raw.sh
#
# Variables:
#   CORPUS=metallurgie
#   CONFIG=configs/frozen_source_prototypes_raw.yaml
#   DEVICE=cuda
#   OUTPUT_DIR=output_test/metallurgie/macro_transfer/frozen_source_prototypes/raw
#   SKIP_BERTOPIC=0   # 1 pour désactiver BERTopic/OpenAI

#SBATCH --job-name=fsp_raw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

CONFIG="${CONFIG:-configs/frozen_source_prototypes_raw.yaml}"
CORPUS="${CORPUS:-metallurgie}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-output_test/${CORPUS}/macro_transfer/frozen_source_prototypes/raw}"
SKIP_BERTOPIC="${SKIP_BERTOPIC:-0}"

echo "[fsp_raw] CORPUS=${CORPUS} DEVICE=${DEVICE} $(date -Iseconds)"
if [[ "${SKIP_BERTOPIC}" == "1" ]]; then
  python -u scripts/run_frozen_source_prototypes.py \
    --config "${CONFIG}" \
    --corpus "${CORPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --method-display-name "Embedding brut + prototypes source" \
    --skip-bertopic
else
  python -u scripts/run_frozen_source_prototypes.py \
    --config "${CONFIG}" \
    --corpus "${CORPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --method-display-name "Embedding brut + prototypes source"
fi
echo "[fsp_raw] terminé $(date -Iseconds)"
