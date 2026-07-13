#!/bin/bash
# Export CSV embeddings encodeur pour un ou plusieurs corpus (registre test_corpora.yaml).
#
# Usage :
#   cd ~/SAFER/text && bash jobs/export_corpus_embeddings.sh
#   CORPUS=metallurgie sbatch jobs/export_corpus_embeddings.sh
#   ALL_CORPORA=1 FORCE=1 sbatch jobs/export_corpus_embeddings.sh

#SBATCH --job-name=corpus_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

mkdir -p embeddings

CORPUS="${CORPUS:-}"
ALL_CORPORA="${ALL_CORPORA:-0}"
BACKBONE_NAME="${BACKBONE_NAME:-}"
FORCE="${FORCE:-0}"

ARGS=(--config configs/export_embeddings.yaml)
if [[ -n "${BACKBONE_NAME}" ]]; then
  ARGS+=(--backbone_name "${BACKBONE_NAME}")
fi
if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "${ALL_CORPORA}" == "1" ]]; then
  ARGS+=(--all)
elif [[ -n "${CORPUS}" ]]; then
  ARGS+=(--corpus "${CORPUS}")
fi

echo "HOST=$(hostname) DATE=$(date -Iseconds) CORPUS=${CORPUS:-all} ALL_CORPORA=${ALL_CORPORA}"

python scripts/export_corpus_embeddings.py "${ARGS[@]}"

echo "[corpus_emb] terminé $(date -Iseconds)"
