#!/bin/bash
# Export CSV Qwen figés pour un corpus test (prérequis eval SCGM test + macro_transfer).
#
# Usage :
#   cd ~/SAFER/text && bash jobs/export_test_embeddings.sh
#   TEST_CORPUS=metallurgie sbatch jobs/export_test_embeddings.sh

#SBATCH --job-name=test_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
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

mkdir -p embeddings/test

TEST_CORPUS="${TEST_CORPUS:-metallurgie}"
BACKBONE_NAME="${BACKBONE_NAME:-Qwen/Qwen3-Embedding-0.6B}"

EMB_TEST_CSV="$(python scripts/_resolve_test_corpus_cli.py "${TEST_CORPUS}" --field emb)"

echo "HOST=$(hostname) DATE=$(date -Iseconds) TEST_CORPUS=${TEST_CORPUS}"

if [[ -f "${EMB_TEST_CSV}" ]]; then
  echo "[test_emb] déjà présent : ${EMB_TEST_CSV}"
  exit 0
fi

echo "[test_emb] export Qwen → ${EMB_TEST_CSV}"
python scripts/export_test_embeddings.py \
  --corpus "${TEST_CORPUS}" \
  --backbone_name "${BACKBONE_NAME}"

echo "[test_emb] terminé $(date -Iseconds)"
