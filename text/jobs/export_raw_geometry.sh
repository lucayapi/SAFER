#!/bin/bash
# Métriques géométrie sur embeddings encodeur Qwen (BTP + corpus test).
# Sorties : output/raw_embedding/ et output_test/<corpus>/raw_embedding/
#
# Usage :
#   cd ~/SAFER/text && bash jobs/export_raw_geometry.sh
#   ALL_TEST_CORPORA=1 bash jobs/export_raw_geometry.sh
#   SKIP_BTP=1 TEST_CORPUS=metallurgie bash jobs/export_raw_geometry.sh

#SBATCH --job-name=raw_geom
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

SKIP_BTP="${SKIP_BTP:-0}"
SKIP_TEST="${SKIP_TEST:-0}"
SKIP_NPY="${SKIP_NPY:-0}"
TEST_CORPUS="${TEST_CORPUS:-metallurgie}"
ALL_TEST_CORPORA="${ALL_TEST_CORPORA:-0}"

RAW_ARGS=()
if [[ "${SKIP_NPY}" == "1" ]]; then
  RAW_ARGS+=(--skip_npy)
fi

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"

if [[ "${SKIP_BTP}" != "1" ]]; then
  echo "[raw_geometry] BTP (embedding brut)…"
  python scripts/export_raw_embeddings.py \
    --config configs/methods/raw_embedding.yaml \
    "${RAW_ARGS[@]}"
fi

if [[ "${SKIP_TEST}" != "1" ]]; then
  if [[ "${ALL_TEST_CORPORA}" == "1" ]]; then
    mapfile -t _corpora < <(python -c "
from safer_core.test_corpus import list_test_corpus_ids
for c in list_test_corpus_ids():
    print(c)
")
  else
    _corpora=("${TEST_CORPUS}")
  fi
  for cid in "${_corpora[@]}"; do
    echo "[raw_geometry] test corpus=${cid}…"
    python scripts/export_raw_embeddings.py \
      --config configs/methods/raw_embedding_test.yaml \
      --corpus "${cid}" \
      "${RAW_ARGS[@]}"
  done
fi

echo "[raw_geometry] terminé $(date -Iseconds)"
