#!/bin/bash
# From SAFER: sbatch --array=0-17%2 text/jobs/export_gold_embeddings.sh
# One pair: sbatch --export=ALL,BACKBONE=qwen3,CORPUS=btp text/jobs/export_gold_embeddings.sh
# Array order: qwen3/e5/camembert x btp/metallurgie/caou/bois_textille/service_temporaire/nicollin.

#SBATCH --job-name=gold-v3-emb
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Set FORCE=1 when resubmitting an encoder/corpus pair whose verified export
# must be regenerated from the current GOLD CSV.
set -euo pipefail
SUBMIT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_ROOT}/text/jobs/_bootstrap.sh" ]]; then
  source "${SUBMIT_ROOT}/text/jobs/_bootstrap.sh"
elif [[ -f "${SUBMIT_ROOT}/jobs/_bootstrap.sh" ]]; then
  source "${SUBMIT_ROOT}/jobs/_bootstrap.sh"
else
  echo "Cannot find text/jobs/_bootstrap.sh below ${SUBMIT_ROOT}" >&2
  exit 2
fi

corpora=(btp metallurgie caou bois_textille service_temporaire nicollin)
backbones=(qwen3 e5 camembert)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  task_id="${SLURM_ARRAY_TASK_ID}"
  if ! [[ "${task_id}" =~ ^[0-9]+$ ]] || (( task_id > 17 )); then
    echo "Array index must be 0..17" >&2
    exit 2
  fi
  BACKBONE="${backbones[task_id / 6]}"
  CORPUS="${corpora[task_id % 6]}"
fi
BACKBONE="${BACKBONE:-}"
CORPUS="${CORPUS:-}"
if [[ -z "${BACKBONE}" || -z "${CORPUS}" ]]; then
  echo "Set BACKBONE and CORPUS for a single task or submit --array=0-17%2" >&2
  exit 2
fi
if [[ ! " ${backbones[*]} " =~ " ${BACKBONE} " ]] || [[ ! " ${corpora[*]} " =~ " ${CORPUS} " ]]; then
  echo "Invalid BACKBONE/CORPUS pair: ${BACKBONE}/${CORPUS}" >&2
  exit 2
fi
if [[ ! -f "dataset/gold_v3_manifest.json" ]]; then
  echo "Copy all six private GOLD CSVs and gold_v3_manifest.json into SAFER/text/dataset first" >&2
  exit 2
fi
echo "[$(date -Iseconds)] host=$(hostname) backbone=${BACKBONE} corpus=${CORPUS}"
args=(--backbone "${BACKBONE}" --corpus "${CORPUS}")
if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--force)
fi
python scripts/export_gold_embeddings.py "${args[@]}"
echo "[$(date -Iseconds)] complete ${BACKBONE}/${CORPUS}"
