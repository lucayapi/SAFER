#!/bin/bash
# Export des embeddings CamemBERT gelés pour les quatre corpus de l'étude.
# Depuis text/ : sbatch jobs/export_camembert_embeddings.sh
# Ré-encoder les fichiers existants : FORCE=1 sbatch jobs/export_camembert_embeddings.sh

#SBATCH --job-name=camembert-emb
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
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

ARGS=(--config configs/export_embeddings_camembert.yaml)
if [[ "${FORCE:-0}" == "1" ]]; then ARGS+=(--force); fi
echo "HOST=$(hostname) DATE=$(date -Iseconds) BACKBONE=almanach/camembert-base"
python -u scripts/export_corpus_embeddings.py "${ARGS[@]}"
