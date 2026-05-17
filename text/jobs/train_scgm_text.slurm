#!/bin/bash
#SBATCH --job-name=scgm_text
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=resultats/scgm_text/logs/slurm-%x-%j.out
#SBATCH --error=resultats/scgm_text/logs/slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."
mkdir -p resultats/scgm_text/logs

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

python scripts/train_scgm_text.py \
  --config configs/methods/scgm_text.yaml \
  --data_csv dataset/data_btp.csv \
  --text_col sentence \
  --label_col pred_label \
  --group_col accident_id \
  --output_dir resultats/scgm_text
