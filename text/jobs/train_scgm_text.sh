#!/bin/bash
#SBATCH --job-name=scgm_text
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

python scripts/train_scgm_text.py \
  --config configs/scgm_text_strict_fidelity.yaml \
  --data_csv dataset/data_btp.csv \
  --text_col sentence \
  --label_col pred_label \
  --group_col accident_id \
  --output_dir resultats/scgm_text
