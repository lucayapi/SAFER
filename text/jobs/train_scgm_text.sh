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
mkdir -p "${HF_HOME}"

# Qwen3-Embedding exige transformers >= 4.51 dans CE venv (pas ~/.local)
python -c "
import sys
import transformers as tr
print('python', sys.executable)
print('transformers', tr.__version__)
parts = tuple(int(x) for x in tr.__version__.split('.')[:3] if x.isdigit())
if parts < (4, 51, 0):
    raise SystemExit(
        'ERREUR: transformers ' + tr.__version__ + ' < 4.51 (qwen3). '
        'Dans text/: source .venv/bin/activate && pip install -U transformers==4.51.3 tokenizers==0.21.0'
    )
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if 'qwen3' not in CONFIG_MAPPING:
    raise SystemExit('ERREUR: architecture qwen3 absente — réinstaller transformers==4.51.3')
print('qwen3 architecture OK')
"

python scripts/train_scgm_text.py \
  --config configs/scgm_text_strict_finetune_identity.yaml \
  --strict_finetune_identity \
  --data_csv dataset/data_btp.csv \
  --text_col sentence \
  --label_col pred_label \
  --group_col accident_id \
  --backbone_model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --output_dir resultats/scgm_text_backbone