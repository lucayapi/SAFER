#!/bin/bash
# Export Qwen3-Embedding-4B vectors for valid B, A0, A1 and C units using
# the role-specific clustering instructions in the accompanying YAML config.
# Submit from ~/SAFER/text with: sbatch jobs/export_all_roles_qwen3_4b_instruct.sh

#SBATCH --job-name=btp_qwen3_4b_instruct
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
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
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
setup_text_job_env
echo "HOST=$(hostname) DATE=$(date -Iseconds) GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python embedding_diagnostics/export_b_instruct_embeddings.py \
  --config embedding_diagnostics/config_qwen3_4b_instruct_all_roles.yaml
echo "[btp_qwen3_4b_instruct] completed $(date -Iseconds)"
