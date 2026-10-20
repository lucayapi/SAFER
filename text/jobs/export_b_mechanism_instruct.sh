#!/bin/bash
# Re-encode valid B units with one task instruction; preserve standard exports.
# Submit from ~/SAFER/text with: sbatch jobs/export_b_mechanism_instruct.sh

#SBATCH --job-name=btp_B_instruct_emb
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
python embedding_diagnostics/export_b_instruct_embeddings.py
echo "[btp_B_instruct_emb] completed $(date -Iseconds)"
