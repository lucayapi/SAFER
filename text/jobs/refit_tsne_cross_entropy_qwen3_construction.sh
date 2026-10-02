#!/bin/bash
# Final cross-entropy fit used only to export representations for the
# qualitative Construction versus Company corpus t-SNE figure.
#
# Outputs:
# output/article_visualization/cross_entropy_qwen3_construction_tsne/embeddings/
#   projected_btp.npy
#   projected_btp_metadata.csv
#   projected_nicollin.npy
#   projected_nicollin_metadata.csv
# The completed cross-validation is not run again. This job applies its
# selected settings directly to the final two-epoch fit.
#
# Submit from the text directory:
#   sbatch jobs/refit_tsne_cross_entropy_qwen3_construction.sh

#SBATCH --job-name=tsne-ce-qwen3
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

CONFIG="${CONFIG:-configs/article_visualization/cross_entropy_qwen3_construction_tsne.yaml}"

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "[tsne-ce-qwen3] Final reference fit: Qwen3 / Construction / cross-entropy / full encoder / no projector"
python -u scripts/refit_tsne_cross_entropy_reference.py --config "${CONFIG}"
echo "[tsne-ce-qwen3] Complete $(date -Iseconds)"
