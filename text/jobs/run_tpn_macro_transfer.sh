#!/bin/bash
# Transfert macro TPN (encodeur gelé modulable + adaptateur) sur corpus test.
#
# Usage :
#   cd ~/SAFER/text && bash jobs/run_tpn_macro_transfer.sh
# Variables :
#   CORPUS=metallurgie
#   CONFIG=configs/tpn_macro_transfer.yaml
#   BASE_METHOD=scgm_text      # scgm_text | softtriple | supcon | batch_triplet
#   CHECKPOINT=output/scgm_text_speed/checkpoints/best_model.pt
#   SKIP_BERTOPIC=0
#   DEVICE=cuda
#   EPOCHS=50
#   TOPIC_EMBEDDING_MODE=mixed   # initial | adapted | mixed
#   TOPIC_ALPHA=0.25
#   RUN_BERTOPIC_GRID=0
#   BERTOPIC_ONLY=0
#   FORCE_REENCODE=1   # ignore source/target_projected.npy existants
#
# Exemples :
#   BASE_METHOD=softtriple bash jobs/run_tpn_macro_transfer.sh
#   CHECKPOINT=output/scgm_text_speed/checkpoints/best_model.pt bash jobs/run_tpn_macro_transfer.sh
#
# Prérequis : checkpoint entraîné pour l'encodeur choisi (train_*.sh)

#SBATCH --job-name=tpn_macro_transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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

CONFIG="${CONFIG:-configs/tpn_macro_transfer.yaml}"
CORPUS="${CORPUS:-metallurgie}"
BASE_METHOD="${BASE_METHOD:-scgm_text}"
CHECKPOINT="${CHECKPOINT:-}"
SKIP_BERTOPIC="${SKIP_BERTOPIC:-0}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"

export TEST_CORPUS="${CORPUS}"

extra=(--config "${CONFIG}" --corpus "${CORPUS}" --device "${DEVICE}" --epochs "${EPOCHS}")
if [[ -n "${BASE_METHOD}" ]]; then
  extra+=(--base-method "${BASE_METHOD}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  extra+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${SKIP_BERTOPIC}" == "1" ]]; then
  extra+=(--skip-bertopic)
fi
if [[ -n "${TOPIC_EMBEDDING_MODE:-}" ]]; then
  extra+=(--topic-embedding-mode "${TOPIC_EMBEDDING_MODE}")
fi
if [[ -n "${TOPIC_ALPHA:-}" ]]; then
  extra+=(--topic-alpha "${TOPIC_ALPHA}")
fi
if [[ "${RUN_BERTOPIC_GRID:-0}" == "1" ]]; then
  extra+=(--run-bertopic-grid)
fi
if [[ "${BERTOPIC_ONLY:-0}" == "1" ]]; then
  extra+=(--bertopic-only)
fi
if [[ "${FORCE_REENCODE:-0}" == "1" ]]; then
  extra+=(--force-reencode)
fi

echo "[tpn_macro_transfer] CORPUS=${CORPUS} BASE_METHOD=${BASE_METHOD} $(date -Iseconds)"
echo "[tpn_macro_transfer] Logs : tail -f slurm-${SLURM_JOB_ID:-local}.out  (PYTHONUNBUFFERED=1)"
# Optionnel : surcharge encoding.log_every_batches du YAML (ex. export TPN_ENCODE_LOG_EVERY_BATCHES=10)
python -u scripts/run_tpn_macro_transfer_discovery.py "${extra[@]}"
echo "[tpn_macro_transfer] terminé CORPUS=${CORPUS} $(date -Iseconds)"
