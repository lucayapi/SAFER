#!/usr/bin/env bash
#SBATCH --job-name=role-gold-e5
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_gold_env.sh"
corpora=(btp metallurgie caou bois_textille service_temporaire nicollin)
task_id="${SLURM_ARRAY_TASK_ID:?}"
corpus="${corpora[$task_id]}"
mkdir -p embeddings/gold_v3/e5
exec 9>"embeddings/gold_v3/e5/.${corpus}.lock"
flock -x 9
python -u scripts/export_gold_embeddings.py --backbone e5 --corpus "${corpus}"
