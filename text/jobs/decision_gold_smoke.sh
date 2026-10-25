#!/usr/bin/env bash
#SBATCH --job-name=decision-gold-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?}"
if [[ -d "${submit}/text/jobs" ]]; then env_script="${submit}/text/jobs/_decision_gold_env.sh"
else env_script="${submit}/jobs/_decision_gold_env.sh"; fi
source "${env_script}"
python -u scripts/decision_gold_models.py smoke --decision "${DECISION_GOLD_DECISION:?}"
