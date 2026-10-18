#!/usr/bin/env bash
#SBATCH --job-name=rq1-final-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_rq1_final_env.sh" smoke
python -u scripts/rq1_final.py check --bundle "${RQ1_FINAL_BUNDLE}" >/dev/null
python -u scripts/rq1_final_lodo.py smoke \
  --bundle "${RQ1_FINAL_BUNDLE}" --run-root "${RQ1_FINAL_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"

