#!/usr/bin/env bash
#SBATCH --job-name=rq1-final-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_rq1_final_env.sh"
python -u scripts/rq1_final_lodo.py summarize \
  --bundle "${RQ1_FINAL_BUNDLE}" --run-root "${RQ1_FINAL_RUN_ROOT}"

