#!/usr/bin/env bash
#SBATCH --job-name=decision-gold-sum
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?}"
if [[ -d "${submit}/text/jobs" ]]; then env_script="${submit}/text/jobs/_decision_gold_env.sh"
else env_script="${submit}/jobs/_decision_gold_env.sh"; fi
source "${env_script}"
python -u scripts/decision_gold_models.py summary --decision "${DECISION_GOLD_DECISION:?}"
