#!/usr/bin/env bash
#SBATCH --job-name=decision-facthood
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/ or SAFER/text/}"
if [[ -d "${submit}/text/jobs" ]]; then launcher="${submit}/text/jobs/_decision_gold_launch.sh"
else launcher="${submit}/jobs/_decision_gold_launch.sh"; fi
bash "${launcher}" facthood
