#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/sat_elementary_full3_launch.sh
#SBATCH --job-name=sat-elem-full3-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
smoke=$(sbatch --parsable text/jobs/sat_elementary_full3_smoke.sh)
train=$(sbatch --parsable --dependency="afterok:${smoke}" text/jobs/sat_elementary_full3_gpu.sh)
summary=$(sbatch --parsable --dependency="afterok:${train}" text/jobs/sat_elementary_full3_summary.sh)
printf 'smoke=%s\ntrain_array=%s (0-5%%3)\nsummary=%s\n' "${smoke}" "${train}" "${summary}"
