#!/usr/bin/env bash
# One command from SAFER/: sbatch text/jobs/rq1_final_launch.sh
#SBATCH --job-name=rq1-final-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_final_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_final_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
cd "${submit}"
prepare="$(sbatch --parsable "${jobs}/rq1_final_prepare.sh")"
smoke="$(sbatch --parsable --dependency="afterok:${prepare}" "${jobs}/rq1_final_smoke.sh")"
array="$(sbatch --parsable --dependency="afterok:${smoke}" "${jobs}/rq1_final_gpu.sh")"
summary="$(sbatch --parsable --dependency="afterok:${array}" "${jobs}/rq1_final_summary.sh")"
printf 'Prepare: %s\nSmoke: %s\nLODO 0-5%%2: %s\nSummary: %s\n' \
  "${prepare}" "${smoke}" "${array}" "${summary}"
