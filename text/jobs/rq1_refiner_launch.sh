#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/rq1_refiner_launch.sh
#SBATCH --job-name=rq1-refiner-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_refiner_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_refiner_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
cd "${submit}"
prepare="$(sbatch --parsable "${jobs}/rq1_refiner_prepare.sh")"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/rq1_refiner_smoke.sh")"
array="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/rq1_refiner_gpu.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${array}" "${jobs}/rq1_refiner_summary.sh")"
printf 'Preflight: %s\nSmoke: %s\nLODO 0-5%%2: %s\nSummary: %s\n' \
  "${prepare}" "${smoke}" "${array}" "${summary}"
