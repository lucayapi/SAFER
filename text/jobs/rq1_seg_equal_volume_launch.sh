#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/rq1_seg_equal_volume_launch.sh
# 30 paired transfers, at most two GPU tasks simultaneously.
#SBATCH --job-name=rq1-eq-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
if [[ -f "${submit}/text/jobs/rq1_seg_equal_volume_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_seg_equal_volume_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
prepare="$(sbatch --parsable "${jobs}/rq1_seg_equal_volume_prepare.sh")"
smoke="$(sbatch --parsable --dependency="afterok:${prepare}" "${jobs}/rq1_seg_equal_volume_smoke.sh")"
fits="$(sbatch --parsable --dependency="afterok:${smoke}" "${jobs}/rq1_seg_equal_volume_gpu.sh")"
report="$(sbatch --parsable --dependency="afterok:${fits}" "${jobs}/rq1_seg_equal_volume_summary.sh")"
printf 'Prepare: %s\nSmoke: %s\n30 transfers (2 GPUs max): %s\nReport: %s\n' \
  "${prepare}" "${smoke}" "${fits}" "${report}"
