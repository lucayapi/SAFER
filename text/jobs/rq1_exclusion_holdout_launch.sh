#!/usr/bin/env bash
#SBATCH --job-name=rq1-exh-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_exclusion_holdout_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_exclusion_holdout_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
export RQ1_EXCLUSION_HOLDOUT_ROOT="${RQ1_EXCLUSION_HOLDOUT_ROOT:-${SCRATCH:-$HOME}/safer_sat_rq1_exclusion_holdout/runs/seed13_v1}"
cd "${submit}"
prepare="$(sbatch --parsable "${jobs}/rq1_exclusion_holdout_prepare.sh")"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/rq1_exclusion_holdout_smoke.sh")"
gpu="$(sbatch --parsable --array=0-5%3 --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/rq1_exclusion_holdout_gpu.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${gpu}" "${jobs}/rq1_exclusion_holdout_summary.sh")"
printf 'Preparation: %s\nSaT smoke: %s\nSix target rotations (max three GPUs): %s\nSummary: %s\nOutput: %s\n' \
  "${prepare}" "${smoke}" "${gpu}" "${summary}" "${RQ1_EXCLUSION_HOLDOUT_ROOT}"
