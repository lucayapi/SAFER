#!/usr/bin/env bash
# Recover only BTP final calibration/refinement from validated v3 feature caches.
#SBATCH --job-name=rq1-e2x-resume
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_e2_crossfit_rebind.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_e2_crossfit_rebind.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
cd "${submit}"
rebind="$(sbatch --parsable "${jobs}/rq1_e2_crossfit_rebind.sh")"
final="$(sbatch --parsable --array=0 --kill-on-invalid-dep=yes --dependency="afterok:${rebind}" "${jobs}/rq1_e2_crossfit_final.sh")"
refine="$(sbatch --parsable --array=0 --kill-on-invalid-dep=yes --dependency="afterok:${final}" "${jobs}/rq1_e2_crossfit_refine.sh")"
summary="$(sbatch --parsable --export=ALL,RQ1_E2X_SCOPE=pilot --kill-on-invalid-dep=yes --dependency="afterok:${refine}" "${jobs}/rq1_e2_crossfit_summary.sh")"
printf 'BTP cache rebind: %s\nBTP final calibration: %s\nBTP E2 refinement: %s\nPilot summary: %s\n' \
  "${rebind}" "${final}" "${refine}" "${summary}"
