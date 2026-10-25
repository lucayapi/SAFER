#!/usr/bin/env bash
#SBATCH --job-name=rq1-excl-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_exclusion_boundaries_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_exclusion_boundaries_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
storage="${SCRATCH:-$HOME}"
export RQ1_E2X_ROOT="${RQ1_E2X_ROOT:-${storage}/safer_sat_rq1_refiner_crossfit/runs/rq1_e2_crossfit_exclusion_source_v1}"
export RQ1_EXCLUSION_ROOT="${RQ1_EXCLUSION_ROOT:-${storage}/safer_sat_rq1_exclusion_boundaries/runs/seed13_v2}"
cd "${submit}"
prepare="$(sbatch --parsable "${jobs}/rq1_exclusion_boundaries_prepare.sh")"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/rq1_e2_crossfit_smoke.sh")"
exclusion_smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/rq1_exclusion_boundaries_smoke.sh")"
oof="$(sbatch --parsable --array=0-29%3 --kill-on-invalid-dep=yes --dependency="afterok:${exclusion_smoke}" "${jobs}/rq1_e2_crossfit_oof.sh")"
final="$(sbatch --parsable --array=0-5%3 --kill-on-invalid-dep=yes --dependency="afterok:${oof}" "${jobs}/rq1_e2_crossfit_final.sh")"
refine="$(sbatch --parsable --array=0-5%3 --kill-on-invalid-dep=yes --dependency="afterok:${final}" "${jobs}/rq1_exclusion_boundaries_gpu.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${refine}" "${jobs}/rq1_exclusion_boundaries_summary.sh")"
printf 'Preparation: %s\nSaT GPU smoke: %s\nExclusion boundary smoke: %s\nFive source folds per target: %s\nFinal adapted SaT: %s\nOne-seed exclusion-aware refiner: %s\nSummary: %s\nOutput: %s\n' \
  "${prepare}" "${smoke}" "${exclusion_smoke}" "${oof}" "${final}" "${refine}" "${summary}" "${RQ1_EXCLUSION_ROOT}"
