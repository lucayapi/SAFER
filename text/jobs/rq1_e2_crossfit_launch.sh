#!/usr/bin/env bash
# Pilot BTP with five cross-fit folds. After review, use --remaining for other targets.
#SBATCH --job-name=rq1-e2x-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_e2_crossfit_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_e2_crossfit_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
cd "${submit}"
mode="pilot"
if [[ "${1:-}" == "--remaining" ]]; then
  mode="remaining"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: sbatch text/jobs/rq1_e2_crossfit_launch.sh [--remaining]" >&2
  exit 2
fi
if [[ "${mode}" == "pilot" ]]; then
  prepare="$(sbatch --parsable "${jobs}/rq1_e2_crossfit_prepare.sh")"
  smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/rq1_e2_crossfit_smoke.sh")"
  oof="$(sbatch --parsable --array=0-4%2 --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/rq1_e2_crossfit_oof.sh")"
  final="$(sbatch --parsable --array=0 --kill-on-invalid-dep=yes --dependency="afterok:${oof}" "${jobs}/rq1_e2_crossfit_final.sh")"
  refine="$(sbatch --parsable --array=0 --kill-on-invalid-dep=yes --dependency="afterok:${final}" "${jobs}/rq1_e2_crossfit_refine.sh")"
  summary="$(sbatch --parsable --export=ALL,RQ1_E2X_SCOPE=pilot --kill-on-invalid-dep=yes --dependency="afterok:${refine}" "${jobs}/rq1_e2_crossfit_summary.sh")"
  printf 'Preflight: %s\nSmoke: %s\nBTP OOF folds 0-4%%2: %s\nBTP final M3: %s\nBTP E2: %s\nPilot report: %s\n' \
    "${prepare}" "${smoke}" "${oof}" "${final}" "${refine}" "${summary}"
else
  prepare="$(sbatch --parsable --export=ALL,RQ1_E2X_REQUIRE_PILOT=1 "${jobs}/rq1_e2_crossfit_prepare.sh")"
  oof="$(sbatch --parsable --array=5-29%2 --export=ALL,RQ1_E2X_SCOPE=all --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/rq1_e2_crossfit_oof.sh")"
  final="$(sbatch --parsable --array=1-5%2 --kill-on-invalid-dep=yes --dependency="afterok:${oof}" "${jobs}/rq1_e2_crossfit_final.sh")"
  refine="$(sbatch --parsable --array=1-5%2 --kill-on-invalid-dep=yes --dependency="afterok:${final}" "${jobs}/rq1_e2_crossfit_refine.sh")"
  summary="$(sbatch --parsable --export=ALL,RQ1_E2X_SCOPE=all --kill-on-invalid-dep=yes --dependency="afterok:${refine}" "${jobs}/rq1_e2_crossfit_summary.sh")"
  printf 'Preflight: %s\nRemaining OOF folds 5-29%%2: %s\nRemaining final M3: %s\nRemaining E2: %s\nFull report: %s\n' \
    "${prepare}" "${oof}" "${final}" "${refine}" "${summary}"
fi
