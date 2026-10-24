#!/usr/bin/env bash
#SBATCH --job-name=rq1-e2d-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER or SAFER/text}"
if [[ -f "${submit}/text/jobs/rq1_e2_disjoint_prepare.sh" ]]; then
  jobs="${submit}/text/jobs"
elif [[ -f "${submit}/jobs/rq1_e2_disjoint_prepare.sh" ]]; then
  jobs="${submit}/jobs"
else
  echo 'Submit from SAFER/ or SAFER/text/.' >&2
  exit 1
fi
cd "${submit}"
prepare="$(sbatch --parsable "${jobs}/rq1_e2_disjoint_prepare.sh")"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/rq1_e2_disjoint_smoke.sh")"
gpu="$(sbatch --parsable --array=0-17%2 --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/rq1_e2_disjoint_gpu.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${gpu}" "${jobs}/rq1_e2_disjoint_summary.sh")"
printf 'Preflight: %s\nGPU smoke: %s\nSix targets x three seeds, max two GPUs: %s\nSummary: %s\n' \
  "${prepare}" "${smoke}" "${gpu}" "${summary}"
