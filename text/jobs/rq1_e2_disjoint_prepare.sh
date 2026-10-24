#!/usr/bin/env bash
#SBATCH --job-name=rq1-e2d-prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_e2_disjoint_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_e2_disjoint_env.sh" smoke
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_e2_disjoint_env.sh" smoke
fi
python -u "${RQ1_E2D_SCRIPT}" preflight --bundle "${RQ1_FINAL_BUNDLE}" --out "${RQ1_E2D_ROOT}" \
  --min-free-gib "${RQ1_E2D_MIN_FREE_GIB:-40}"
