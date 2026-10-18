#!/usr/bin/env bash
#SBATCH --job-name=rq1-refiner-preflight
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
available_kb="$(df -Pk "${SCRATCH:-$HOME}" | awk 'NR==2 {print $4}')"
required_kb=$((12 * 1024 * 1024))
[[ "${available_kb}" =~ ^[0-9]+$ && "${available_kb}" -ge "${required_kb}" ]] || {
  echo 'Less than 12 GiB free before preparing the refiner runtime.' >&2
  exit 1
}
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_refiner_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_refiner_env.sh" smoke
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_refiner_env.sh" smoke
fi
python -u scripts/rq1_refiner.py preflight \
  --bundle "${RQ1_FINAL_BUNDLE}" --m3-root "${RQ1_REFINER_M3_ROOT}" \
  --out "${RQ1_REFINER_OUT}"
