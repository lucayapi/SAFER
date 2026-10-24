#!/usr/bin/env bash
#SBATCH --job-name=rq1-eq-prepare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_seg_equal_volume_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_seg_equal_volume_env.sh" prepare
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_seg_equal_volume_env.sh" prepare
fi
available_kb="$(df -Pk "${RQ1_EQUAL_RUN_ROOT}" | awk 'NR==2 {print $4}')"
echo "Free KiB: ${available_kb}; two concurrent GPU tasks require at least 12 GiB."
[[ "${available_kb}" =~ ^[0-9]+$ && "${available_kb}" -ge $((12 * 1024 * 1024)) ]] || {
  echo 'Insufficient free storage for two optimizer states.' >&2
  exit 1
}
if command -v quota >/dev/null 2>&1; then quota -s || true; fi
python -u scripts/rq1_seg_equal_volume.py prepare \
  --source-bundle "${RQ1_EQUAL_SOURCE_BUNDLE}" \
  --reference "${RQ1_EQUAL_REFERENCE}" --run-root "${RQ1_EQUAL_RUN_ROOT}"
