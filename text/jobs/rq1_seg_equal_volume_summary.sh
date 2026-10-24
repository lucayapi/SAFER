#!/usr/bin/env bash
#SBATCH --job-name=rq1-eq-report
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_seg_equal_volume_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_seg_equal_volume_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_seg_equal_volume_env.sh"
fi
python -u scripts/rq1_seg_equal_volume.py summary \
  --source-bundle "${RQ1_EQUAL_SOURCE_BUNDLE}" \
  --reference "${RQ1_EQUAL_REFERENCE}" --run-root "${RQ1_EQUAL_RUN_ROOT}"
