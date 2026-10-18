#!/usr/bin/env bash
# Check private input, storage, and construct the shared-test LODO bundle.
#SBATCH --job-name=rq1-final-prepare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR:?}/text/jobs/_rq1_final_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_rq1_final_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_rq1_final_env.sh"
fi

# Two simultaneous jobs each retain one resumable full state while the next
# checkpoint is being written. Twelve GiB covers this transient overlap with
# headroom; a separate user quota can be tighter than the filesystem free space.
available_kb="$(df -Pk "${storage_root}" | awk 'NR==2 {print $4}')"
required_kb=$((12 * 1024 * 1024))
echo "Available filesystem KiB: ${available_kb}; required: ${required_kb}"
[[ "${available_kb}" =~ ^[0-9]+$ && "${available_kb}" -ge "${required_kb}" ]] || {
  echo 'Less than 12 GiB free for the two concurrent GPU tasks.' >&2
  exit 1
}
if command -v quota >/dev/null 2>&1; then quota -s || true; fi
python -u scripts/rq1_final.py prepare --source "${RQ1_FINAL_SOURCE}" --out "${RQ1_FINAL_BUNDLE}"
python -u scripts/rq1_final.py check --bundle "${RQ1_FINAL_BUNDLE}" >/dev/null
echo "Prepared ${RQ1_FINAL_BUNDLE}"

