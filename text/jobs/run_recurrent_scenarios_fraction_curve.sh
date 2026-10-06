#!/usr/bin/env bash
# One Slurm job: paired recovery at several report-sampling fractions.
# Replicates run concurrently within this job, using its allocated CPUs.
# The full-data point is a one-replicate implementation check.
#SBATCH --job-name=paired_fraction_curve
#SBATCH --partition=normal
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

DATASET="${DATASET:-btp_carpentry_and_joinery}"
CONFIG_PATH="${CONFIG_PATH:-recurrent_scenarios/config.yaml}"
RUN_DIR="${RUN_DIR:-recurrent_scenarios/runs/theme_discovery_audit/${DATASET}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/paired_fraction_sensitivity}"
N_REPLICATES="${N_REPLICATES:-200}"
SEED="${SEED:-2026}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-30}}"

run_fraction() {
  local fraction="$1"
  local replicates="$2"
  local tag="${fraction/./p}"
  local output="${OUTPUT_ROOT}/fraction_${tag}"
  echo "Running fraction=${fraction}; replicates=${replicates}; workers=${N_WORKERS}"
  python -u recurrent_scenarios/paired_recovery.py run \
    --config "${CONFIG_PATH}" --dataset "${DATASET}" \
    --run-dir "${RUN_DIR}" --output "${output}" \
    --replicates "${replicates}" --fraction "${fraction}" \
    --seed "${SEED}" --workers "${N_WORKERS}"
  python -u recurrent_scenarios/paired_recovery.py summarise \
    --config "${CONFIG_PATH}" --dataset "${DATASET}" \
    --run-dir "${RUN_DIR}" --output "${output}"
}

# Four sampling fractions, each with the same replicate count and seed scheme.
for fraction in 0.6 0.7 0.8 0.9; do
  run_fraction "${fraction}" "${N_REPLICATES}"
done

# With all reports retained there is no sampling variation. One replicate is
# sufficient to check whether the refitted partition and summaries recover
# the reference under the same settings and random seed.
run_fraction "1.0" "1"

python -u recurrent_scenarios/summarise_fraction_curve.py \
  --output-root "${OUTPUT_ROOT}" --replicates "${N_REPLICATES}"
echo "Fraction-sensitivity outputs: ${OUTPUT_ROOT}"
