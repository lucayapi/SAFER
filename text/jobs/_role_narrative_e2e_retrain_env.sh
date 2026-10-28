#!/usr/bin/env bash
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_qwen3_lodo_env.sh"
# Keep the earlier classification campaign intact. This new root is a full
# retraining campaign whose selected models are evaluated on SaT spans before
# their large checkpoint directories are pruned.
export ROLE_QWEN3_RUN_ROOT="${ROLE_E2E_RETRAIN_ROOT:-${SCRATCH:-${SLURM_SUBMIT_DIR}}/safer_role_qwen3_narrative_e2e_reduced_v1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
