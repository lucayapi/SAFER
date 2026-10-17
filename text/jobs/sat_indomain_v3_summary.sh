#!/usr/bin/env bash
# Build the six-domain in-domain table and single-source transfer matrix.
# Submit only after the GPU array succeeds.
#SBATCH --job-name=sat-indomain-summary
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
fi
private_root="${SCRATCH:-$HOME}/safer_sat_indomain"
export SAT_LODO_BUNDLE="${private_root}/bundle/sat12l_sm_gold_v3_indomain_v1"
export SAT_LODO_RUN_ROOT="${private_root}/runs/sat12l_sm_gold_v3_indomain_v1"
sat_gold_v3_setup
python -u scripts/sat_indomain_v3.py summarize \
  --bundle "${SAT_LODO_BUNDLE}" --run-root "${SAT_LODO_RUN_ROOT}"
