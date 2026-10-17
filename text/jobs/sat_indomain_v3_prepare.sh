#!/usr/bin/env bash
# Create the private 70/15/15 bundle on the cluster from the existing GOLD v3 bundle.
# Submit from SAFER/: sbatch text/jobs/sat_indomain_v3_prepare.sh
#SBATCH --job-name=sat-indomain-prep
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
unset SAT_LODO_BUNDLE
sat_gold_v3_setup
private_root="${SCRATCH:-$HOME}/safer_sat_indomain"
bundle="${private_root}/bundle/sat12l_sm_gold_v3_indomain_v1"
if [[ ! -f "${bundle}/complete.json" ]]; then
  python -u scripts/sat_indomain_v3.py prepare \
    --source-bundle "${SAT_LODO_BUNDLE}" --out "${bundle}"
fi
python -u scripts/sat_indomain_v3.py check-bundle --bundle "${bundle}"
echo "In-domain bundle ready: ${bundle}"
