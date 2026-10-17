#!/usr/bin/env bash
# Short GPU training and adapter-reload check for the new single-source bundle.
# Submit from SAFER/: sbatch text/jobs/sat_indomain_v3_smoke.sh
#SBATCH --job-name=sat-indomain-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
sat_gold_v3_setup smoke

python -u scripts/sat_indomain_v3.py smoke \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
