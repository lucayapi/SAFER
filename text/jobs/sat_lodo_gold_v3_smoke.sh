#!/usr/bin/env bash
# GPU smoke test for the confirmed narrative_v3 bundle.
# From SAFER/: sbatch text/jobs/sat_lodo_gold_v3_smoke.sh
#SBATCH --job-name=sat-gold-v3-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
fi
sat_gold_v3_setup smoke
python -u scripts/sat_lodo_gold_v3.py check-bundle --bundle "${SAT_LODO_BUNDLE}"
python -u scripts/sat_lodo_gold_v3.py smoke \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
