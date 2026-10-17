#!/usr/bin/env bash
# Six RQ1 LoRA adaptations, one target held out per task.
# From SAFER/: sbatch text/jobs/sat_lodo_gold_v3_gpu.sh
#SBATCH --job-name=sat-gold-v3
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_lodo_gold_v3_env.sh"
else
  source "${SLURM_SUBMIT_DIR}/jobs/_sat_lodo_gold_v3_env.sh"
fi
sat_gold_v3_setup
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
python -u scripts/sat_lodo_gold_v3.py check-bundle --bundle "${SAT_LODO_BUNDLE}"
python -u scripts/sat_lodo_gold_v3.py fit \
  --bundle "${SAT_LODO_BUNDLE}" \
  --run-root "${SAT_LODO_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
