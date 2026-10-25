#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/sat_elementary_checkpoint_smoke.sh
#SBATCH --job-name=sat-elem-ckpt-smoke
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
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_checkpoint_env.sh"
sat_elementary_checkpoint_setup smoke
python -u scripts/sat_elementary_checkpoint_lodo.py check \
  --bundle "${SAT_LODO_BUNDLE}" --prepared "${SAT_ELEMENTARY_PREPARED}" \
  --run-root "${SAT_ELEMENTARY_CHECKPOINT_RUN_ROOT}"
python -u scripts/sat_elementary_checkpoint_lodo.py smoke \
  --bundle "${SAT_LODO_BUNDLE}" --prepared "${SAT_ELEMENTARY_PREPARED}" \
  --run-root "${SAT_ELEMENTARY_CHECKPOINT_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
