#!/usr/bin/env bash
# From SAFER/: sbatch text/jobs/sat_narrative_only_checkpoint_smoke.sh
# Smoke test for training, exporting, and reloading an intermediate LoRA checkpoint.
#SBATCH --job-name=sat-narr-ckpt-smoke
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
export SAT_NARRATIVE_RUN_ROOT="${SAT_NARRATIVE_RUN_ROOT:-${SCRATCH:-$HOME}/safer_sat_lodo/runs/narrative_only_lodo_checkpoint_v1}"
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_narrative_only_env.sh"
sat_narrative_only_setup smoke
python -u scripts/sat_narrative_only_lodo.py check --prepared "${SAT_NARRATIVE_PREPARED}"
python -u scripts/sat_narrative_only_checkpoint_lodo.py smoke \
  --prepared "${SAT_NARRATIVE_PREPARED}" --run-root "${SAT_NARRATIVE_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
