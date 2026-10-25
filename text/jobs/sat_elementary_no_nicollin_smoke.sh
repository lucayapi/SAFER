#!/usr/bin/env bash
#SBATCH --job-name=sat-no-nic-smoke
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
source "${SLURM_SUBMIT_DIR}/text/jobs/_sat_elementary_no_nicollin_env.sh"
sat_elementary_no_nicollin_setup smoke
python -u scripts/sat_elementary_no_nicollin.py smoke \
  --bundle "${SAT_LODO_BUNDLE}" --run-root "${SAT_NO_NICOLLIN_RUN_ROOT}" \
  --wtpsplit-repo "${SAT_WTPSPLIT_REPO}"
