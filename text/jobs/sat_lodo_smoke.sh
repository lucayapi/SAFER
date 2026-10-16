#!/usr/bin/env bash
# Verifie un pas GPU et l'adaptateur; installe automatiquement le runtime.
# Depuis SAFER/ : sbatch text/jobs/sat_lodo_smoke.sh
# Depuis SAFER/text/ : sbatch jobs/sat_lodo_smoke.sh
#SBATCH --job-name=sat-lodo-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/jobs/sat_lodo_gpu.sh" ]]; then
  script="${SLURM_SUBMIT_DIR}/jobs/sat_lodo_gpu.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/text/jobs/sat_lodo_gpu.sh" ]]; then
  script="${SLURM_SUBMIT_DIR}/text/jobs/sat_lodo_gpu.sh"
else
  echo "Soumettre depuis SAFER/ ou SAFER/text/." >&2
  exit 1
fi
export SAT_LODO_SMOKE=1
exec bash "${script}"
