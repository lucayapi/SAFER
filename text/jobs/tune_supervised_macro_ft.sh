#!/bin/bash
# Campagne de variantes supervised_macro_ft (tableau article).
#
# 8 combos : last 1/2/3 layers + full encoder × projector MLP Yes/No.
# Pour chaque combo : CV GroupKFold BTP + fit 100 % + predictions train/OOD + metrics.
# Pas de sélection « best only » — toutes les lignes du tableau sont exportées.
#
# Usage : cd ~/SAFER/text && sbatch jobs/tune_supervised_macro_ft.sh
# Variables :
#   GRID_CONFIG=configs/tuning/supervised_macro_ft_grid.yaml
#   TEST_CORPORA=metallurgie,caou,nicollin
#   MAX_COMBOS=2            # smoke test
#   SKIP_FINAL_FIT=1        # CV seule (debug)
#   SEED=42
#
# Sorties : output/supervised_macro_ft/variants/
#   results_summary.csv / results_summary.json
#   grid_summary.csv
#   combos/<combo_id>/{cv,metrics,predictions}/

#SBATCH --job-name=macro_ft_var
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=lucayapi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

GRID_CONFIG="${GRID_CONFIG:-configs/tuning/supervised_macro_ft_grid.yaml}"
export TEST_CORPORA="${TEST_CORPORA:-metallurgie,caou,nicollin}"

ARGS=(--grid-config "${GRID_CONFIG}")
if [[ -n "${MAX_COMBOS:-}" ]]; then
  ARGS+=(--max-combos "${MAX_COMBOS}")
fi
if [[ "${SKIP_FINAL_FIT:-0}" == "1" ]]; then
  ARGS+=(--skip-final-fit)
fi
if [[ -n "${SEED:-}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

echo "HOST=$(hostname) DATE=$(date -Iseconds) JOB_ID=${SLURM_JOB_ID:-local}"
echo "[macro_ft_var] GRID_CONFIG=${GRID_CONFIG} TEST_CORPORA=${TEST_CORPORA} MAX_COMBOS=${MAX_COMBOS:-all} SKIP_FINAL_FIT=${SKIP_FINAL_FIT:-0}"

export PYTHONUNBUFFERED=1
python -u scripts/tune_supervised_macro_ft.py "${ARGS[@]}"

echo "[macro_ft_var] terminé $(date -Iseconds)"
echo "Voir : output/supervised_macro_ft/variants/results_summary.csv"
