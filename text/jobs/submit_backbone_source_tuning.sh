#!/bin/bash
# Soumet les neuf nouvelles sÃ©lections source-only en trois groupes atomiques.
# Groupe 2 dÃ©marre seulement si les trois jobs du groupe 1 ont rÃ©ussi ; mÃªme
# rÃ¨gle entre les groupes 2 et 3. L'ordre est E5/BTP, Qwen3/mÃ©tallurgie,
# E5/mÃ©tallurgie.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

python scripts/prepare_backbone_source_tuning.py
ROOT="output/backbone_source_factorial/source_only_tuning"

submit_group() {
  local label="$1"
  local dependency="${2:-}"
  shift 2
  local ids=()
  local sbatch_args=()
  if [[ -n "${dependency}" ]]; then
    sbatch_args+=(--dependency="afterok:${dependency}")
  fi
  echo "=== Groupe ${label} ==="
  for spec in "$@"; do
    read -r method stem <<< "${spec}"
    local grid="${ROOT}/${stem}_grid.yaml"
    local job_id
    job_id="$(sbatch --parsable "${sbatch_args[@]}" --job-name="select_${stem}" --export=ALL,TUNING_METHOD="${method}",GRID_CONFIG="${grid}" "${TEXT_JOBS_DIR}/tune_backbone_source_selection.sh")"
    job_id="${job_id%%;*}"
    ids+=("${job_id}")
    echo "${stem} -> ${job_id}"
  done
  local IFS=:
  echo "${ids[*]}"
}

group1="$(submit_group '1: E5 / BTP' '' \
  'cross_entropy multilingual_e5_large_btp_cross_entropy' \
  'supcon multilingual_e5_large_btp_supcon' \
  'softtriple multilingual_e5_large_btp_softtriple' | tail -n 1)"
group2="$(submit_group '2: Qwen3 / métallurgie' "${group1}" \
  'cross_entropy qwen3_metallurgie_cross_entropy' \
  'supcon qwen3_metallurgie_supcon' \
  'softtriple qwen3_metallurgie_softtriple' | tail -n 1)"
group3="$(submit_group '3: E5 / métallurgie' "${group2}" \
  'cross_entropy multilingual_e5_large_metallurgie_cross_entropy' \
  'supcon multilingual_e5_large_metallurgie_supcon' \
  'softtriple multilingual_e5_large_metallurgie_softtriple' | tail -n 1)"
echo "Groupes soumis : ${group1} -> ${group2} -> ${group3}"
