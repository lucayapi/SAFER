#!/bin/bash
# Source-only selection of CE, SupCon and SoftTriple on both training sectors.
# Run from text/: bash jobs/submit_camembert_source_tuning.sh
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_bootstrap.sh"
fi

python scripts/prepare_backbone_source_tuning.py --backbones camembert_base
ROOT="output/backbone_source_factorial/source_only_tuning"
ids=()
for source in btp metallurgie; do
  for method in cross_entropy supcon softtriple; do
    stem="camembert_base_${source}_${method}"
    grid="${ROOT}/${stem}_grid.yaml"
    job_id="$(sbatch --parsable --job-name="select_${stem}" \
      --export=ALL,TUNING_METHOD="${method}",GRID_CONFIG="${grid}" \
      "${TEXT_JOBS_DIR}/tune_backbone_source_selection.sh")"
    job_id="${job_id%%;*}"
    ids+=("${job_id}")
    echo "${stem} -> ${job_id}"
  done
done
printf 'Sélections soumises : %s\n' "${ids[*]}"
