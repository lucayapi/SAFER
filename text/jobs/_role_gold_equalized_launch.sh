#!/usr/bin/env bash
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
if [[ -d "${submit}/text/jobs" ]]; then jobs="${submit}/text/jobs";
elif [[ -d "${submit}/jobs" ]]; then jobs="${submit}/jobs";
else echo 'Submit from SAFER/ or SAFER/text/' >&2; exit 2; fi

launch_arm() {
  local experiment="$1"
  local prepare embeddings smoke frozen train summary
  prepare="$(sbatch --parsable --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_prepare.sh")"
  embeddings="$(sbatch --parsable --dependency="afterok:${prepare}" --array=0-5%2 --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_embeddings.sh")"
  smoke="$(sbatch --parsable --dependency="afterok:${embeddings}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_smoke.sh")"
  frozen="$(sbatch --parsable --dependency="afterok:${smoke}" --array=0-5%1 --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_frozen.sh")"
  # Serialize GPU stages within each arm while allowing two tasks in each GPU
  # array. The two arms together therefore use at most four GPUs.
  train="$(sbatch --parsable --dependency="afterok:${frozen}" --array=0-53%2 --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_train.sh")"
  summary="$(sbatch --parsable --dependency="afterok:${frozen}:${train}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_summary.sh")"
  printf '%s: prepare=%s embeddings=%s smoke=%s frozen=%s train=%s summary=%s\n' \
    "${experiment}" "${prepare}" "${embeddings}" "${smoke}" "${frozen}" "${train}" "${summary}" >&2
  echo "${summary}"
}

single_summary="$(launch_arm single_source_equalized | tail -n 1)"
lodo_summary="$(launch_arm lodo_equalized | tail -n 1)"
compare="$(sbatch --parsable --dependency="afterok:${single_summary}:${lodo_summary}" \
  "${jobs}/role_gold_equalized_compare.sh")"
printf 'Comparison summary: %s (after %s and %s)\n' "${compare}" "${single_summary}" "${lodo_summary}"
