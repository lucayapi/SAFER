#!/usr/bin/env bash
set -euo pipefail
experiment="${1:?single_source or lodo required}"
if [[ "${experiment}" != single_source && "${experiment}" != lodo ]]; then exit 2; fi
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
if [[ -d "${submit}/text/jobs" ]]; then jobs="${submit}/text/jobs";
elif [[ -d "${submit}/jobs" ]]; then jobs="${submit}/jobs";
else echo 'Submit from SAFER/ or SAFER/text/' >&2; exit 2; fi
prepare="$(sbatch --parsable --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_prepare.sh")"
embeddings="$(sbatch --parsable --dependency="afterok:${prepare}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_embeddings.sh")"
smoke="$(sbatch --parsable --dependency="afterok:${embeddings}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_smoke.sh")"
frozen="$(sbatch --parsable --dependency="afterok:${smoke}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_frozen.sh")"
train="$(sbatch --parsable --dependency="afterok:${smoke}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_train.sh")"
summary="$(sbatch --parsable --dependency="afterok:${frozen}:${train}" --export="ALL,ROLE_GOLD_EXPERIMENT=${experiment}" "${jobs}/role_gold_summary.sh")"
printf '%s: prepare=%s embeddings=%s smoke=%s frozen=%s train=%s summary=%s\n' \
  "${experiment}" "${prepare}" "${embeddings}" "${smoke}" "${frozen}" "${train}" "${summary}"
