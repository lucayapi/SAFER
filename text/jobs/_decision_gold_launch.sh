#!/usr/bin/env bash
set -euo pipefail
decision="${1:?facthood or roles required}"
[[ "${decision}" == facthood || "${decision}" == roles ]] || exit 2
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/ or SAFER/text/}"
if [[ -d "${submit}/text/jobs" ]]; then jobs="${submit}/text/jobs"
elif [[ -d "${submit}/jobs" ]]; then jobs="${submit}/jobs"
else echo 'Submit from SAFER/ or SAFER/text/.' >&2; exit 2; fi
export DECISION_GOLD_DECISION="${decision}"
if command -v squeue >/dev/null 2>&1; then
  active_campaign=0
  while read -r job_id job_name; do
    [[ -z "${job_id:-}" || "${job_id}" == "${SLURM_JOB_ID:-}" ]] && continue
    case "${job_name}" in
      facthood-*|roles-*|decision-facthood|decision-roles) active_campaign=1 ;;
    esac
  done < <(squeue -h -u "${USER:?}" -o '%i %j')
  if [[ "${active_campaign}" == 1 ]]; then
    echo "An active facthood/roles campaign already exists for this user; launch these campaigns one at a time to keep their combined GPU concurrency at three." >&2
    exit 2
  fi
fi
prepare="$(sbatch --parsable --job-name="${decision}-prep" --export=ALL "${jobs}/decision_gold_prepare.sh")"
if [[ "${decision}" == facthood ]]; then
  embeddings="$(sbatch --parsable --job-name="${decision}-excl-emb" --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" \
    --array=0-5%3 --export=ALL "${jobs}/decision_gold_embed_exclusions.sh")"
  smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${embeddings}" \
    --job-name="${decision}-smoke" --export=ALL "${jobs}/decision_gold_smoke.sh")"
else
  embeddings="-"
  smoke="$(sbatch --parsable --job-name="${decision}-smoke" --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" \
    --export=ALL "${jobs}/decision_gold_smoke.sh")"
fi
fit="$(sbatch --parsable --job-name="${decision}-fit" --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" \
  --array=0-17%3 --export=ALL "${jobs}/decision_gold_train.sh")"
summary="$(sbatch --parsable --job-name="${decision}-sum" --kill-on-invalid-dep=yes --dependency="afterok:${fit}" \
  --export=ALL "${jobs}/decision_gold_summary.sh")"
printf 'Decision: %s\nPreparation: %s\nExcluded-span Qwen embeddings: %s\nGPU smoke: %s\nLODO training (18 tasks, max 3 concurrent): %s\nSummary: %s\n' \
  "${decision}" "${prepare}" "${embeddings}" "${smoke}" "${fit}" "${summary}"
