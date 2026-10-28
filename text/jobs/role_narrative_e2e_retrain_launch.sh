#!/usr/bin/env bash
# Reduced role-aware Table 4 requires NEW CE, SupCon and SoftTriple training.
# Run from SAFER/. The existing classification results remain untouched.
set -euo pipefail
jobs="${PWD}/text/jobs"
[[ -d "$jobs" ]] || { echo 'Run from SAFER/.' >&2; exit 2; }
after="${ROLE_E2E_AFTEROK:-}"
if [[ -n "$after" && ! "$after" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
  echo 'ROLE_E2E_AFTEROK must contain Slurm IDs separated by colons.' >&2
  exit 2
fi
dependency=''
[[ -z "$after" ]] || dependency="--dependency=afterok:${after}"
if [[ -n "$dependency" ]]; then
  prepare="$(sbatch --parsable --kill-on-invalid-dep=yes "$dependency" "${jobs}/role_narrative_e2e_retrain_prepare.sh")"
else
  prepare="$(sbatch --parsable --kill-on-invalid-dep=yes "${jobs}/role_narrative_e2e_retrain_prepare.sh")"
fi
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${prepare}" "${jobs}/role_narrative_e2e_retrain_smoke.sh")"
array="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/role_narrative_e2e_retrain_gpu.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${array}" "${jobs}/role_narrative_e2e_retrain_summary.sh")"
printf 'prepare=%s smoke=%s array=%s summary=%s\n' "$prepare" "$smoke" "$array" "$summary"
