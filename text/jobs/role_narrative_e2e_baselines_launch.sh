#!/usr/bin/env bash
# Run from SAFER/. This evaluates only recoverable role baselines; no encoder training.
set -euo pipefail
jobs="${PWD}/text/jobs"
[[ -d "${jobs}" ]] || { echo 'Run from SAFER/.' >&2; exit 2; }
after="${ROLE_E2E_AFTEROK:-}"
if [[ -n "$after" && ! "$after" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
  echo 'ROLE_E2E_AFTEROK must contain Slurm job/array IDs separated by colons.' >&2
  exit 2
fi
preflight="$(sbatch --parsable --kill-on-invalid-dep=yes "${jobs}/role_narrative_e2e_baselines_prepare.sh")"
dependency="afterok:${preflight}"
[[ -z "$after" ]] || dependency="${dependency}:${after}"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="$dependency" "${jobs}/role_narrative_e2e_baselines_smoke.sh")"
array="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/role_narrative_e2e_baselines_gpu.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${array}" "${jobs}/role_narrative_e2e_baselines_summary.sh")"
printf 'preflight=%s gpu_smoke=%s six_target_array=%s summary=%s\n' "$preflight" "$smoke" "$array" "$summary"
