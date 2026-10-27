#!/usr/bin/env bash
set -euo pipefail
submit="${PWD}"
[[ -d "${submit}/text/jobs" ]] || { echo 'Run from SAFER/.' >&2; exit 2; }
jobs="${submit}/text/jobs"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes "${jobs}/role_qwen3_lodo_missing_smoke.sh")"
training="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/role_qwen3_lodo_missing_train.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${training}" "${jobs}/role_qwen3_lodo_missing_summary.sh")"
printf 'smoke=%s train_array=%s summary=%s\n' "$smoke" "$training" "$summary"
