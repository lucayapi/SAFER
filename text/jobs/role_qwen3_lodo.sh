#!/usr/bin/env bash
#SBATCH --job-name=role-qwen3-launch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
submit="${SLURM_SUBMIT_DIR:?Submit from SAFER/}"
[[ -d "${submit}/text/jobs" ]] || { echo 'Submit from SAFER/.' >&2; exit 2; }
jobs="${submit}/text/jobs"
preflight="$(sbatch --parsable --kill-on-invalid-dep=yes "${jobs}/role_qwen3_lodo_prepare.sh")"
smoke="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${preflight}" "${jobs}/role_qwen3_lodo_smoke.sh")"
baselines="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${smoke}" "${jobs}/role_qwen3_lodo_baselines.sh")"
training="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${baselines}" --array=0-53%3 "${jobs}/role_qwen3_lodo_train.sh")"
summary="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${training}" "${jobs}/role_qwen3_lodo_summary.sh")"
printf 'preflight=%s smoke=%s baselines=%s train_array=%s summary=%s\n' "${preflight}" "${smoke}" "${baselines}" "${training}" "${summary}"
