#!/usr/bin/env bash
#SBATCH --job-name=role-gold-prepare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/text/jobs/_role_gold_env.sh"
mkdir -p "${ROLE_GOLD_RUN_ROOT:-${SCRATCH:-$PWD/..}/safer_role_gold_v3}"
exec 9>"${ROLE_GOLD_RUN_ROOT:-${SCRATCH:-$PWD/..}/safer_role_gold_v3}/.prepare.lock"
flock -x 9
python -u scripts/role_gold_campaign.py prepare --experiment "${ROLE_GOLD_EXPERIMENT:?}" --min-free-gib "${ROLE_GOLD_MIN_FREE_GIB:-170}"
