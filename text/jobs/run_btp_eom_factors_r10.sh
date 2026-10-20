#!/usr/bin/env bash
# Submit from text/: sbatch jobs/run_btp_eom_factors_r10.sh
# EOM factor discovery and annotation only; no Bayesian-network fit.
#SBATCH --job-name=btp_eom_factors
#SBATCH --partition=normal
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG=bayesian_network/config_btp_building_construction_and_masonry_eom_r10.yaml
export STAGE=factors REESTIMATE=0
exec bash "${SCRIPT_DIR}/run_bayesian_network.sh"
