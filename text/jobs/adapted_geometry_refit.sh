#!/bin/bash
# One selected model x one seed; GPU refit preserves its own adapted embeddings.
#SBATCH --job-name=adapt-geo-fit
#SBATCH --partition=gpu
#SBATCH --exclude=hpcnode39,piafgpu01,iccfgpu01
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/jobs/_bootstrap.sh"
export OMP_NUM_THREADS=29 OPENBLAS_NUM_THREADS=29 MKL_NUM_THREADS=29 NUMEXPR_NUM_THREADS=29
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-output/replication_recipes/adapted_geometry.yaml}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?submit with the campaign launcher}"
python -u scripts/run_replication.py --config "${CONFIG}" --task-index "${TASK_INDEX}"
python - "${CONFIG}" "${TASK_INDEX}" <<'PY'
import sys
from pathlib import Path
from replication.runner import load_replication_config, output_root, task_matrix
config = load_replication_config(sys.argv[1])
task = task_matrix(config)[int(sys.argv[2])]
model_id, seed = task['model_id'], task['training_seed']
run = output_root(config) / model_id / f'seed_{seed}'
source = 'metallurgie' if '_metallurgie_' in model_id else 'btp'
for corpus in ('btp', 'metallurgie', 'caou', 'nicollin'):
    for suffix in ('.npy', '_metadata.csv'):
        file = run / 'embeddings' / f'projected_{corpus}{suffix}'
        if not file.is_file():
            raise FileNotFoundError(file)
for corpus in ('btp', 'metallurgie', 'caou', 'nicollin'):
    if corpus != source and not (run / 'predictions' / f'predictions_{corpus}.csv').is_file():
        raise FileNotFoundError(f'Missing predictions for {corpus}: {run}')
print(f'[verified] {model_id} seed={seed}: source and three targets exported')
PY
