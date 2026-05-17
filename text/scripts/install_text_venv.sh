#!/bin/bash
# Installe le venv SAFER/text avec versions compatibles (transformers 4.51 + numpy 1.26).
# Usage : cd text && source .venv/bin/activate && bash scripts/install_text_venv.sh
#
# Option GPU (ex. CUDA 12.4) :
#   INSTALL_TORCH_CUDA=1 bash scripts/install_text_venv.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERREUR: activez d'abord le venv : source .venv/bin/activate" >&2
  exit 1
fi

echo "python: $(which python)"
echo "pip install (requirements + constraints)..."
pip install -U pip wheel
pip install -r requirements.txt -c constraints.txt

if [[ "${INSTALL_TORCH_CUDA:-0}" == "1" ]]; then
  echo "Réinstallation torch CUDA 2.5.1 (cu124)..."
  pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi

# Verrouillage explicite HF (évite downgrade résiduel vers 4.46.x)
pip install --force-reinstall --no-deps transformers==4.51.3 tokenizers==0.21.0
pip install "huggingface-hub>=0.26.0,<1.0" "safetensors>=0.4.3,<1.0"
pip install numpy==1.26.4 "fsspec>=2023.1.0,<=2024.9.0"

echo ""
python -c "
import numpy as np
import transformers as tr
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print('numpy', np.__version__)
print('transformers', tr.__version__)
print('qwen3', 'qwen3' in CONFIG_MAPPING)
assert tuple(int(x) for x in tr.__version__.split('.')[:3]) >= (4, 51, 0)
assert np.__version__.startswith('1.26')
print('OK — environnement text prêt')
"

echo "Terminé."
