#!/bin/bash
# Rétrocompat — délègue à export_corpus_embeddings.sh
#
# Usage :
#   cd ~/SAFER/text && bash jobs/export_test_embeddings.sh
#   TEST_CORPUS=metallurgie sbatch jobs/export_test_embeddings.sh

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CORPUS="${TEST_CORPUS:-metallurgie}"
exec bash "${DIR}/export_corpus_embeddings.sh" "$@"
