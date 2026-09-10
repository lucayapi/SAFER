#!/bin/bash
set -euo pipefail
MODEL=cross_entropy_full_yes "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_replications_model.sh"
