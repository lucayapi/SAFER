#!/bin/bash
set -euo pipefail
MODEL=softtriple_full_yes "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_replications_model.sh"
