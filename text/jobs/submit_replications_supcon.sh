#!/bin/bash
set -euo pipefail
MODEL=supcon_full_yes "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_replications_model.sh"
