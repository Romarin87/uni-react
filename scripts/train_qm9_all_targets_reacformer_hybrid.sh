#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENCODER_TYPE="reacformer_hybrid"

bash "${SCRIPT_DIR}/train_qm9_all_targets.sh"
