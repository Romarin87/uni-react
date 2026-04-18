#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRETRAINED_CKPT=""
export OUT_PREFIX="${OUT_PREFIX:-qm9_scratch_reacformer_so2}"

bash "${SCRIPT_DIR}/train_qm9_all_targets_reacformer_so2.sh"
