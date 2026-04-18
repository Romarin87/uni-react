#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRETRAINED_CKPT=""
export OUT_PREFIX="${OUT_PREFIX:-qm9_scratch}"

bash "${SCRIPT_DIR}/train_qm9_all_targets_single_mol.sh"
