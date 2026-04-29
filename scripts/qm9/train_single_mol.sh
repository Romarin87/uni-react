#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/single_mol/qm9.yaml}"
source "${ROOT_DIR}/scripts/common_torchrun.sh"

run_torch_module uni_react.train_qm9 --config "${CONFIG}" --model_name single_mol "$@"
