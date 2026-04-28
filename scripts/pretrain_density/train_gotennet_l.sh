#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/gotennet_l/density.yaml}"
source "${ROOT_DIR}/scripts/common_torchrun.sh"

run_torch_module uni_react.train_pretrain_density --config "${CONFIG}" "$@"
