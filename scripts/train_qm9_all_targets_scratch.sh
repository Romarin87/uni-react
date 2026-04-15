#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Scratch 训练：强制不加载预训练权重。
# 其余超参直接复用 train_qm9_all_targets.sh，并允许通过环境变量覆盖。

export PRETRAINED_CKPT=""
ENCODER_TYPE="${ENCODER_TYPE:-single_mol}"
export ENCODER_TYPE

if [[ -z "${OUT_PREFIX:-}" ]]; then
  case "${ENCODER_TYPE}" in
    single_mol) OUT_PREFIX="qm9_scratch" ;;
    *) OUT_PREFIX="qm9_scratch_${ENCODER_TYPE}" ;;
  esac
fi
export OUT_PREFIX

# r2 常用特例参数（仅 r2 生效；按需覆盖）
source "${SCRIPT_DIR}/qm9_r2_defaults.sh"

bash "${SCRIPT_DIR}/train_qm9_all_targets.sh"
