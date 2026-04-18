#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 第二阶段（CDFT）预训练权重初始化的 QM9 下游训练脚本。
# 默认全量 12 目标；仅当 target=r2 时自动使用 R2_* 特例参数。
# 其余目标按 train_qm9_all_targets.sh 的通用参数正常运行。

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
OUT_PREFIX="${OUT_PREFIX:-}"
ENCODER_TYPE="${ENCODER_TYPE:-}"

if [[ -z "${ENCODER_TYPE}" && -n "${PRETRAINED_CKPT}" ]]; then
  case "${PRETRAINED_CKPT}" in
    *gotennet_l*) ENCODER_TYPE="gotennet_l" ;;
    *reacformer_se3*) ENCODER_TYPE="reacformer_se3" ;;
    *reacformer_so2*) ENCODER_TYPE="reacformer_so2" ;;
    *) ENCODER_TYPE="single_mol" ;;
  esac
fi

if [[ -z "${ENCODER_TYPE}" ]]; then
  ENCODER_TYPE="single_mol"
fi

if [[ -z "${PRETRAINED_CKPT}" ]]; then
  case "${ENCODER_TYPE}" in
    gotennet_l) PRETRAINED_CKPT="runs/gotennet_l_cdft/best.pt" ;;
    reacformer_se3) PRETRAINED_CKPT="runs/reacformer_se3_cdft/best.pt" ;;
    reacformer_so2) PRETRAINED_CKPT="runs/reacformer_so2_cdft/best.pt" ;;
    *) PRETRAINED_CKPT="runs/single_mol_cdft/best.pt" ;;
  esac
fi
export PRETRAINED_CKPT
export ENCODER_TYPE

if [[ -z "${OUT_PREFIX}" ]]; then
  case "${ENCODER_TYPE}" in
    single_mol) OUT_PREFIX="qm9_pretrain_cdft" ;;
    *) OUT_PREFIX="qm9_pretrain_${ENCODER_TYPE}_cdft" ;;
  esac
fi
export OUT_PREFIX

# r2 常用特例参数（仅 r2 生效；按需覆盖）
source "${SCRIPT_DIR}/qm9_r2_defaults.sh"

bash "${SCRIPT_DIR}/train_qm9_all_targets.sh"
