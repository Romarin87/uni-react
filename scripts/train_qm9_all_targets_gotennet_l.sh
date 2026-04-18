#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GotenNet-L QM9 launcher using the official QM9 backbone settings.
# This keeps a dedicated script so the model does not inherit the generic
# backbone hyper-parameters used by the other encoders.

export ENCODER_TYPE="gotennet_l"
export SPLIT="${SPLIT:-gotennet}"
export SEED="${SEED:-1}"
export NO_CENTER_COORDS="${NO_CENTER_COORDS:-1}"
export QM9_TARGET_VARIANT="${QM9_TARGET_VARIANT:-gotennet}"
export EMB_DIM="${EMB_DIM:-256}"
export SE3_LAYER="${SE3_LAYER:-4}"
export HEADS="${HEADS:-8}"
export ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-128}"
export CUTOFF="${CUTOFF:-5.0}"
export NUM_KERNEL="${NUM_KERNEL:-64}"
export PATH_DROPOUT="${PATH_DROPOUT:-0.1}"
export ACTIVATION_DROPOUT="${ACTIVATION_DROPOUT:-0.1}"
export ATTN_DROPOUT="${ATTN_DROPOUT:-0.1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export BACKBONE_LR="${BACKBONE_LR:-1e-4}"
export HEAD_LR="${HEAD_LR:-1e-4}"
export LR_SCHEDULER="${LR_SCHEDULER:-none}"
export WARMUP_STEPS="${WARMUP_STEPS:-10000}"
export LR_FACTOR="${LR_FACTOR:-0.8}"
export LR_PATIENCE="${LR_PATIENCE:-15}"
export LR_MIN="${LR_MIN:-1e-7}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-150}"
export EPOCHS="${EPOCHS:-3000}"

bash "${SCRIPT_DIR}/train_qm9_all_targets.sh"
