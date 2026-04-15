#!/usr/bin/env bash

# Shared R2-specialized defaults for QM9 finetuning wrappers.
# Each value can still be overridden by environment variables.
export R2_BATCH_SIZE="${R2_BATCH_SIZE:-16}"
export R2_NUM_WORKERS="${R2_NUM_WORKERS:-16}"
export R2_EPOCHS="${R2_EPOCHS:-300}"
export R2_SAVE_EVERY="${R2_SAVE_EVERY:-10}"
export R2_BACKBONE_LR="${R2_BACKBONE_LR:-2e-5}"
export R2_HEAD_LR="${R2_HEAD_LR:-2e-4}"
export R2_WEIGHT_DECAY="${R2_WEIGHT_DECAY:-0.0}"
export R2_CUTOFF="${R2_CUTOFF:-8.0}"
export R2_HEAD_DROPOUT="${R2_HEAD_DROPOUT:-0.0}"
