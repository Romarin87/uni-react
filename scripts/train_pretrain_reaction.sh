#!/usr/bin/env bash
# Stage-3 reaction triplet pretraining launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_H5="${TRAIN_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/React/RGD1_react.h5}"
VAL_H5="${VAL_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/React/t1x_react.h5}"
VAL_RATIO="${VAL_RATIO:-0.1}"

OUT_DIR="${OUT_DIR:-}"
INIT_CKPT="${INIT_CKPT:-}"
ENCODER_TYPE="${ENCODER_TYPE:-}"
RESTART="${RESTART:-}"
RESTART_IGNORE_CONFIG="${RESTART_IGNORE_CONFIG:-0}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-20}"

BACKBONE_LR="${BACKBONE_LR:-2e-5}"
HEAD_LR="${HEAD_LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEACHER_MOMENTUM="${TEACHER_MOMENTUM:-0.999}"

CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
COMPLETION_WEIGHT="${COMPLETION_WEIGHT:-128.0}"
NEG_RATIO="${NEG_RATIO:-0.5}"

if [[ -z "${TRAIN_H5}" ]]; then
  echo "[error] TRAIN_H5 is required."
  echo "Example: TRAIN_H5=/path/to/train_triplets.h5 bash scripts/train_pretrain_reaction.sh"
  exit 1
fi

if [[ -z "${ENCODER_TYPE}" ]]; then
  case "${INIT_CKPT}" in
    *reacformer_se3*) ENCODER_TYPE="reacformer_se3" ;;
    *reacformer_so2*) ENCODER_TYPE="reacformer_so2" ;;
    *) ENCODER_TYPE="single_mol" ;;
  esac
fi

if [[ -z "${INIT_CKPT}" ]]; then
  case "${ENCODER_TYPE}" in
    reacformer_se3) INIT_CKPT="runs/reacformer_se3_cdft/best.pt" ;;
    reacformer_so2) INIT_CKPT="runs/reacformer_so2_cdft/best.pt" ;;
    *) INIT_CKPT="runs/single_mol_cdft/best.pt" ;;
  esac
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="runs/${ENCODER_TYPE}_reaction"
fi

cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m uni_react.train_pretrain_reaction
  --train_h5 "${TRAIN_H5}"
  --encoder_type "${ENCODER_TYPE}"
  --val_ratio "${VAL_RATIO}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --backbone_lr "${BACKBONE_LR}"
  --head_lr "${HEAD_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --teacher_momentum "${TEACHER_MOMENTUM}"
  --consistency_weight "${CONSISTENCY_WEIGHT}"
  --completion_weight "${COMPLETION_WEIGHT}"
  --neg_ratio "${NEG_RATIO}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${VAL_H5}" ]]; then
  cmd+=(--val_h5 "${VAL_H5}")
fi

if [[ -n "${INIT_CKPT}" ]]; then
  cmd+=(--init_ckpt "${INIT_CKPT}")
fi

if [[ -n "${RESTART}" ]]; then
  cmd+=(--restart "${RESTART}")
fi

if [[ "${RESTART_IGNORE_CONFIG}" == "1" ]]; then
  cmd+=(--restart_ignore_config)
fi

"${cmd[@]}"
