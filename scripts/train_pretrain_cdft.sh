#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_H5="${TRAIN_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/CDFT/RGD1_CDFT/cdft_shard_000000.h5}"
VAL_H5="${VAL_H5:-/inspire/qb-ilm/project/chemicalreaction/libowen-253207030265/dataset/CDFT/t1x_CDFT/cdft_shard_000000.h5}"
ENCODER_TYPE="${ENCODER_TYPE:-}"
OUT_DIR="${OUT_DIR:-}"
RESTART="${RESTART:-}"
INIT_CKPT="${INIT_CKPT:-}"
GEOMETRIC_OUT_DIR="${GEOMETRIC_OUT_DIR:-}"

if [[ -z "${TRAIN_H5}" ]]; then
  echo "[error] TRAIN_H5 is required."
  echo "Example: TRAIN_H5='/path/to/cdft_train1.h5 /path/to/cdft_train2.h5' bash scripts/train_pretrain_cdft.sh"
  exit 1
fi

read -r -a train_h5_arr <<<"${TRAIN_H5}"
if [[ "${#train_h5_arr[@]}" -eq 0 ]]; then
  echo "[error] TRAIN_H5 parsed empty."
  exit 1
fi

if [[ -z "${ENCODER_TYPE}" ]]; then
  case "${INIT_CKPT}" in
    *reacformer_se3*) ENCODER_TYPE="reacformer_se3" ;;
    *reacformer_so2*) ENCODER_TYPE="reacformer_so2" ;;
    *) ENCODER_TYPE="single_mol" ;;
  esac
fi

if [[ -z "${GEOMETRIC_OUT_DIR}" ]]; then
  GEOMETRIC_OUT_DIR="runs/${ENCODER_TYPE}_geometric"
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="runs/${ENCODER_TYPE}_cdft"
fi

if [[ -z "${INIT_CKPT}" ]]; then
  if [[ -f "${GEOMETRIC_OUT_DIR}/best.pt" ]]; then
    INIT_CKPT="${GEOMETRIC_OUT_DIR}/best.pt"
  elif [[ -f "${GEOMETRIC_OUT_DIR}/latest.pt" ]]; then
    INIT_CKPT="${GEOMETRIC_OUT_DIR}/latest.pt"
  else
    echo "[error] Cannot find stage1 checkpoint under ${GEOMETRIC_OUT_DIR}."
    echo "Set INIT_CKPT explicitly, or ensure ${GEOMETRIC_OUT_DIR}/best.pt (or latest.pt) exists."
    exit 1
  fi
fi

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "[error] INIT_CKPT not found: ${INIT_CKPT}"
  exit 1
fi

cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m uni_react.train_pretrain_cdft
  --train_h5 "${train_h5_arr[@]}"
  --encoder_type "${ENCODER_TYPE}"
  --mask_ratio 0.0
  --noise_std 0.0
  --init_ckpt "${INIT_CKPT}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${VAL_H5}" ]]; then
  read -r -a val_h5_arr <<<"${VAL_H5}"
  cmd+=(--val_h5 "${val_h5_arr[@]}")
fi

if [[ -n "${RESTART}" ]]; then
  cmd+=(--restart "${RESTART}")
fi

"${cmd[@]}"
